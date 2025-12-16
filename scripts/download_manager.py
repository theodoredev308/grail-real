#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import redis.asyncio as aioredis

from grail.infrastructure.checkpoint_consumer import (
    CheckpointManager,
    default_checkpoint_cache_root,
)
from grail.infrastructure.credentials import load_r2_credentials
from grail.infrastructure.chain import GrailChainManager
from grail.infrastructure.network import create_subtensor
from grail.infrastructure.comms import clear_client_cache
from grail.shared.constants import TRAINER_UID, NETUID
from grail.shared.constants import WINDOW_LENGTH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [dl-manager] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


async def _get_active_window(r: aioredis.Redis) -> int | None:
    try:
        s = await r.get("grail:active_window")
        return int(s) if s is not None else None
    except Exception:
        return None


async def _get_checkpoint_window(r: aioredis.Redis) -> int | None:
    try:
        s = await r.get("grail:checkpoint_window")
        return int(s) if s is not None else None
    except Exception:
        return None


async def _resolve_checkpoint_credentials() -> Any:
    # Prefer trainer bucket from chain (matches default miner behavior)
    use_trainer = str(os.getenv("USE_TRAINER_BUCKET", "1")).lower() in ("1", "true", "yes", "on")
    if use_trainer:
        try:
            import bittensor as bt  # type: ignore

            wallet = bt.wallet(
                name=os.getenv("BT_WALLET_COLD", "default"),
                hotkey=os.getenv("BT_WALLET_HOT", "default"),
            )
            netuid = int(os.getenv("NETUID", str(NETUID)))
            subtensor = await create_subtensor()
            metagraph = await subtensor.metagraph(netuid)
            chain = GrailChainManager(
                type("Cfg", (), {"netuid": netuid})(),
                wallet,
                metagraph,
                subtensor,
                load_r2_credentials(),
            )
            await chain.initialize()
            trainer_bucket = chain.get_bucket(TRAINER_UID)
            if trainer_bucket is not None:
                return trainer_bucket
        except Exception as e:
            logging.warning("Trainer bucket resolution failed: %s; falling back to local creds", e)

    try:
        creds = load_r2_credentials()
        return creds.get_read_dict()
    except Exception:
        logging.warning("Failed to load R2 credentials; set GRAIL_* env")
        return {
            "name": os.getenv("GRAIL_CKPT_BUCKET", ""),
            "account_id": os.getenv("GRAIL_CKPT_ACCOUNT_ID", ""),
            "access_key_id": os.getenv("GRAIL_CKPT_ACCESS_KEY", ""),
            "secret_access_key": os.getenv("GRAIL_CKPT_SECRET_KEY", ""),
        }


async def main() -> None:
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    r = aioredis.from_url(redis_url, decode_responses=True)
    keep_limit = int(os.getenv("GRAIL_CKPT_KEEP", "3"))

    creds = await _resolve_checkpoint_credentials()
    mgr = CheckpointManager(
        cache_root=default_checkpoint_cache_root(),
        credentials=creds,
        keep_limit=keep_limit,
    )

    last_ckpt_window: int | None = None
    logging.info("Starting download manager (redis=%s, keep=%d)", redis_url, keep_limit)

    while True:
        try:
            window_start = await _get_active_window(r)
            if window_start is None:
                await asyncio.sleep(1.0)
                continue
            ckpt_window = await _get_checkpoint_window(r)
            if ckpt_window is None:
                logging.warning(
                    "No checkpoint_window set in Redis for active window=%s; waiting for miner...",
                    window_start,
                )
                await asyncio.sleep(1.0)
                continue

            if last_ckpt_window is not None and ckpt_window == last_ckpt_window:
                await asyncio.sleep(1.0)
                continue

            logging.info("Ensuring checkpoint for ckpt_window=%s (window=%s)", ckpt_window, window_start)
            try:
                await clear_client_cache()
            except Exception:
                pass
            # Fast path: try READY-based check immediately so we hit remote quickly
            path = await mgr.get_checkpoint(ckpt_window)
            if path is not None:
                try:
                    await mgr.cleanup_local(window_start, keep_extra={ckpt_window})
                except Exception:
                    pass
                last_ckpt_window = ckpt_window
                logging.info("Checkpoint ready at %s", path)
                await asyncio.sleep(1.0)
                continue

            # Preflight (bounded): probe manifest/shards briefly, then fall back to READY
            try:
                from grail.infrastructure.comms import get_file, get_file_size

                def _index_key() -> str:
                    return f"grail/checkpoints/checkpoint-{ckpt_window}/model.safetensors.index.json"

                async def _collect_required_files() -> set[str]:
                    # Prefer index.json (gz fallback handled by get_file())
                    idx = await get_file(_index_key(), credentials=creds, use_write=False)
                    required: set[str] = set()
                    if isinstance(idx, dict):
                        weight_map = idx.get("weight_map") or {}
                        if isinstance(weight_map, dict):
                            for fname in weight_map.values():
                                if isinstance(fname, str) and fname:
                                    required.add(fname)
                    return required

                async def _all_sizes_positive(files: set[str]) -> bool:
                    if not files:
                        return False
                    for fname in files:
                        size = await get_file_size(
                            f"grail/checkpoints/checkpoint-{ckpt_window}/{fname}",
                            credentials=creds,
                            use_write=False,
                        )
                        if size is None or int(size) <= 0:
                            return False
                    return True

                # Bound preflight duration by env (default 30s). Set 0 to skip preflight.
                preflight_max_s = max(0, int(os.getenv("GRAIL_CKPT_PREFLIGHT_MAX_S", "30")))
                attempts = max(0, preflight_max_s // 3)

                shards: set[str] = set()
                if attempts == 0:
                    logging.info(
                        "Skipping preflight (GRAIL_CKPT_PREFLIGHT_MAX_S=%s); falling back to READY check for ckpt_window=%s",
                        preflight_max_s,
                        ckpt_window,
                    )
                else:
                    # Wait for index.json to appear and list all shards
                    for i in range(attempts):
                        shards = await _collect_required_files()
                        if shards:
                            break
                        if i % 3 == 0:
                            logging.info("Waiting for shard index for ckpt_window=%s...", ckpt_window)
                        # Abort early if active window advanced
                        aw = await _get_active_window(r)
                        if aw is not None and int(aw) != int(window_start):
                            logging.info(
                                "Active window advanced to %s; stopping preflight for ckpt_window=%s",
                                aw,
                                ckpt_window,
                            )
                            shards = set()
                            break
                        await asyncio.sleep(3.0)
                    if not shards:
                        logging.warning("Shard index not yet available for ckpt_window=%s", ckpt_window)
                        logging.info(
                            "Bypassing preflight; falling back to READY check for ckpt_window=%s",
                            ckpt_window,
                        )

                # Require shard sizes to be positive for two consecutive checks
                stable = False
                if shards:
                    for _ in range(max(1, attempts)):
                        ok1 = await _all_sizes_positive(shards)
                        if not ok1:
                            await asyncio.sleep(3.0)
                            continue
                        await asyncio.sleep(3.0)
                        ok2 = await _all_sizes_positive(shards)
                        if ok2:
                            stable = True
                            break
                    if not stable:
                        logging.warning("Shard files not yet fully published for ckpt_window=%s", ckpt_window)
                        logging.info(
                            "Bypassing preflight; falling back to READY check for ckpt_window=%s",
                            ckpt_window,
                        )
            except Exception:
                # If preflight fails, fall through to get_checkpoint which also verifies integrity
                pass
            path = await mgr.get_checkpoint(ckpt_window)
            if path is None:
                logging.warning("READY not found yet for ckpt_window=%s; will retry automatically", ckpt_window)
                await asyncio.sleep(3.0)
                continue

            try:
                await mgr.cleanup_local(window_start, keep_extra={ckpt_window})
            except Exception:
                pass

            last_ckpt_window = ckpt_window
            logging.info("Checkpoint ready at %s", path)
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.warning("download loop error: %s", e)
            await asyncio.sleep(2.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


