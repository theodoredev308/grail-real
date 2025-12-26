#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
import zstandard as zstd

from grail.infrastructure.checkpoint_consumer import (
    CheckpointManager,
    default_checkpoint_cache_root,
)
from grail.infrastructure.comms import download_file_chunked
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


def _delta_cache_root(checkpoint_cache_root: Path) -> Path:
    # default_checkpoint_cache_root() => ~/.cache/grail/checkpoints
    # we store deltas alongside it => ~/.cache/grail/deltas
    return checkpoint_cache_root.parent / "deltas"


async def _cache_delta_window(
    *,
    window: int,
    prev_window: int,
    anchor_window: int | None,
    delta_prefix: str,
    credentials: Any,
    checkpoint_cache_root: Path,
) -> Path | None:
    """Download DELTA artifacts for `window` into a local delta cache directory.

    This avoids reconstructing a full model.safetensors on disk. Workers will apply
    the delta in-place to their already-loaded model.
    """
    delta_root = _delta_cache_root(checkpoint_cache_root)
    delta_root.mkdir(parents=True, exist_ok=True)

    final_dir = delta_root / f"delta-{window}"
    if final_dir.exists():
        return final_dir
    tmp_dir = delta_root / f"delta-{window}.partial"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write minimal metadata needed by workers.
        meta = {
            "window": window,
            "checkpoint_type": "DELTA",
            "prev_window": prev_window,
            "anchor_window": anchor_window,
        }
        (tmp_dir / "metadata.json").write_text(
            __import__("json").dumps(meta, indent=2), encoding="utf-8"
        )

        # Download delta files (small)
        delta_meta_key = f"{delta_prefix}/delta_metadata.json"
        delta_zst_key = f"{delta_prefix}/delta_sparse.safetensors.zst"

        delta_meta = await download_file_chunked(
            delta_meta_key, credentials=credentials, use_write=False
        )
        if delta_meta is None:
            raise RuntimeError(f"Missing {delta_meta_key}")
        (tmp_dir / "delta_metadata.json").write_bytes(delta_meta)

        delta_zst = await download_file_chunked(
            delta_zst_key, credentials=credentials, use_write=False
        )
        if delta_zst is None:
            raise RuntimeError(f"Missing {delta_zst_key}")
        (tmp_dir / "delta_sparse.safetensors.zst").write_bytes(delta_zst)

        # Decompress once per node so workers can load safetensors directly.
        decompressor = zstd.ZstdDecompressor()
        decompressed = decompressor.decompress(delta_zst)
        (tmp_dir / "delta_sparse.safetensors").write_bytes(decompressed)

        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        shutil.move(str(tmp_dir), str(final_dir))
        return final_dir
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


async def _ensure_delta_chain_cached(
    *,
    target_window: int,
    mgr: CheckpointManager,
    credentials: Any,
    checkpoint_cache_root: Path,
) -> None:
    """Ensure we have enough local artifacts for workers to update without disk reconstruction.

    - Ensures anchor FULL checkpoint is present locally (for cold starts)
    - Caches DELTA artifacts ONLY from anchor -> target_window (not the full history)
    
    IMPORTANT: This function now only caches deltas from the immediate anchor to
    the target window. Previously it would walk backwards through the entire history
    (100+ deltas) because anchor windows have both FULL and DELTA checkpoints, and
    _fetch_metadata prefers DELTA. Now we explicitly stop at the anchor window.
    """
    from grail.shared.retention_utils import is_anchor_window
    
    meta = await mgr._fetch_metadata(target_window)  # noqa: SLF001
    if meta is None or not meta.is_delta() or meta.prev_window is None:
        return

    anchor = meta.anchor_window
    if anchor is None:
        logging.warning("DELTA checkpoint %s has no anchor_window set; skipping delta cache", target_window)
        return
        
    # Ensure anchor FULL checkpoint is present locally (for cold starts)
    anchor_dir = checkpoint_cache_root / f"checkpoint-{int(anchor)}"
    if not anchor_dir.exists():
        logging.info("Downloading anchor FULL checkpoint %s for delta chain", anchor)
        await mgr.get_checkpoint(int(anchor))

    # Walk backwards caching deltas ONLY until we reach the anchor.
    # We stop at the anchor window because that's where we have a FULL checkpoint.
    seen: set[int] = set()
    cur = meta
    while cur is not None and cur.is_delta() and cur.prev_window is not None:
        w = int(cur.window)
        if w in seen:
            break
        seen.add(w)
        
        # Stop if we've reached the anchor window (don't cache deltas beyond it)
        if w == anchor:
            logging.debug("Reached anchor window %s, stopping delta cache walk", anchor)
            break
            
        # Also stop if prev_window IS the anchor (we're at the first delta after anchor)
        if int(cur.prev_window) == anchor:
            await _cache_delta_window(
                window=w,
                prev_window=int(cur.prev_window),
                anchor_window=cur.anchor_window,
                delta_prefix=cur.remote_prefix(),
                credentials=credentials,
                checkpoint_cache_root=checkpoint_cache_root,
            )
            logging.debug("Cached final delta %s -> anchor %s, stopping", w, anchor)
            break

        await _cache_delta_window(
            window=w,
            prev_window=int(cur.prev_window),
            anchor_window=cur.anchor_window,
            delta_prefix=cur.remote_prefix(),
            credentials=credentials,
            checkpoint_cache_root=checkpoint_cache_root,
        )
        
        # Stop if we hit an anchor window during the walk (shouldn't happen with
        # proper anchor_window tracking, but safety check)
        if is_anchor_window(int(cur.prev_window)):
            logging.debug("Hit anchor window %s during walk, stopping", cur.prev_window)
            break

        # Continue walking back toward the anchor.
        cur = await mgr._fetch_metadata(int(cur.prev_window))  # noqa: SLF001
    
    logging.info("Delta chain cached: anchor=%s -> target=%s (%d deltas)", anchor, target_window, len(seen))


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
    checkpoint_cache_root = default_checkpoint_cache_root()

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
            # Prefer DELTA caching (fast) and avoid full checkpoint reconstruction.
            # Workers will apply deltas in-place to their already-loaded models.
            metadata = await mgr._fetch_metadata(ckpt_window)  # noqa: SLF001 (intentional for internal reuse)
            if metadata is not None and metadata.is_delta() and metadata.prev_window is not None:
                await _ensure_delta_chain_cached(
                    target_window=ckpt_window,
                    mgr=mgr,
                    credentials=creds,
                    checkpoint_cache_root=checkpoint_cache_root,
                )
                try:
                    await mgr.cleanup_local(window_start)
                except Exception:
                    pass
                last_ckpt_window = ckpt_window
                logging.info("Delta ready at %s", _delta_cache_root(checkpoint_cache_root) / f"delta-{ckpt_window}")
                await asyncio.sleep(1.0)
                continue

            # FULL (or unknown): fall back to the original behavior.
            path = await mgr.get_checkpoint(ckpt_window)
            if path is None:
                logging.warning(
                    "Checkpoint not ready yet for ckpt_window=%s; will retry automatically", ckpt_window
                )
                await asyncio.sleep(3.0)
                continue

            try:
                await mgr.cleanup_local(window_start)
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


