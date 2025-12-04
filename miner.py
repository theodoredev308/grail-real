#!/usr/bin/env python3
"""
GRAIL Miner - Redis-Based Job Scheduler
Follows the clean architecture from grail/neurons/miner.py
Schedules jobs to Redis queue and collects results.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import sys
import time
from typing import Any

import bittensor as bt
import orjson as json
import redis.asyncio as aioredis

from grail.infrastructure.chain import GrailChainManager
from grail.infrastructure.comms import sink_window_inferences
from grail.infrastructure.credentials import load_r2_credentials
from grail.infrastructure.drand import get_beacon as get_drand_beacon
from grail.infrastructure.network import create_subtensor
from grail.shared.constants import ROLLOUTS_PER_PROBLEM, TRAINER_UID, WINDOW_LENGTH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger("miner")


def env(key: str, default: str | None = None) -> str:
    """Get environment variable with optional default."""
    v = os.getenv(key)
    if v is None:
        if default is None:
            raise RuntimeError(f"Missing env {key}")
        return default
    return v


async def derive_randomness(
    subtensor: bt.subtensor, window_start: int, use_drand: bool
) -> tuple[str, str]:
    """Derive randomness for window using block hash and optional drand.

    Args:
        subtensor: Bittensor subtensor client
        window_start: Window start block
        use_drand: Whether to mix in drand randomness

    Returns:
        (window_block_hash, combined_randomness)
    """
    window_block_hash = await subtensor.get_block_hash(window_start)

    if not use_drand:
        return window_block_hash, window_block_hash

    try:
        beacon = await asyncio.to_thread(get_drand_beacon, "latest", True)
        drand_randomness = beacon.get("randomness", "")
        if drand_randomness:
            combined = hashlib.sha256(
                (window_block_hash + drand_randomness).encode()
            ).hexdigest()
            logger.info(f"🎲 Using drand randomness from round {beacon.get('round')}")
            return window_block_hash, combined
    except Exception as e:
        logger.warning(f"Failed to get drand beacon: {e}")

    return window_block_hash, window_block_hash


async def schedule_window_jobs(
    redis_client: aioredis.Redis,
    job_queue: str,
    wallet: bt.wallet,
    window_start: int,
    window_block_hash: str,
    randomness_hex: str,
    trainer_creds: dict,
    max_groups: int,
    temperature: float,
    difficulty: float,
) -> int:
    """Schedule all jobs for a window to Redis queue.

    Args:
        redis_client: Redis client
        job_queue: Queue name for jobs
        wallet: Miner wallet
        window_start: Window start block
        window_block_hash: Block hash
        randomness_hex: Combined randomness
        trainer_creds: Trainer checkpoint credentials
        max_groups: Maximum number of groups to schedule
        temperature: Generation temperature
        difficulty: Problem difficulty

    Returns:
        Number of jobs scheduled
    """
    jobs_to_submit = []

    # Schedule problem groups
    for group_id in range(max_groups):
        job = {
            "wallet_cold": wallet.name,
            "wallet_hot": wallet.hotkey_str,
            "window_start": window_start,
            "window_block_hash": window_block_hash,
            "randomness_hex": randomness_hex,
            "difficulty": difficulty,
            "group_id": group_id,
            "temperature": temperature,
            "checkpoint_credentials": trainer_creds,
            "use_drand": True,
        }
        jobs_to_submit.append(json.dumps(job))

    # Submit jobs in batches
    batch_size = 1000
    for i in range(0, len(jobs_to_submit), batch_size):
        batch = jobs_to_submit[i : i + batch_size]
        pipeline = redis_client.pipeline()
        for job in batch:
            pipeline.rpush(job_queue, job)
        await pipeline.execute()

    logger.info(f"📝 Scheduled {len(jobs_to_submit)} jobs for window {window_start}")
    return len(jobs_to_submit)


async def collect_window_results(
    redis_client: aioredis.Redis,
    result_queue: str,
    window_start: int,
    upload_deadline_block: int,
    subtensor: bt.subtensor,
    min_rollouts: int,
) -> list[dict]:
    """Collect results from workers until deadline.

    Args:
        redis_client: Redis client
        result_queue: Queue name for results
        window_start: Window start block
        upload_deadline_block: Block deadline for uploads
        subtensor: Subtensor for block queries
        min_rollouts: Minimum rollouts needed to upload

    Returns:
        List of collected inferences
    """
    inferences: list[dict] = []
    groups_collected: set[int] = set()
    last_block_check = time.time()
    current_block = window_start

    logger.info(f"📥 Collecting results until block {upload_deadline_block}...")

    while True:
        # Periodically check current block
        if time.time() - last_block_check > 2.0:
            try:
                current_block = await asyncio.wait_for(
                    subtensor.get_current_block(), timeout=3
                )
                last_block_check = time.time()

                # Log progress
                blocks_remaining = upload_deadline_block - current_block
                if blocks_remaining <= 20 and blocks_remaining % 5 == 0:
                    logger.warning(
                        f"⏰ {blocks_remaining} blocks until deadline! "
                        f"Collected {len(inferences)} rollouts"
                    )

                # Check deadline
                if current_block >= upload_deadline_block:
                    logger.info(f"✅ Deadline reached at block {current_block}")
                    break
            except Exception as e:
                logger.warning(f"Failed to check block: {e}")

        # Try to collect results in batch
        results_batch = []
        pipeline = redis_client.pipeline()

        # Try to get up to 100 results without blocking
        for _ in range(100):
            pipeline.lpop(result_queue)

        try:
            popped_items = await pipeline.execute()
            results_batch = [item for item in popped_items if item is not None]
        except Exception as e:
            logger.warning(f"Pipeline error: {e}")
            results_batch = []

        # If no results, use blocking wait with short timeout
        if not results_batch:
            item = await redis_client.blpop(result_queue, timeout=0.5)
            if item:
                _, data_str = item
                results_batch = [data_str]

        # Process batch of results
        for data_str in results_batch:
            try:
                data = json.loads(data_str)
                group_id = int(data.get("group_id", -1))
                rollouts = data.get("inferences", [])

                if group_id < 0 or not isinstance(rollouts, list):
                    continue

                # Filter to current window only
                valid_rollouts = []
                for rollout in rollouts:
                    try:
                        rollout_window = int(rollout.get("window_start", -1))
                        if rollout_window == window_start:
                            valid_rollouts.append(rollout)
                    except Exception:
                        continue

                # Accept full groups only
                if (
                    len(valid_rollouts) >= ROLLOUTS_PER_PROBLEM
                    and group_id not in groups_collected
                ):
                    # Sort by rollout_index to maintain order
                    valid_rollouts.sort(key=lambda x: int(x.get("rollout_index", 0)))
                    inferences.extend(valid_rollouts[:ROLLOUTS_PER_PROBLEM])
                    groups_collected.add(group_id)

            except Exception as e:
                logger.debug(f"Failed to process result: {e}")
                continue

        # Log progress periodically
        if len(inferences) > 0 and len(inferences) % 400 == 0:
            logger.info(f"📊 Collected {len(inferences)} rollouts so far...")

    logger.info(
        f"✅ Collection complete: {len(inferences)} rollouts from "
        f"{len(groups_collected)} groups"
    )
    return inferences


async def upload_window_inferences(
    wallet: bt.wallet,
    window_start: int,
    inferences: list[dict],
    credentials: Any,
) -> None:
    """Upload inferences to R2 storage.

    Args:
        wallet: Miner wallet
        window_start: Window start block
        inferences: List of rollout data
        credentials: R2 credentials
    """
    if not inferences:
        logger.warning(f"No inferences to upload for window {window_start}")
        return

    # Ensure contiguous group ordering (gid 0, 1, 2, ...)
    groups: dict[int, list[dict]] = {}
    for inf in inferences:
        try:
            gid = int(inf.get("rollout_group", -1))
        except Exception:
            gid = -1
        if gid < 0:
            continue
        groups.setdefault(gid, []).append(inf)

    # Sort each group by rollout_index
    for gid, rollouts in groups.items():
        groups[gid] = sorted(rollouts, key=lambda x: int(x.get("rollout_index", 0)))

    # Find maximum contiguous prefix from gid 0
    contiguous = 0
    while contiguous in groups:
        contiguous += 1

    # Build final ordered list
    ordered_inferences: list[dict] = []
    for gid in range(contiguous):
        ordered_inferences.extend(groups.get(gid, []))

    if not ordered_inferences:
        logger.warning(f"No contiguous groups for window {window_start}")
        ordered_inferences = inferences  # Fallback to all inferences

    # Enforce nonce uniqueness: stop at first duplicate and upload only prior rollouts
    seen_nonce_to_index: dict[int, int] = {}
    first_dup_idx: int | None = None
    dup_nonce_val: int | None = None
    for i, inf in enumerate(ordered_inferences):
        try:
            nonce_val = int(inf.get("nonce"))  # type: ignore[arg-type]
        except Exception:
            nonce_val = None  # Treat malformed nonce as duplication trigger
        if not isinstance(nonce_val, int):
            first_dup_idx = i
            dup_nonce_val = None
            break
        prev = seen_nonce_to_index.get(nonce_val)
        if prev is not None:
            first_dup_idx = i
            dup_nonce_val = nonce_val
            break
        seen_nonce_to_index[nonce_val] = i

    if first_dup_idx is not None:
        kept_inferences = ordered_inferences[:first_dup_idx]
        # Compute helpful diagnostics
        bad_group = None
        try:
            bad_group = int(ordered_inferences[first_dup_idx].get("rollout_group", -1))  # type: ignore[arg-type]
        except Exception:
            bad_group = None
        last_kept_group = None
        if kept_inferences:
            try:
                last_kept_group = int(kept_inferences[-1].get("rollout_group", -1))  # type: ignore[arg-type]
            except Exception:
                last_kept_group = None
        logger.warning(
            "Duplicate or malformed nonce detected at index %s (nonce=%s, rollout_group=%s). "
            "Uploading only the first %s rollouts (through rollout_group=%s) and skipping %s trailing rollouts.",
            first_dup_idx,
            dup_nonce_val,
            bad_group,
            len(kept_inferences),
            last_kept_group,
            len(ordered_inferences) - first_dup_idx,
        )
        ordered_inferences = kept_inferences
        if not ordered_inferences:
            logger.error(
                "All rollouts were dropped due to nonce duplication/malformation; skipping upload for window %s",
                window_start,
            )
            return

    logger.info(
        f"📤 Uploading {len(ordered_inferences)} rollouts for window {window_start}..."
    )

    try:
        await sink_window_inferences(wallet, window_start, ordered_inferences, credentials)
        logger.info(
            f"✅ Successfully uploaded window {window_start} with {len(ordered_inferences)} rollouts"
        )
    except Exception as e:
        logger.error(f"❌ Upload failed for window {window_start}: {e}")
        raise


async def run(args: argparse.Namespace) -> None:
    """Main miner loop."""
    # Initialize wallet
    wallet = bt.wallet(
        name=env("BT_WALLET_COLD", "default"), hotkey=env("BT_WALLET_HOT", "default")
    )
    logger.info(f"🔑 Hotkey: {wallet.hotkey.ss58_address}")

    # Redis configuration
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    job_queue = os.getenv("GRAIL_JOB_QUEUE", "grail:jobs")
    result_queue = os.getenv("GRAIL_RESULT_QUEUE", "grail:results")
    redis_client = aioredis.from_url(redis_url, decode_responses=True)
    logger.info(f"✅ Connected to Redis at {redis_url}")

    # Load credentials and initialize chain manager
    credentials = load_r2_credentials()
    netuid = int(os.getenv("NETUID", "81"))
    subtensor = await create_subtensor()
    metagraph = await subtensor.metagraph(netuid)

    from types import SimpleNamespace

    cfg = SimpleNamespace(netuid=netuid)
    chain = GrailChainManager(cfg, wallet, metagraph, subtensor, credentials)
    await chain.initialize()
    logger.info("✅ Initialized chain manager")

    # Get trainer checkpoint credentials
    trainer_bucket = chain.get_bucket(TRAINER_UID)
    trainer_creds = {
        "account_id": trainer_bucket.account_id,
        "access_key_id": trainer_bucket.access_key_id,
        "secret_access_key": trainer_bucket.secret_access_key,
        "name": trainer_bucket.name,
    }
    logger.info(f"✅ Using trainer UID {TRAINER_UID} bucket for checkpoints")

    # Configuration
    max_groups_per_window = int(os.getenv("GRAIL_MAX_GROUPS_PER_WINDOW", "2560"))
    temperature = float(os.getenv("GRAIL_TEMPERATURE", "0.7"))
    upload_threshold = int(os.getenv("GRAIL_UPLOAD_THRESHOLD", "64"))
    use_drand_flag = os.getenv("GRAIL_USE_DRAND", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )
    safety_blocks = int(os.getenv("GRAIL_MINER_SAFETY_BLOCKS", "2"))

    last_window = -1

    logger.info("=" * 80)
    logger.info("GRAIL Miner Started")
    logger.info("=" * 80)
    logger.info(f"Max Groups: {max_groups_per_window}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Upload Threshold: {upload_threshold}")
    logger.info(f"Use Drand: {use_drand_flag}")
    logger.info(f"Safety Blocks: {safety_blocks}")
    logger.info("=" * 80)

    while True:
        try:
            # Get current block
            try:
                current_block = await asyncio.wait_for(
                    subtensor.get_current_block(), timeout=3
                )
            except Exception as e:
                logger.warning(f"Failed to get current block: {e}")
                await asyncio.sleep(2)
                continue

            window_start = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH

            # Update current block in Redis for monitoring
            try:
                await redis_client.set("grail:current_block", str(current_block))
            except Exception:
                pass

            # Wait for new window
            if window_start <= last_window:
                await asyncio.sleep(0.5)
                continue

            # New window detected
            logger.info("=" * 80)
            logger.info(f"🔥 NEW WINDOW: {window_start}")
            logger.info("=" * 80)

            # Derive randomness
            window_block_hash, randomness = await derive_randomness(
                subtensor, window_start, use_drand_flag
            )
            logger.info(
                f"🎲 Window {window_start}: hash={window_block_hash[:12]}... "
                f"rand={randomness[:12]}..."
            )

            # Clear old jobs/results
            try:
                await redis_client.delete(job_queue)
                await redis_client.delete(result_queue)
                await redis_client.set("grail:active_window", str(window_start), ex=3600)
                logger.info("🧹 Cleared old jobs and results")
            except Exception as e:
                logger.warning(f"Failed to clear queues: {e}")

            # Schedule jobs
            difficulty = float(os.getenv("GRAIL_BASE_DIFFICULTY", "0.5"))
            num_jobs = await schedule_window_jobs(
                redis_client,
                job_queue,
                wallet,
                window_start,
                window_block_hash,
                randomness,
                trainer_creds,
                max_groups_per_window,
                temperature,
                difficulty,
            )

            # Calculate upload deadline
            upload_deadline_block = window_start + WINDOW_LENGTH - safety_blocks
            logger.info(f"⏰ Upload deadline: block {upload_deadline_block}")

            # Collect results
            inferences = await collect_window_results(
                redis_client,
                result_queue,
                window_start,
                upload_deadline_block,
                subtensor,
                upload_threshold,
            )

            # Upload inferences in background
            async def upload_task():
                try:
                    if len(inferences) >= upload_threshold:
                        await upload_window_inferences(
                            wallet, window_start, inferences, credentials
                        )
                    else:
                        logger.warning(
                            f"Insufficient rollouts for window {window_start}: "
                            f"{len(inferences)} < {upload_threshold}"
                        )
                except Exception as e:
                    logger.error(f"Upload error for window {window_start}: {e}")

            # Launch upload as background task
            asyncio.create_task(upload_task())

            # Clear queues for next window
            await redis_client.delete(job_queue)
            await redis_client.delete(result_queue)

            last_window = window_start
            logger.info(f"✅ Window {window_start} complete, continuing to next...")

        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
            break
        except Exception as e:
            logger.error(f"Miner loop error: {e}")
            import traceback

            traceback.print_exc()
            await asyncio.sleep(2)
            continue


def main() -> None:
    """Entry point for miner process."""
    parser = argparse.ArgumentParser(description="GRAIL Miner - Redis Job Scheduler")
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
        sys.exit(0)


if __name__ == "__main__":
    main()
