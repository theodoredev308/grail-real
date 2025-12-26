#!/usr/bin/env python3
"""
GRAIL Worker - Redis Queue Consumer
Follows the clean architecture from grail/neurons/miner.py
Processes jobs from Redis queue using the default miner's generation pipeline.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import bittensor as bt
import orjson as json
import redis.asyncio as aioredis
import torch
from safetensors.torch import load_file

from grail.cli.mine import (
    MiningTimers,
    get_window_randomness,
    package_rollout_data,
)
from grail.environments.loop import AgentEnvLoop
from grail.environments.factory import create_env
from grail.grail import derive_env_seed
from grail.infrastructure.checkpoint_consumer import default_checkpoint_cache_root
from grail.model.provider import clear_model_and_tokenizer, get_model, get_tokenizer
from grail.shared.chat_templates import build_qwen_chat_template
from grail.shared.constants import (
    CHALLENGE_K,
    CURRENT_ENV_ID,
    ROLLOUTS_PER_PROBLEM,
    WINDOW_LENGTH,
)
from grail.shared.prompt_constants import SYSTEM_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger("worker")


class WorkerState:
    """Clean state management for worker resources."""

    def __init__(self, redis_client: aioredis.Redis | None = None):
        # Decide device automatically.
        # With per-process CUDA_VISIBLE_DEVICES set, "cuda" will map to the
        # single visible GPU for this worker. If no GPU is available, fall
        # back to CPU.
        env_device = os.getenv("GRAIL_DEVICE")
        if env_device is not None:
            self.device = env_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.current_checkpoint_window: int | None = None
        self.agent_loop: AgentEnvLoop | None = None
        self.timers = MiningTimers()
        self.loop_batch_size: int | None = None
        self.redis_client = redis_client
        
    async def ensure_checkpoint_loaded(
        self,
        window_start: int,
        checkpoint_credentials: dict,
        checkpoint_window: int | None = None,
    ) -> bool:
        """Load checkpoint for the target window.
        
        Workers WAIT for checkpoints to be downloaded by download_manager.py
        instead of downloading themselves (more efficient, avoids race conditions).

        Args:
            window_start: Window start block
            checkpoint_credentials: Credentials from job payload
            checkpoint_window: Optional explicit checkpoint window to use. If None,
                defaults to previous-window heuristic (window_start - WINDOW_LENGTH).

        Returns:
            True if checkpoint loaded successfully
        """
        # Determine checkpoint window:
        # - Prefer explicit checkpoint_window from job payload (matches miner)
        # - Fallback to previous-window heuristic for backward compatibility
        if checkpoint_window is None:
            checkpoint_window = max(0, window_start - WINDOW_LENGTH)

        # Check if already loaded
        if self.current_checkpoint_window == checkpoint_window and self.model is not None:
            return True

        try:
            cache_root = default_checkpoint_cache_root()
            checkpoint_dir = cache_root / f"checkpoint-{checkpoint_window}"
            delta_root = cache_root.parent / "deltas"
            delta_dir = delta_root / f"delta-{checkpoint_window}"
            
            # DEBUG: Log the exact path being checked
            logger.info(f"[DEBUG] cache_root={cache_root}, checkpoint_dir={checkpoint_dir}")
            logger.info(f"[DEBUG] checkpoint_dir type={type(checkpoint_dir)}, exists={checkpoint_dir.exists()}")
            
            # Prefer applying locally cached DELTAs in-place (no per-worker downloads,
            # no multi-GB checkpoint reconstruction on disk).
            logger.info(f"⏳ Waiting for local delta/checkpoint for {checkpoint_window}...")
            
            max_wait_seconds = 370  # 6 minutes + 10 seconds
            check_interval = 2.0
            waited = 0.0
            
            checkpoint_path = None
            while waited < max_wait_seconds:
                if self.redis_client is not None:
                    try:
                        active_str = await self.redis_client.get("grail:active_window")
                        if active_str is not None:
                            active_window = int(active_str)
                            if active_window > window_start:
                                logger.warning(
                                    f"Active window advanced to {active_window} while waiting for "
                                    f"checkpoint {checkpoint_window}; abandoning job for window {window_start}"
                                )
                                return False
                    except Exception as e:
                        logger.warning(f"Error checking active window: {e}")

                # DELTA path: wait for delta cache dir then apply in-place
                if delta_dir.exists() and delta_dir.is_dir():
                    try:
                        meta_path = delta_dir / "metadata.json"
                        delta_meta_path = delta_dir / "delta_metadata.json"
                        delta_sparse_path = delta_dir / "delta_sparse.safetensors"
                        if meta_path.exists() and delta_meta_path.exists() and delta_sparse_path.exists():
                            meta = json.loads(meta_path.read_bytes())
                            prev_window = int(meta.get("prev_window"))

                            # Ensure we have the base checkpoint loaded (usually already the prev window).
                            if self.model is None or self.current_checkpoint_window != prev_window:
                                base_dir = cache_root / f"checkpoint-{prev_window}"
                                if not base_dir.exists():
                                    # Fall back to waiting for full checkpoint if base isn't present yet.
                                    raise FileNotFoundError(str(base_dir))

                                logger.info(
                                    f"🔁 Loading base checkpoint for window {prev_window} from {base_dir}"
                                )
                                self.model, self.tokenizer = clear_model_and_tokenizer(self.model, self.tokenizer)
                                self.model = get_model(str(base_dir), device=None, eval_mode=True).to(self.device)
                                self.tokenizer = get_tokenizer(str(base_dir))

                                # Apply canonical chat template (same as slow path)
                                try:
                                    canonical_template = build_qwen_chat_template(SYSTEM_PROMPT)
                                    if hasattr(self.tokenizer, "chat_template"):
                                        current_template = getattr(self.tokenizer, "chat_template", None)
                                        if not current_template or current_template != canonical_template:
                                            setattr(self.tokenizer, "chat_template", canonical_template)
                                except Exception:
                                    pass

                                self.current_checkpoint_window = prev_window
                                self.loop_batch_size = min(
                                    int(os.getenv("GRAIL_GENERATION_BATCH_SIZE", "16")),
                                    ROLLOUTS_PER_PROBLEM,
                                )
                                self.agent_loop = AgentEnvLoop(self.model, self.tokenizer, self.model.device)

                            # Apply delta in-place on the model weights (GPU scatter updates).
                            if self.model is None:
                                raise RuntimeError("Model not loaded")

                            t0 = time.time()
                            sparse_tensors = load_file(str(delta_sparse_path))
                            with torch.no_grad():
                                sd = self.model.state_dict()
                                applied = 0
                                for name, tensor in sd.items():
                                    ik = f"{name}.indices"
                                    vk = f"{name}.values"
                                    if ik not in sparse_tensors:
                                        continue
                                    idx = sparse_tensors[ik].to(device=tensor.device, dtype=torch.long)
                                    vals = sparse_tensors[vk].to(device=tensor.device, dtype=tensor.dtype)
                                    flat = tensor.view(-1)
                                    flat.index_copy_(0, idx, vals)
                                    applied += 1

                            self.current_checkpoint_window = checkpoint_window
                            try:
                                setattr(self.model, "grail_checkpoint_window", checkpoint_window)
                            except Exception:
                                pass

                            logger.info(
                                f"⚡ Applied delta {prev_window} → {checkpoint_window} "
                                f"(touched={applied} tensors) in {time.time() - t0:.2f}s"
                            )
                            return True
                    except Exception:
                        # If delta apply fails, fall through to slow path wait.
                        pass

                # Slow path (FULL checkpoint dir prepared by download_manager)
                if checkpoint_dir.exists() and checkpoint_dir.is_dir():
                    partial_dir = cache_root / f"checkpoint-{checkpoint_window}.partial"
                    if partial_dir.exists():
                        await asyncio.sleep(check_interval)
                        waited += check_interval
                        continue

                    monolithic_file = checkpoint_dir / "model.safetensors"
                    if monolithic_file.exists():
                        await asyncio.sleep(2.0)
                        checkpoint_path = checkpoint_dir
                        logger.info(f"✅ Checkpoint {checkpoint_window} ready after {waited:.1f}s")
                        break
                
                # Log progress every 30 seconds
                if int(waited) % 30 == 0 and waited > 0:
                    logger.info(
                        f"⏳ Still waiting for checkpoint {checkpoint_window}... ({int(waited)}s elapsed)"
                    )
                
                await asyncio.sleep(check_interval)
                waited += check_interval
            
            if checkpoint_path is None:
                logger.error(
                    f"❌ Checkpoint {checkpoint_window} not ready after {max_wait_seconds}s. "
                    f"Is download_manager.py running?"
                )
                return False

            logger.info(
                f"🔁 Loading checkpoint for window {checkpoint_window} from {checkpoint_path}"
            )

            # Clean up previous model to prevent VRAM growth
            self.model, self.tokenizer = clear_model_and_tokenizer(self.model, self.tokenizer)

            # Load new checkpoint
            # CRITICAL: Pass device=None to auto-detect and get correct dtype (bfloat16)
            # Then manually move to specific GPU to match validator's dtype
            self.model = get_model(str(checkpoint_path), device=None, eval_mode=True)
            self.model = self.model.to(self.device)
            self.tokenizer = get_tokenizer(str(checkpoint_path))

            # Ensure tokenizer uses the canonical Qwen chat template.
            # This guarantees that prompts rendered by AgentEnvLoop match
            # the validator's env-registry prompts exactly, preventing
            # env_prompt_valid failures due to template drift.
            try:
                canonical_template = build_qwen_chat_template(SYSTEM_PROMPT)
                # Not all tokenizers expose chat_template, so guard access.
                if hasattr(self.tokenizer, "chat_template"):
                    current_template = getattr(self.tokenizer, "chat_template", None)
                    if not current_template or current_template != canonical_template:
                        setattr(self.tokenizer, "chat_template", canonical_template)
                        logger.info("✅ Applied canonical Qwen chat template to tokenizer")
                else:
                    # Best-effort log; validator will still rely on its own tokenizer.
                    logger.warning(
                        "Tokenizer has no 'chat_template' attribute; "
                        "prompt alignment relies on checkpoint configuration."
                    )
            except Exception as e:
                logger.warning(f"Failed to apply canonical chat template: {e}")

            self.current_checkpoint_window = checkpoint_window

            # Initialize AgentEnvLoop for generation
            batch_size = int(os.getenv("GRAIL_GENERATION_BATCH_SIZE", "16"))
            batch_size = min(batch_size, ROLLOUTS_PER_PROBLEM)
            # Store desired batch size, but do NOT pass to AgentEnvLoop ctor (unsupported)
            self.loop_batch_size = batch_size
            self.agent_loop = AgentEnvLoop(self.model, self.tokenizer, self.model.device)
            logger.info(f"✅ Loaded checkpoint {checkpoint_window}, batch_size={batch_size} (applied at run_grpo_group)")

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint for window {checkpoint_window}: {e}")
            traceback.print_exc()
            return False

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.model, self.tokenizer = clear_model_and_tokenizer(self.model, self.tokenizer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


async def generate_group_rollouts(
    state: WorkerState,
    wallet: bt.wallet,
    window_start: int,
    window_block_hash: str,
    randomness_hex: str,
    group_id: int,
    use_drand: bool,
) -> list[dict]:
    """Generate GRPO rollouts using the default miner's clean pipeline.

    Args:
        state: Worker state with loaded model/tokenizer
        wallet: Miner wallet
        window_start: Window start block
        window_block_hash: Block hash for window
        randomness_hex: Combined randomness
        group_id: Problem group ID
        use_drand: Whether drand was used

    Returns:
        List of packaged rollout data ready for upload
    """
    if state.model is None or state.tokenizer is None or state.agent_loop is None:
        logger.error("Model/tokenizer/agent_loop not loaded")
        return []

    try:
        # Derive environment seed deterministically (matches validator)
        seed_int = derive_env_seed(wallet.hotkey.ss58_address, window_block_hash, group_id)

        logger.debug(
            f"WORKER SEED: hotkey={wallet.hotkey.ss58_address[:12]} "
            f"hash={window_block_hash[:12]} group={group_id} -> seed={seed_int}"
        )

        # Environment factory (uses CURRENT_ENV_ID under the hood; defaults to 'math')
        def _env_factory():
            return create_env()

        # Generate GRPO rollouts using AgentEnvLoop (same as default miner)
        gen_start = time.time()
        # Respect configured batch size (capped by ROLLOUTS_PER_PROBLEM)
        configured_bs = state.loop_batch_size if state.loop_batch_size is not None else min(int(os.getenv("GRAIL_GENERATION_BATCH_SIZE", "16")), ROLLOUTS_PER_PROBLEM)
        grpo_rollouts = await asyncio.to_thread(
            state.agent_loop.run_grpo_group,
            _env_factory,
            ROLLOUTS_PER_PROBLEM,
            randomness_hex,
            wallet,
            batch_size=configured_bs,
            seed=seed_int,
        )
        gen_duration = time.time() - gen_start

        if not grpo_rollouts:
            logger.warning(f"No rollouts generated for group {group_id}")
            return []

        # Log generation stats
        successful_count = sum(1 for r in grpo_rollouts if r.success)
        mean_reward = sum(r.reward for r in grpo_rollouts) / len(grpo_rollouts)
        logger.info(
            f"✅ Group {group_id}: {successful_count}/{len(grpo_rollouts)} successful, "
            f"reward={mean_reward:.3f}, time={gen_duration:.2f}s"
        )

        # Package rollouts (matches default miner format exactly)
        current_block = window_start  # Use window_start as block for consistency
        packaged_rollouts: list[dict] = []

        # Use the checkpoint window actually loaded by this worker.
        checkpoint_window = state.current_checkpoint_window
        if checkpoint_window is None:
            logger.error(
                "Checkpoint window is not set in state while packaging rollouts for group %s",
                group_id,
            )
            return []

        for rollout_idx, rollout in enumerate(grpo_rollouts):
            rollout_data = package_rollout_data(
                state.model,
                wallet,
                rollout,
                group_id,  # base_nonce = group_id
                rollout_idx,
                len(grpo_rollouts),
                window_start,
                current_block,
                window_block_hash,
                randomness_hex,
                use_drand,
                checkpoint_window,
            )
            packaged_rollouts.append(rollout_data)

        # Update timing EMA
        state.timers.update_gen_time_ema(gen_duration)

        return packaged_rollouts

    except Exception as e:
        logger.error(f"Failed to generate rollouts for group {group_id}: {e}")
        traceback.print_exc()
        return []


async def process_job(state: WorkerState, job_data: dict) -> dict:
    """Process a single job from the queue.

    Args:
        state: Worker state
        job_data: Job payload from Redis

    Returns:
        Result payload to push back to results queue
    """
    try:
        # Extract job parameters
        wallet_cold = job_data.get("wallet_cold", "default")
        wallet_hot = job_data.get("wallet_hot", "default")
        window_start = int(job_data.get("window_start", 0))
        window_block_hash = str(job_data.get("window_block_hash", ""))
        randomness_hex = str(job_data.get("randomness_hex", ""))
        group_id = int(job_data.get("group_id", 0))
        checkpoint_credentials = job_data.get("checkpoint_credentials")
        # Optional explicit checkpoint window (preferred when provided)
        raw_ckpt_window = job_data.get("checkpoint_window")
        checkpoint_window: int | None
        try:
            checkpoint_window = int(raw_ckpt_window) if raw_ckpt_window is not None else None
        except Exception:
            checkpoint_window = None
        use_drand = bool(job_data.get("use_drand", True))

        # Handle preload requests
        if job_data.get("preload", False):
            logger.info(f"Preloading checkpoint for window {window_start}")
            success = await state.ensure_checkpoint_loaded(
                window_start,
                checkpoint_credentials,
                checkpoint_window,
            )
            if success:
                logger.info(f"✅ Preloaded checkpoint for window {window_start}")
            return {"group_id": group_id, "inferences": [], "rewards": []}

        # Load checkpoint for this window
        success = await state.ensure_checkpoint_loaded(
            window_start,
            checkpoint_credentials,
            checkpoint_window,
        )
        if not success:
            logger.error(f"Failed to load checkpoint for window {window_start}")
            return {"group_id": group_id, "inferences": [], "rewards": []}

        # Create wallet
        wallet = bt.wallet(name=wallet_cold, hotkey=wallet_hot)

        # Generate rollouts
        logger.info(f"⚡ Generating rollouts for group {group_id}, window {window_start}")
        inferences = await generate_group_rollouts(
            state,
            wallet,
            window_start,
            window_block_hash,
            randomness_hex,
            group_id,
            use_drand,
        )

        # Extract rewards
        rewards = [
            float(inf.get("commit", {}).get("rollout", {}).get("total_reward", 0.0))
            for inf in inferences
        ]

        logger.info(
            f"✅ Completed group {group_id}: {len(inferences)} rollouts, "
            f"avg_reward={sum(rewards)/len(rewards) if rewards else 0:.3f}"
        )

        return {
            "group_id": group_id,
            "inferences": inferences,
            "rewards": rewards,
        }

    except Exception as e:
        logger.error(f"Job processing failed: {e}")
        traceback.print_exc()
        return {
            "group_id": job_data.get("group_id", -1),
            "inferences": [],
            "rewards": [],
        }


async def worker_loop(
    redis_url: str,
    job_queue: str,
    result_queue: str,
    max_concurrent: int = 1,
) -> None:
    """Main worker loop - consume jobs from Redis and process them.

    Args:
        redis_url: Redis connection URL
        job_queue: Queue name for incoming jobs
        result_queue: Queue name for results
        device: Device for model (cuda/cpu)
        max_concurrent: Maximum concurrent jobs (default 1 for sequential processing)
    """

    try:
        # Connect to Redis
        redis_client = aioredis.from_url(redis_url, decode_responses=True)
        logger.info(f"✅ Connected to Redis at {redis_url}")
        # Decide device once per process for logging; WorkerState also
        # determines its own device (using the same env/heuristics).
        env_device = os.getenv("GRAIL_DEVICE")
        device = env_device if env_device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"📥 Consuming from queue: {job_queue}")
        logger.info(f"📤 Publishing to queue: {result_queue}")
        logger.info(f"🖥️  Device: {device}")

        # Initialize worker state
        state = WorkerState(redis_client=redis_client)
        logger.info(f"✅ Initialized worker state")
        logger.info(f"✅ Redis client: {redis_client}")

        while True:
            try:
                # Blocking pop with timeout
                item = await redis_client.blpop(job_queue, timeout=5)

                if item is None:
                    # No jobs available, continue
                    await asyncio.sleep(0.1)
                    continue

                _, job_payload = item

                # Parse job
                try:
                    job_data = json.loads(job_payload)
                except Exception as e:
                    logger.error(f"Failed to parse job payload: {e}")
                    continue

                # Process job
                result = await process_job(state, job_data)

                # Push result back to Redis
                try:
                    await redis_client.rpush(result_queue, json.dumps(result))
                except Exception as e:
                    logger.error(f"Failed to push result to Redis: {e}")

            except asyncio.CancelledError:
                logger.info("Worker cancelled, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    finally:
        # Cleanup
        state.cleanup()
        logger.info("Worker stopped")


def main() -> None:
    """Entry point for worker process."""
    parser = argparse.ArgumentParser(description="GRAIL Worker - Redis Queue Consumer")
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
        help="Redis connection URL",
    )
    parser.add_argument(
        "--job-queue",
        default=os.getenv("GRAIL_JOB_QUEUE", "grail:jobs"),
        help="Job queue name",
    )
    parser.add_argument(
        "--result-queue",
        default=os.getenv("GRAIL_RESULT_QUEUE", "grail:results"),
        help="Result queue name",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=int(os.getenv("GRAIL_WORKER_CONCURRENCY", "1")),
        help="Maximum concurrent jobs (default: 1 for sequential)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("GRAIL Worker Starting")
    logger.info("=" * 80)
    # For visibility, log the device decision using the same heuristic as
    # worker_loop / WorkerState.
    env_device = os.getenv("GRAIL_DEVICE")
    device = env_device if env_device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")
    logger.info(f"Redis URL: {args.redis_url}")
    logger.info(f"Job Queue: {args.job_queue}")
    logger.info(f"Result Queue: {args.result_queue}")
    logger.info(f"Max Concurrent: {args.max_concurrent}")
    logger.info(f"Current Environment: {CURRENT_ENV_ID}")
    logger.info("=" * 80)

    try:
        asyncio.run(
            worker_loop(
                redis_url=args.redis_url,
                job_queue=args.job_queue,
                result_queue=args.result_queue,
                max_concurrent=args.max_concurrent,
            )
        )
    except KeyboardInterrupt:
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
