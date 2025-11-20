#!/usr/bin/env python3
"""
Debug script to compare tokenizer behavior between worker and validator
"""
import sys
sys.path.insert(0, "/root/grail")

from grail.model.provider import get_tokenizer
from grail.environments.registry import get_adapter
from grail.shared.constants import CURRENT_ENV_ID

# Load tokenizer from checkpoint (same as worker/validator)
checkpoint_path = "/root/.cache/grail/checkpoints/checkpoint-6885270"
tokenizer = get_tokenizer(checkpoint_path)

print("=" * 80)
print("TOKENIZER DEBUG")
print("=" * 80)
print(f"Checkpoint: {checkpoint_path}")
print(f"Tokenizer class: {tokenizer.__class__.__name__}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"Has chat_template attr: {hasattr(tokenizer, 'chat_template')}")

if hasattr(tokenizer, "chat_template"):
    template = tokenizer.chat_template
    print(f"\nChat template length: {len(template) if template else 0}")
    if template:
        print(f"Chat template preview (first 200 chars):\n{template[:200]}")
        print(f"\nChat template end (last 200 chars):\n{template[-200:]}")

# Test with the exact seed from the error
seed = 4216992985

print("\n" + "=" * 80)
print(f"TESTING WITH SEED: {seed}")
print("=" * 80)

# Get adapter for current environment
adapter = get_adapter(CURRENT_ENV_ID)
print(f"Environment: {CURRENT_ENV_ID}")
print(f"Adapter: {adapter.__class__.__name__}")

# Build prompt IDs (same as validator does)
prompt_ids = adapter.build_prompt_ids(seed, tokenizer)
print(f"\nPrompt length: {len(prompt_ids)} tokens")

# Decode and show
prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
print(f"\nPrompt text length: {len(prompt_text)} chars")
print(f"\nFirst 300 chars of prompt:\n{prompt_text[:300]}")
print(f"\nLast 200 chars of prompt:\n{prompt_text[-200:]}")

print(f"\nFirst 10 token IDs: {prompt_ids[:10]}")
print(f"Last 10 token IDs: {prompt_ids[-10:]}")

