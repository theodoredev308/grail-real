#!/usr/bin/env python3
"""
Test what the BASE MODEL tokenizer generates (not checkpoint)
"""
import sys
sys.path.insert(0, "/root/grail")

from grail.model.provider import get_tokenizer
from grail.environments.registry import get_adapter

# Load BASE MODEL tokenizer (not checkpoint)
base_model = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading tokenizer from BASE MODEL: {base_model}")
tokenizer = get_tokenizer(base_model)

print(f"\nTokenizer class: {tokenizer.__class__.__name__}")
print(f"Has chat_template: {hasattr(tokenizer, 'chat_template')}")

if hasattr(tokenizer, "chat_template"):
    template = tokenizer.chat_template
    print(f"Template length: {len(template) if template else 0}")
    if template and len(template) < 1000:
        print(f"Full template:\n{template}")
    elif template:
        print(f"Template preview (first 200): {template[:200]}")
        print(f"Template end (last 200): {template[-200:]}")

# Test with the failing seed
seed = 4216992985
adapter = get_adapter("math")

prompt_ids = adapter.build_prompt_ids(seed, tokenizer)
print(f"\nPrompt length: {len(prompt_ids)} tokens")

prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
print(f"First 300 chars: {prompt_text[:300]}")
print(f"First 10 token IDs: {prompt_ids[:10]}")

