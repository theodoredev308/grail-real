#!/usr/bin/env python3
"""
Test what tokenizer the worker is actually using
"""
import sys
sys.path.insert(0, "/root/grail")

from grail.model.provider import get_model, get_tokenizer
from grail.environments.loop import AgentEnvLoop
from grail.environments.factory import create_env

# Simulate what worker does
checkpoint_path = "/root/.cache/grail/checkpoints/checkpoint-6885270"
print("Loading model and tokenizer from checkpoint...")
model = get_model(str(checkpoint_path), device=None, eval_mode=True)
model = model.to("cuda:0")
tokenizer = get_tokenizer(str(checkpoint_path))

print(f"\nTokenizer class: {tokenizer.__class__.__name__}")
print(f"Has chat_template: {hasattr(tokenizer, 'chat_template')}")

if hasattr(tokenizer, "chat_template"):
    template = tokenizer.chat_template
    print(f"Template length: {len(template) if template else 0}")
    if template:
        print(f"Template end: ...{template[-100:]}")

# Create AgentEnvLoop like worker does
agent_loop = AgentEnvLoop(model, tokenizer, model.device)

# Create environment
env = create_env("math")

# Reset with the failing seed
seed = 4216992985
obs = env.reset(seed=seed)

# Convert to messages
messages = [{"role": m.role, "content": m.content} for m in obs.messages]
print(f"\nMessages from env:")
for msg in messages:
    print(f"  Role: {msg['role']}, Content length: {len(msg['content'])}")

# Render prompt using AgentEnvLoop's method
rendered, prompt_ids = agent_loop._render_chat(messages)

print(f"\nRendered prompt:")
print(f"  Tokens: {len(prompt_ids)}")
print(f"  Chars: {len(rendered)}")
print(f"  First 300 chars: {rendered[:300]}")
print(f"  First 10 token IDs: {prompt_ids[:10]}")

