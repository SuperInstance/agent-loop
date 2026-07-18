# agent-loop

A **constantly-thinking local autonomous agent** with a **flock of tiny models** that load and unload based on task complexity. Zero dependencies. Python 3.9+. Stdlib only. Runs on [Ollama](https://ollama.ai).

```
you write goals.md ──► the agent works through them autonomously
     ▲                        │
     │                   iterator (0.5B, always loaded)
     │                        │
     │                   escalates when stumped
     │                        ▼
     │                   specialist loads (coder/thinker/vision)
     │                   solves → evicts from VRAM
     └── results land in stream.md + working_memory.jsonl
```

## What this is

A local-first autonomous agent for your laptop. It thinks continuously — not just when you type. It works through a goal queue, uses sandboxed tools, and escalates to bigger models when the small one is stumped.

**The flock**: 14 models across 5 tiers. The smallest (qwen2.5:0.5b, ~400MB) runs permanently as the heartbeat/dispatcher. When it hits something hard, it loads a specialist — a coder, a reasoner, a vision model — solves the problem, then unloads it to free VRAM.

## Quickstart

```bash
# 1. Install Ollama (if not already)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull the iterator (required)
ollama pull qwen2.5:0.5b

# 3. Pull at least one specialist (recommended)
ollama pull qwen2.5-coder:3b

# 4. Source the hardware env
source setup_env.sh

# 5. Run the autonomous agent
python autonomous.py
```

Open `goals.md` in your editor and add a goal. The agent starts working on it within 30 seconds.

## Files

### Core
| File | Purpose | Lines |
|------|---------|-------|
| `autonomous.py` | The constantly-thinking agent (tick loop, goals, tools, memory) | 838 |
| `model_router.py` | Multi-model flock dispatcher (14 models, VRAM manager) | 994 |
| `loop.py` | Original reactive pair-programming loop (preserved) | 279 |
| `setup_env.sh` | Hardware env vars (CUDA, MMQ, flash attention, VRAM budget) | 48 |
| `flock.yaml` | Hardware-specific config (RTX 4050/4060 default) | 100 |

### Protocol Files (created at runtime)
| File | Owner | Purpose |
|------|-------|---------|
| `workspace.md` | Human | Your input, goals, `!commands` |
| `goals.md` | Human | Goal queue the agent works through |
| `stream.md` | Agent | One-line micro-steps (suggestions) |
| `working_memory.jsonl` | Agent | Persistent memory across restarts |
| `preferences.jsonl` | Agent | Style corrections (Alpaca-shaped, trainer-ready) |
| `tick.md` | Agent | Status: last tick, active goal, next tick |
| `config.yaml` | Human | Safety fences, tick settings, tool allowlist |

### Design Docs
| Doc | Author | Purpose |
|-----|--------|---------|
| `EXTENSION_DESIGN.md` | Claude Code | Architecture: tick, goals, tools, memory, safety |
| `MODEL_ROSTER_DESIGN.md` | Claude Code | 14-model roster, escalation protocol, VRAM math |
| `WHITEPAPER.md` | Kimi K2.7 | "Emergence by Arrangement" — the theory |
| `SESSION_REPORT.md` | Kimi K2.7 | Live-fire test evidence |

## Commands

Type these in `workspace.md`:

| Command | Effect |
|---------|--------|
| `!goal Write tests for auth module` | Add a goal to the queue |
| `!goals` | List all goals and status |
| `!status` | Show tick status and active goal |
| `!fix var x => const x` | Learn a style correction (permanent) |
| `!note use pathlib not os.path` | Ephemeral guidance (last 3 kept) |
| `!step` | Force one suggestion now |
| `!pause` / `!resume` | Freeze / unfreeze the agent |
| `!approve` / `!reject` | Respond to approval requests |
| `!clear` | Wipe stream.md |

## The Model Flock

| Tier | Model | Tag | Specialty | VRAM |
|------|-------|-----|-----------|------|
| 0 | qwen2.5:0.5b | iterator | dispatch/heartbeat | 0.5GB |
| 1 | qwen2.5-coder:1.5b | fast-coder | coding | 1.2GB |
| 1 | deepseek-r1:1.5b | fast-thinker | reasoning | 1.2GB |
| 1 | gemma2:2b | generalist | general | 1.8GB |
| 2 | qwen2.5-coder:3b | coder | coding | 2.5GB |
| 2 | qwen2.5:3b | thinker | reasoning | 2.5GB |
| 2 | phi3:3.8b | scholar | reasoning | 3.0GB |
| 3 | qwen2.5-coder:7b | architect | coding | 5.0GB |
| 3 | deepseek-r1:7b | sage | reasoning | 5.0GB |
| 3 | mistral:7b | general-heavy | general | 5.0GB |
| 3.5 | qwen3.5-next-moe | moe-oracle | knowledge (80B/3B active) | 4.2GB |
| 3.5 | minicpm-o:4.5 | herald | audio/speech | 3.8GB |
| 4 | qwen2.5-vl:3b | eyes | vision | 2.5GB |
| 4 | llava:7b | watchman | vision (heavy) | 5.0GB |

## Hardware Requirements

- **Minimum**: 6GB VRAM (RTX 4050), 16GB RAM
- **Recommended**: 8GB VRAM (RTX 4060), 32GB RAM
- **Ollama** running on localhost:11434
- Python 3.9+
- No pip dependencies — stdlib only

## Architecture

Read [EXTENSION_DESIGN.md](EXTENSION_DESIGN.md) and [MODEL_ROSTER_DESIGN.md](MODEL_ROSTER_DESIGN.md) for the full specification.

The key insight: **capability emerges from arrangement, not from model size**. A 0.5B model in the right protocol outperforms a 7B model in the wrong one. The flock architecture gives you the 7B when you need it, at the cost of the 0.5B when you don't.

## License

MIT
