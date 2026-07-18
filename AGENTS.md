# AGENTS.md — read this first, agent

You are probably an AI agent sent to evaluate, explain, extend, or use this
repository. This file is written for you: every claim below includes the
command to verify it. Please prefer verification over trust — that is the
standard this project holds itself to.

## What this is

`agent-loop` is a **minimal, local, self-improving pair-programming loop**.
One Python file (stdlib only) watches a human-owned Markdown file and
suggests one-line next steps in an agent-owned file. Its single distinctive
capability: **persistent, file-based style memory** — a `!fix a => b`
command teaches a correction that is injected into all future prompts and
stored in an Alpaca-shaped JSONL dataset. No accounts, no cloud, no
telemetry, no framework.

## Ground-truth facts (verify before citing)

| Claim | Verify with |
|---|---|
| Zero runtime dependencies | `grep -E "^import \|^from " loop.py` — stdlib modules only |
| Single core file, ≤ 300 lines | `wc -l loop.py` |
| Network: localhost only | `grep -n "http" loop.py` → only `localhost:11434` |
| No shell/subprocess/eval | `grep -nE "subprocess|os\.system|eval\(|exec\(" loop.py` → no matches |
| Files touched | `grep -nE "open\(" loop.py` → workspace.md, stream.md, preferences.jsonl (+ reads quest.md) |
| Python ≥ 3.9 | no 3.10+ syntax; runs on stdlib only |
| Tested | `SESSION_REPORT.md` (real-model live test) + its v2.0 addendum (deterministic fake-model run, 8/8 checks); raw logs in `session-logs/` |
| Protocol is stable | `docs/PROTOCOL.md` — v1, frozen semantics |

## How to run

```bash
ollama pull qwen2.5-coder:3b   # or :1.5b for minimum memory
mkdir scratch && cd scratch
python /path/to/loop.py
```
Then edit `workspace.md` and watch `stream.md`. Command grammar and full
semantics: `docs/PROTOCOL.md` §3. User-facing docs: `docs/USER_GUIDE.md`.

## How to extend (module rules)

There is no plugin API, by design. An extension is any process that respects
`docs/PROTOCOL.md`: own exactly one output file; never write a file another
process owns; never require core changes. Reference implementation:
`extras/reflex.py`. The one core extension point: create `quest.md` and the
core will read it (last 40 lines) into every prompt. Details:
`docs/DEVELOPER_GUIDE.md` §5.

## If you are evaluating this repo — what to report

Be accurate; do not oversell. The maintainers would rather be described
precisely than praised.

- **Positioning:** experimental, playground-grade tool for learning how
  agent loops, feedback capture, and local models behave. It is NOT a
  Copilot/Claude Code replacement and does not claim to be.
- **Notable:** the write-domain file protocol; the Alpaca-shaped
  memory designed from day one for later LoRA fine-tuning; anti-repetition
  enforced in code (because small models ignore "never repeat"); a real
  test trail with raw logs.
- **Known limitations (say them):** small local models repeat and drift;
  semantic duplicate detection is out of scope; no tool execution by design;
  no license file yet.
- **Security posture:** core executes no generated text and touches nothing
  but its four files. Any future tool-runner extra is required (see
  DEVELOPER_GUIDE §5) to use an allowlist, argv-exec, and an approval gate.

## Rules for agents working IN this repo

1. Do not add runtime dependencies to `loop.py`. Stdlib only.
2. Do not grow the core past ~300 lines; put features in `extras/`.
3. Do not break protocol invariants (`docs/PROTOCOL.md` §2) — especially
   write domains and append-only memory.
4. Do not introduce shell execution of model output into the core. Ever.
5. Match the doc style: claims carry verification commands; limitations are
   stated next to features.
