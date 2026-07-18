# agent-loop

A self-improving local pair-programming loop. **One file, zero dependencies,
Python 3.9+.** Its distinguishing feature is persistent style memory: a tiny
local agent that *learns your corrections and never forgets them* — no
account, no cloud, no subscription.

```
you write in workspace.md  ──►  the loop reads it  ──►  one micro-step lands in stream.md
         ▲                                                        │
         └────────────  you copy back what you like  ◄────────────┘
   !fix this => that   ──►  remembered forever in preferences.jsonl
```

## The trust contract

- **One file.** `loop.py` is the entire core. Read it in five minutes; there
  is no framework, no plugin system, no build step.
- **Zero dependencies.** Standard library only. The only requirement is a
  local model server (Ollama).
- **It touches exactly four files** in the folder you run it in:
  `workspace.md`, `stream.md`, `preferences.jsonl`, and `quest.md` (only if
  you create one). Nothing else. No git, no shell, no network beyond
  `localhost:11434`.
- **~15 MB RAM, near-zero CPU** (it stats one file per second). The only real
  cost is the model, and you choose its size (`qwen2.5-coder:1.5b` ≈ 1 GB,
  `:3b` ≈ 2 GB).
- **Everything else is optional.** Extras are *separate processes* in
  `extras/` that speak the same file protocol. Run them when you want them;
  the core is unaffected either way.

## Quickstart (60 seconds)

```bash
ollama pull qwen2.5-coder:3b        # or :1.5b for minimum memory
mkdir scratch && cd scratch
python /path/to/loop.py
```

Open `workspace.md` and `stream.md` side by side in your editor. Write a
goal, save, wait ~3 seconds. Copy back what you like; teach it with a
command:

| Command | Effect |
|---|---|
| `!fix var x = 1; => const x = 1;` | Correction → `preferences.jsonl` + live style rule |
| `!note use pathlib, not os.path` | Ephemeral guidance (last 3 kept) |
| `!step` | Force one suggestion now |
| `!pause` / `!resume` | Freeze / unfreeze |
| `!clear` | Wipe `stream.md` |

Suggested `.gitignore` for the scratch folder: `stream.md`, `preferences.jsonl`.

## Modules (all optional, all separate processes)

| Module | What it adds | When you want it | Cost |
|---|---|---|---|
| `loop.py` (core) | The self-improving loop | Always | ~15 MB, 0 deps |
| `extras/reflex.py` | Expands each micro-step into a code block in `reflex.md` | Only if you DON'T already have autocomplete (Copilot/Continue/CodeGeeX do this job) | ~15 MB proc, shares model |
| `extras/conductor_driver.py` | Scripted human — replays the live-fire test | Testing, demos, CI | none |
| A Director tool (Claude Code, etc.) | Long-horizon planning | Write a `quest.md` in the folder; the core picks it up next turn — that file is the *only* extension point | zero in core |
| LoRA trainer (roadmap) | Bakes `preferences.jsonl` into the model weights | After a few hundred corrections | GPU-hours, only when run |

Why files as the module API: any tool that can write text can extend the
system, in any language, with zero changes to the core — and nothing you
don't run can cost you anything.

## Provenance

This is a stripped-down, live-tested descendant of a much larger multi-agent
design. The full critique, the fixes, and the unedited session transcripts
(including the three bugs only a real model could expose) are in
`SESSION_REPORT.md` and `session-logs/`.

## Documentation map

| Doc | For | Contents |
|---|---|---|
| `docs/USER_GUIDE.md` | users | install, quickstart, command reference, personas, troubleshooting, FAQ |
| `docs/DEVELOPER_GUIDE.md` | developers | architecture, code walkthrough, design rationale, testing, writing extras |
| `docs/PROTOCOL.md` | developers | the normative file-protocol spec (v1, frozen) |
| `AGENTS.md` | AI agents | verifiable facts table, evaluation guidance, repo rules |
| `llms.txt` | AI agents | llms.txt-standard summary + doc index |
| `agent-card.json` | AI agents | machine-readable capability & trust manifest |
| `SESSION_REPORT.md` | everyone | live-fire test evidence + v2.0 validation addendum |
