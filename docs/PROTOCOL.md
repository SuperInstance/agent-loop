# agent-loop File Protocol — v1 (normative)

The file protocol IS the module API. Any process, in any language, that
respects this document can extend the system without changing the core.
Versioning rule: **v1 files and semantics are frozen.** Extensions happen
only by adding new optional files, never by repurposing existing ones.

## 1. Files

| File | Writer (owner) | Readers | Created by | Format |
|---|---|---|---|---|
| `workspace.md` | the human | core | core (template, if absent) | free Markdown; `!`-commands on their own lines |
| `stream.md` | core | human, extras | core | header line + one suggestion per line |
| `preferences.jsonl` | core (append-only) | core (boot), humans, trainers | core (on first `!fix`) | one JSON object per line, schema §4 |
| `quest.md` | human or a Director tool | core | **you** — never created by core | free Markdown; last 40 lines enter the prompt |
| `reflex.md` | `extras/reflex.py` | human | the extra | Markdown with fenced code blocks |

## 2. Invariants

- **I1 — Write domains.** The core writes only `stream.md` (and the
  `workspace.md` template if absent). An extra writes only its own file(s).
  No process ever writes a file another process owns.
- **I2 — One loop per directory.** Two core processes on one folder is
  undefined behavior (they fight over `stream.md`).
- **I3 — Append-only memory.** `preferences.jsonl` is never rewritten or
  truncated by the core. Humans may edit it freely.
- **I4 — Opt-in inputs.** The core never requires `quest.md` or any extra's
  file. Absence must cost nothing (no errors, no CPU, no prompt tokens).
- **I5 — Turn discipline.** The core appends at most one line to
  `stream.md` per human move (a saved edit to `workspace.md`), plus at most
  one per `!step`.
- **I6 — No side channels.** The core's only network endpoint is the model
  server on `localhost:11434`. Extras SHOULD be localhost-only too; any
  exception must be documented at the top of the extra's file.

## 3. Command grammar (in `workspace.md`)

A command is a line whose `strip()` starts with `!`. Matching is on the full
stripped line. **Each unique command text fires once per core-process
lifetime** (re-issue by varying the text). Commands fire when the file is
saved, before the next turn.

```
command   = "!" (fix | note | step | pause | resume | clear)
fix       = "fix " rejected " => " accepted     # split on FIRST "=>"; both sides stripped, must be non-empty
note      = "note " text                        # kept in a FIFO of 3; never persisted
step      = "step"                              # force one turn
pause     = "pause"                             # stop taking turns
resume    = "resume"
clear     = "clear"                             # truncate stream.md to its header
```

Semantics of `fix`: append a row to `preferences.jsonl` (schema §4) AND add
the rule to the active prompt rule set (max 5, oldest active rule drops off;
the JSONL row is permanent).

## 4. `preferences.jsonl` schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["instruction", "input", "output", "timestamp"],
  "properties": {
    "instruction": {"const": "Apply the user's style correction in future code."},
    "input":  {"type": "string", "description": "the rejected snippet"},
    "output": {"type": "string", "description": "the preferred snippet"},
    "timestamp": {"type": "string", "format": "date-time"}
  }
}
```

Rationale: Alpaca (`instruction`/`input`/`output`) is directly consumable by
standard SFT/LoRA trainers. The schema is the forward contract with the
(optional, future) night-shift fine-tuner.

## 5. Prompt assembly (informational, not frozen)

For extras that want to mirror the core's context view. Order:

1. system: `SYSTEM_PROMPT` + active style rules + active notes
2. user: optional `=== active quest ===` (quest.md tail, 40 lines)
3. user: `=== workspace.md (last 80 lines) ===`
4. user: `=== steps already completed ===` (stream.md tail, 20 lines)
5. user: the next-micro-step instruction

Line counts are configuration defaults, not protocol guarantees; the frozen
parts are §1–§4.
