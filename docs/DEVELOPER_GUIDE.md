# agent-loop — Developer Guide

How the core works, why it looks the way it does, and how to extend it
without breaking the two rules that make it trustworthy:

1. **The core stays tiny and dependency-free.** Target: ≤ 300 readable lines.
2. **Features live in `extras/`** — separate processes speaking the file
   protocol. If a feature needs the core changed, the burden of proof is on
   the change, and it must cost zero when unused (like `quest.md`).

Read `docs/PROTOCOL.md` for the normative file/API spec. This guide is the
"narrative" version.

---

## 1. Architecture in one picture

```
            ┌────────────────────────┐
            │   human (any editor)   │
            └───────┬────────▲───────┘
              writes│        │copies back what they like
                    ▼        │
   workspace.md ──────────── stream.md          (write domains)
        ▲    ▲                ▲   ▲
        │    │ reads          │   │ writes
        │    └────────────────┤   │
        │ quest.md (optional) │   │
        │                     │   │
   preferences.jsonl ◄──── loop.py│
   (boot: load rules;            │   extras/reflex.py (optional,
    runtime: append !fix)        └──► separate process, writes
                                      only reflex.md)
```

One process, one loop: poll `workspace.md` mtime once a second; on change,
debounce 2.5 s; build one prompt; take one turn; append at most one line to
`stream.md`. That is 90% of the system. The rest is memory and hygiene.

## 2. `loop.py` walkthrough (by section)

**Config block (top of file).** The entire config surface. Constants, not
flags: editing the file is the config system. `NUM_CTX=8192` exists because
Ollama's server-side default is 2048, which silently truncates this app's
prompts — a common Ollama integration bug.

**`_get_json` / `_post_json`.** Stdlib `urllib` wrappers. No `requests`
dependency — the core must run on a bare Python install. Errors propagate as
exceptions and are caught at the call site.

**Files.** `read_text` returns `""` on missing files so every reader is
total. `mtime()` feeds the poll loop. `ensure_files()` creates the two
working files (never `quest.md` — it must be opt-in by existing).

**`check_server()`.** One `GET /api/tags` at boot; exits with a clear
message if the server is down, warns if the configured model isn't pulled.
Fail fast, in prose a human can act on.

**Memory (`load_rules` / `log_fix`).**
- `log_fix` appends an Alpaca-shaped row (`instruction`/`input`/`output` +
  timestamp) to `preferences.jsonl`. Alpaca because the night-shift LoRA
  trainer (roadmap) consumes that shape natively — the schema *is* the
  future training contract.
- `load_rules` reads the file once at boot and keeps the last
  `MAX_FIX_RULES` (5). The file is append-only; the loop never rewrites it.

**`build_messages()`.** The prompt contract:
- system = `SYSTEM_PROMPT` + style rules (from memory) + live notes.
- user = optional `=== active quest ===` block (only if `quest.md` exists)
  + workspace tail (80 lines) + completed-steps tail (20 lines)
  + the instruction: *output the single NEXT micro-step that comes AFTER
  the completed steps*. Positive framing — see §4 for why.

**`_first_line()`.** Extracts the one-line suggestion from a model reply.
Checks `startswith("```")` **before** stripping backticks — the live test
caught a model's fenced reply becoming the suggestion `python`. (Bug class:
parse-then-validate ordering.)

**`_normalize()` / `_already_done()`.** Repeat detection. Normalization:
lowercase, strip leading bullets/backticks, strip trailing period. Compare
against *every* line in `stream.md`. Exact-after-normalization match only —
semantic dedup ("Add" vs "Define") was judged overkill for the core.

**`ask_architect()`.** One turn, two attempts:
1. temperature 0.3 (focused). If the suggestion repeats history →
2. temperature 0.85 ("hot retry"). Still repeating → return `""`, turn
   skipped. The console says what happened on both paths.
Rationale: never write a duplicate to the human-visible stream; a skipped
turn is better than a spammed one. Live-test evidence: the retry fired on
nearly every turn with a 1.5B model — this code is load-bearing.

**`handle_command()`.** Full-line commands from `workspace.md`, matched
after `strip()`. Returns `(paused, force_step)`. `!fix` splits on the first
`=>`. Unknown `!`-lines get a help print, not an error.

**`main()` — the state machine.** Six pieces of state:
`paused`, `dirty`, `force_step`, `last_edit`, `workspace_seen`,
`initial_content`, plus `seen_commands` and `last_mtime`.

```
each second:
  mtime changed? ──no──► sleep
       │yes
       read file; content changed? ──no──► sleep
       │yes
       dirty = True; last_edit = now
       consume any new "!"-lines (each unique text fires once)
  if workspace == initial_content: dirty = False   # only react to edits after boot
  if force_step: dirty = True; last_edit = 0
  if paused or not dirty or now-last_edit < IDLE_DEBOUNCE: sleep
  else: take one turn; dirty = False               # one suggestion per human move
```

Notable invariants:
- **Boot guard** (`initial_content`): the loop only reacts to edits made
  after it started — reopening an old workspace doesn't trigger a burst.
- **Command dedup** is by exact line text and lives in `seen_commands`
  (per-process). Documented behavior; re-issue by varying the text.
- **One turn per human move** (`dirty = False` after each turn) — the
  human's edit is the clock. `!step` exists for autonomous nudges.

## 3. The two lessons the live test taught (design rationale)

These are encoded in the code; know them before changing anything.

1. **Never ask a small model to enforce an invariant the code can enforce.**
   "Never repeat yourself" in the system prompt failed 100% of the time in
   testing; `_already_done()` catches it deterministically. Same class:
   output length is capped by `num_predict`, not by "under 15 words".
2. **Frame instructions positively.** "Steps already completed → what comes
   AFTER" works measurably better than "don't repeat your recent
   suggestions". Small models follow "do X" far better than "don't do Y".

Full evidence: `SESSION_REPORT.md` + `session-logs/`.

## 4. Testing methodology (use it for your changes)

Two layers, both cheap:

**A. Deterministic fake model (for logic changes).** A ~60-line stdlib HTTP
server that speaks Ollama's `/api/tags` + `/api/chat` shape and answers
*content-aware canned replies*: repeat yourself at low temperature (tests
dedup + hot retry), answer fenced blocks (tests parsing), assert on what the
prompt contained (tests quest inclusion + rule injection) — the assertions
are checked **server-side**, i.e., on what the loop actually transmitted.
This ran the stripped core end-to-end in ~30 s with 8/8 checks passing.
Write one for any protocol-touching PR.

**B. Scripted human (`extras/conductor_driver.py`).** Drives a real session
against a real model: seeds a goal, steers with `!note`, accepts a reflex
block, teaches a `!fix`, forces `!step`. Use before releases.

Gotchas discovered (avoid re-learning):
- Run **one loop per directory**. Two loops on one folder fight over
  `stream.md` and produce contaminated, hard-to-debug behavior.
- When `pkill`-ing test processes, match the full path or use `[b]racket`
  patterns — `pkill -f loop.py` matches the shell running your own command.
- Python's `atexit` handlers don't run on SIGTERM; log test assertions
  incrementally, at request time.

## 5. Writing an extra (the module system that isn't)

There is no plugin API. An extra is **any process, in any language, that
respects the file protocol** (`docs/PROTOCOL.md`). Rules:

1. Own exactly one output file; never write a file you don't own.
2. Read whatever you like.
3. Prefer the standard library; if you need a dependency, your extra must
   still be optional to run the core.
4. Don't require core changes. (If you truly do, that's a PROTOCOL version
   bump conversation, not a PR.)

**Skeleton (Python, stdlib):**

```python
import time
def read(p):
    try: return open(p, encoding="utf-8").read()
    except FileNotFoundError: return ""
seen = set(read("stream.md").splitlines())
while True:
    time.sleep(0.5)
    for line in read("stream.md").splitlines():
        if line not in seen and line.strip() and not line.startswith(("#", "!")):
            seen.add(line)
            with open("myextra.md", "a", encoding="utf-8") as f:
                f.write(f"- reaction to: {line}\n")
```

That's a complete extra. `extras/reflex.py` is the reference implementation.

**Extras on the roadmap (and their hard requirements):**
- **LoRA trainer** — consumes `preferences.jsonl` (schema already stable),
  Unsloth, runs offline. Never part of the core.
- **Sandboxed tool runner** — only acceptable as: allowlisted verbs,
  argv-exec (**never** `shell=True`), no secrets in env, human approval gate
  file, audit log. The original design's `[TOOL: …] → shell(True)` is the
  canonical example of what we will not ship. The system's own first sketch
  of the safe version is in `session-logs/reflex.md`.
- **Quest director** — needs nothing: write `quest.md`; the core already
  reads it.

## 6. Security model (and how to keep it)

The core's guarantees, which users are told to verify:
- Network: `localhost:11434` only. (`grep -n "http" loop.py`)
- Files: reads/writes only `workspace.md`, `stream.md`,
  `preferences.jsonl`, and `quest.md` if present.
- No shell, no `subprocess`, no `eval`, no git, no telemetry, no accounts.

Any PR weakening these needs a threat model in the description. Tool-calling
belongs in an extra with the properties in §5, not in the core.

## 7. Performance notes

- Poll loop: one `stat` per second on an unchanged file; one read on change.
  CPU unmeasurable. RSS ~15 MB (CPython + stdlib).
- Prompt budget: (80 + 20 + 40) tail lines ≈ well under half of `NUM_CTX`
  at typical code width — headroom for rules/notes/quest. If you widen
  tails, widen `NUM_CTX` to match; Ollama truncates silently.
- The model call dominates wall time; everything else is noise.

## 8. Contributing & compatibility

- **Style:** stdlib only in core; readable over clever; comments explain
  *why*, not what.
- **PRs:** include a fake-model test for protocol changes; keep the core
  under 300 lines or justify every line over.
- **Non-Ollama servers:** the core speaks Ollama's HTTP shape. For
  OpenAI-shaped servers (llama.cpp, LM Studio), run a translating shim —
  a ~60-line pattern mapping `/api/chat` ↔ `/v1/chat/completions`
  (`options.num_predict` ↔ `max_tokens`, etc.). The v2.0 validation used
  exactly this against llama.cpp.
- **License:** none yet — add one (MIT recommended) before publishing.
