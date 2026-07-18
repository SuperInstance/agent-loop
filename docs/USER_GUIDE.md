# agent-loop — User Guide

A self-improving local pair-programming loop. One file, zero dependencies,
Python 3.9+. Its distinguishing feature is persistent style memory: a small
local agent that **learns your corrections and never forgets them** — no
account, no cloud, no subscription.

> **In one paragraph:** You write in `workspace.md`. A local model (via
> Ollama) watches, and after you pause, it drops a one-line micro-step into
> `stream.md`. You copy back what you like. When it gets your style wrong,
> you type `!fix wrong => right` once — and the rule is injected into every
> later prompt and kept across sessions.

---

## 1. Requirements

| Requirement | Minimum | Comfortable |
|---|---|---|
| Python | 3.9 (stdlib only, nothing to pip install) | 3.11+ |
| Model server | [Ollama](https://ollama.com) running locally | same |
| Model | `qwen2.5-coder:1.5b` (~1 GB) | `qwen2.5-coder:3b` (~2 GB) |
| Hardware | any CPU that runs the model | GPU with 4–6 GB VRAM |
| RAM for the loop itself | ~15 MB | ~15 MB |

The loop is deliberately tiny; the model is the only real resource cost, and
you choose its size.

## 2. Install

```bash
# 1. get the model (one time)
ollama pull qwen2.5-coder:3b        # or qwen2.5-coder:1.5b for minimum memory

# 2. put loop.py anywhere, make a scratch folder, run
mkdir scratch && cd scratch
python /path/to/loop.py
```

You should see:

```
* loop up — watching workspace.md, writing stream.md
* commands: !fix a => b | !note txt | !step | !pause | !resume | !clear
```

If it says it can't reach the model server, start Ollama first
(`ollama serve`, or just launch the Ollama app).

**Uninstall:** delete the folder. Optionally `ollama rm qwen2.5-coder:3b`.
There is nothing else — no config directories, no background services, no
registry keys.

## 3. Your first five minutes

1. Run `loop.py`. It creates two files in the folder:
   - `workspace.md` — **yours.** Goals, code, commands.
   - `stream.md` — **the agent's.** Its suggestions appear here.
2. Open both side by side in your editor. (In VS Code, external changes
   auto-reload as long as you have no unsaved edits in the file you're
   watching.)
3. In `workspace.md`, replace the goal line, e.g.:

   ```markdown
   ## Goal: a CLI that renames photos by EXIF date
   ```

4. Save. Wait ~3 seconds. A micro-step appears in `stream.md`:

   ```
   Import os and exifread.
   ```

5. React however you want:
   - **Copy** the line into `workspace.md` and keep building on it — copying
     is the "accept" gesture, and it's itself a human move that triggers the
     next suggestion.
   - **Ignore it** — the agent waits; it never types over you.
   - **Steer it** with a `!note`. **Correct it** with a `!fix`.

That's the whole loop. Everything else is refinement.

## 4. The mental model

- **Write domains.** You own `workspace.md`; the agent owns `stream.md`.
  Neither ever writes in the other's file. This is what keeps the loop calm:
  the agent can't fight you, and it can't mistake its own output for yours.
- **One suggestion per human move.** The agent takes exactly one turn each
  time you change `workspace.md` and then stop typing for ~2.5 s. It will
  not monologue. If you want another take without editing, type `!step`.
- **Memory is a file.** Corrections live in `preferences.jsonl` in the same
  folder — human-readable, append-only, yours to inspect, edit, or delete.
- **It's a scratchpad, not an IDE.** The loop doesn't know your project
  structure and doesn't run your code. It suggests next steps, remembers
  your taste, and stays out of the way.

## 5. Command reference

Commands are ordinary lines in `workspace.md` that start with `!`. A command
fires **once per unique line** per session — to re-issue the identical
command, change it slightly (e.g. add a space or a word).

| Command | Syntax | Effect |
|---|---|---|
| Fix (teach) | `!fix <rejected> => <preferred>` | Appends an Alpaca-shaped row to `preferences.jsonl` **and** injects the rule into the prompt for all future turns. Last 5 rules stay active; all rows are kept forever. |
| Note (steer) | `!note <text>` | Adds ephemeral guidance to the next prompts. Last 3 notes are kept; notes do **not** survive restarts. |
| Step | `!step` | Forces one suggestion immediately, without waiting for an edit + pause. |
| Pause | `!pause` | Freezes the agent (file watching continues). |
| Resume | `!resume` | Unfreezes. |
| Clear | `!clear` | Wipes `stream.md` back to its header. |

**Examples**

```
!fix var x = 1; => const x = 1;
!fix ALLOWED = ['ls'] => ALLOWED: set[str] = {'ls'}
!note prefer dataclasses over dicts
!note this is a CLI, argparse not click
```

**Edge cases**

- `!fix` without `=>` prints a usage hint and does nothing.
- The `!fix` line stays visible in `workspace.md`; that's fine — it's also
  useful context. Delete it if it bothers you; the rule is already saved.
- If the agent's suggestion is a near-repeat of an earlier one, the loop
  detects it, retries once at higher temperature, and skips the turn if the
  model insists. You'll see `— repeat detected …` on the console.

## 6. Working styles

### 6.1 Solo (just the core)
You + the loop. Best for thinking through a design, sketching a module, or
learning what the memory system feels like.

### 6.2 With the local reflex (optional extra)
```bash
python /path/to/extras/reflex.py      # second terminal, same folder
```
Watches `stream.md`; expands each micro-step into a 3–8 line code block in
`reflex.md`. Accept a block by copying it into `workspace.md`. Use this only
if you don't already have an autocomplete tool — see 6.3.

### 6.3 With your existing autocomplete (recommended)
Copilot, Continue, CodeGeeX, etc. already complete code at your cursor —
keep using them in `workspace.md`. The loop handles *direction*, your
autocomplete handles *syntax*. The reflex extra exists for people who want
every layer local and account-free.

### 6.4 With a Director agent (quest.md)
If you use an agentic tool (Claude Code, Codex, etc.), point it at the
folder and let it write `quest.md` — a mission brief. The core reads it (if
it exists) on every turn and steers all suggestions toward it. This is the
entire integration: one file, zero configuration. Example `quest.md`:

```markdown
# Quest: streaming log parser
Target: zero memory growth on 10 GB files.
Prefer iterators and generators everywhere.
```

### 6.5 One folder per project (important)
`preferences.jsonl` lives in the folder — style memory is **per project by
design**. Run one loop per folder. Never run two loops on the same folder:
they fight over `stream.md` (observed in testing; the result is unusable).
To share taste across projects, copy `preferences.jsonl`.

## 7. Configuration

All configuration is six constants at the top of `loop.py`. Editing the file
*is* the config system — there is no settings file to learn.

| Constant | Default | What it does |
|---|---|---|
| `MODEL_NAME` | `qwen2.5-coder:3b` | Any chat model your Ollama has. Instruct-tuned recommended. |
| `NUM_CTX` | `8192` | Context window passed to Ollama. Its default (2048) is too small; don't lower this. |
| `IDLE_DEBOUNCE` | `2.5` | Seconds of typing quiet before the agent takes a turn. Raise if it interrupts you. |
| `WORKSPACE_TAIL_LINES` | `80` | How much of your file the agent sees. |
| `STREAM_TAIL_LINES` | `20` | How much of its own history it sees. |
| `MAX_FIX_RULES` | `5` | Active style rules (oldest drops off; the JSONL keeps everything). |

Generation parameters (temperatures `0.3`/`0.85`, 64-token cap) are in
`ask_architect()` if you want to tune the model's voice.

### Persona recipes (drop-in `SYSTEM_PROMPT` replacements)
From the original design docs:

- **Bare-metal architect:** *"You are a strict, low-level systems architect.
  Output ONLY the next variable declaration, algorithmic step, or structural
  construct. Focus on performance, edge cases, explicit types. Under 10
  words."*
- **TDD specialist:** *"You are a Test-Driven Development specialist. Your
  sole job is to output the next boundary test case, assertion, or
  validation step. Force the human to implement logic that satisfies your
  tests. Under 12 words."*
- **Minimalist:** *"You are an ultra-minimalist. Output a single line of
  code or a three-word comment. Nothing else, ever."*

## 8. The memory file (`preferences.jsonl`)

One JSON object per line:

```json
{"instruction": "Apply the user's style correction in future code.", "input": "var x = 1;", "output": "const x = 1;", "timestamp": "2026-07-17T04:22:12Z"}
```

- **Alpaca-shaped** (`instruction`/`input`/`output`) so a future fine-tuning
  script can consume it directly.
- **Append-only** — the loop never rewrites it. Edit it by hand whenever you
  want; deleting bad rows is how you "un-teach".
- The last `MAX_FIX_RULES` rows are reloaded at startup, so style memory
  survives restarts.
- Roadmap: once you have a few hundred rows, a LoRA night-shift (Unsloth,
  ~1 GPU-hour for a 3B) bakes your style into the weights. The JSONL schema
  was designed for this from day one.

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Cannot reach the model server…` at startup | Ollama isn't running. Start it; rerun. |
| `warning: 'qwen2.5-coder:3b' not found` | `ollama pull qwen2.5-coder:3b`. Or pick a model you have (`ollama list`) and edit `MODEL_NAME`. |
| Nothing happens after I edit | Check the console: `!pause` may be on. Confirm you *saved* the file — the loop watches disk, not the editor buffer. Confirm the file differs from boot state (the loop reacts to edits made after it started). |
| `— repeat detected … skipping turn` | Small models repeat themselves; the loop caught it and refused to spam. Edit `workspace.md` to give the model new information, or use `!note` to redirect. |
| Suggestions are generic | The model is small; give it more to work with — paste real code into `workspace.md`, write a sharper goal, or add a `quest.md`. |
| Editor doesn't show new suggestions | VS Code: the file must have no unsaved changes (or set `"files.autoSave": "afterDelay"`). Other editors: enable "reload file on external change". |
| `stream.md` is huge | `!clear` it. The agent only reads the last 20 lines anyway. |
| Two loops accidentally ran on one folder | Stop both, `!clear`-equivalent: delete `stream.md` body, restart one. Never run two on one folder. |
| Windows paths / antivirus | Pure Python, no native code. If AV sandbox-blocks file watching, run in a normal user folder rather than a synced/protected one. |

## 10. FAQ

**Does my code leave the machine?** No. The loop's only network I/O is HTTP
to `localhost:11434`. Verify: `grep -n "http" loop.py` — you'll find exactly
the localhost URLs. (The *reflex extra* is the same; CodeGeeX itself is a
cloud service.)

**What does it cost to run?** The Python process is ~15 MB and stats one
file per second — unmeasurable in normal use. The model cost is whatever
Ollama already costs you.

**CPU-only?** Works — a 1.5B model answers a one-liner in a few seconds on
2 cores (measured in testing). A GPU makes it feel instant.

**Non-Python languages?** Language-agnostic: it's all just text. The default
prompt is code-neutral; persona recipes can bias it (see §7).

**Other model servers (LM Studio, llama.cpp)?** The core speaks the Ollama
HTTP shape. For OpenAI-shaped servers, run a tiny translating shim — the
test suite used exactly that pattern; see the Developer Guide.

**Is this a Copilot/Claude Code replacement?** No, and it isn't trying to
be. It's a small local scratchpad that learns your taste. See the Session
Report for what it does well and where it falls short.

**Can I use it in a real repo?** Yes — run it in the repo root and add
`stream.md` + `preferences.jsonl` to `.gitignore`. It will never touch
anything else.

## 11. Getting help / giving feedback

Read `docs/DEVELOPER_GUIDE.md` before opening the code — it's short, and it
explains the rules that keep the core tiny. Bugs and extras are welcome;
features belong in `extras/`, not the core.
