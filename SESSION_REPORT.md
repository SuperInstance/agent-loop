# Live Fire Test — Session Report

**Date:** 2026-07-17 · **Goal under test:** the system ideates about its own v2
(the sandboxed tool layer: allowlist, approval gate, audit log).

## Test setup

Sandbox had no GPU (2 cores / 4 GB), so the stack was adjusted for the rig:

| Component | Your setup | Test rig |
|---|---|---|
| Model | qwen2.5-coder:3b on Ollama (RTX 4050) | qwen2.5-coder:**1.5b**-instruct Q4_K_M on llama.cpp, CPU-only |
| API | Ollama `/api/chat` | llama.cpp OpenAI-compatible endpoint (protocol logic identical) |
| Reflex | CodeGeeX (cloud) | `reflex_simulator.py` — same model, temp 0.1, syntax-only |
| Conductor | you | `conductor_driver.py` — scripted human with adaptive waits |

Expectation setting: a 1.5B CPU model is the *floor*. Everything that worked
here works better on your 3B/GPU setup; the bugs it exposed are real bugs.

## What worked (verified in the transcript — see `session-logs/`)

1. **The turn-taking protocol held.** Goal → micro-step → reflex expansion →
   human acceptance → next step. No self-amplification, no stalls; every
   Conductor edit re-armed the loop exactly once.
2. **The learning loop visibly functions.** After `!fix ALLOWED = ['ls', ...]
   => class Tool(Enum)...`, the Architect's very next micro-steps were about a
   `Tool` enum with `LS`/`PYTEST`/`GIT_STATUS` — the taught rule. The rule
   also **persisted across restarts** via `preferences.jsonl` (Alpaca-shaped,
   trainer-ready).
3. **The Reflex stand-in behaved like CodeGeeX should.** Instant expansion of
   each intent; its approval-gate sketch (`ApprovalStatus` enum +
   `approve_tool()`) was usable boilerplate.
4. **Commands all parsed:** `!note` injected guidance, `!fix` logged+taught,
   `!step` forced a turn, dedup of repeated command lines worked.

## Bugs the test found (and the fixes shipped in v1.1)

1. **Fence-parsing bug.** The model answered with a fenced block; the parser
   stripped backticks *before* checking for fences, so ` ```python ` became the
   suggestion `python` — which was then dutifully written to `stream.md` and
   expanded by the Reflex. **Fix:** check `startswith("```")` before stripping.
2. **Repetition loop.** Session 1 produced the *identical* micro-step six
   times despite the system prompt saying "never repeat". Small models don't
   obey negation, and near-static context at temp 0.3 converges to the same
   answer. **Fix:** code-enforced dedup (normalized compare against stream
   history), one hot retry at temp 0.85, then skip the turn. The session-3 log
   shows this firing 3 times — it was load-bearing on every turn.
3. **Prompt framing.** "Your recent suggestions / don't repeat" → replaced
   with "steps already completed (these are DONE) → output the NEXT step that
   comes AFTER". Positive framing measurably helped.
4. **Test harness bugs (also instructive):** a conductor that waited for 3
   suggestions deadlocked against the one-turn-per-human-move protocol; a
   second identical `!step` was deduped by design (commands fire once per
   unique text — documented, and the driver now uses distinct edits).

## Remaining limitations

- **Near-duplicates pass.** `- Define a Tool enum` vs `Define a Tool enum`
  differ only by a leading dash; normalization catches some, semantic repeats
  ("Add" vs "Define") don't get caught. For a 1.5B model this is a ceiling
  issue; your 3B will repeat less. If it becomes a problem, raise the base
  temperature or add an embedding-similarity check — overkill for the MVP.
- **The Architect tracks, it doesn't plan.** With no quest decomposition it
  just suggests the next obvious step. That's the Director's job (quest.md) —
  still on the roadmap.
- **Timing on CPU:** ~5 s per Architect turn, ~30–60 s per Reflex expansion
  (shared single-slot server serializes them — which accidentally mirrors the
  intended "dance" turn-taking). On your GPU this is near-instant.

## Transcript highlights (session 3, unedited)

```
[conductor] teaching: !fix ALLOWED = ['ls', 'pytest', 'git status'] => class Tool(Enum): LS='ls'; PYTEST='pytest'
— learned: `ALLOWED = ['ls', 'pytest', 'git status']` -> `class Tool(Enum)...` (saved to preferences.jsonl)
— repeat detected (attempt 1); retrying hot
-> Add a `Tool` enum with `LS`, `PYTEST`, and `GIT_STATUS` values.
— repeat detected (attempt 1); retrying hot
-> Add a human approval gate.
— repeat detected (attempt 1); retrying hot
-> - Implement the human approval gate function.
```

Full files in `session-logs/`: `workspace.md`, `stream.md`, `reflex.md`,
`preferences.jsonl`, plus console logs for all three processes.

## Verdict

The v1.0 design was protocol-sound but had three real bugs that only a live
model could expose. v1.1 (this folder) is the version that survived contact.
Next per the roadmap: the night-shift trainer on the collected
`preferences.jsonl`, or the sandboxed tool layer — which the system itself
just started designing for us (see the Reflex's approval-gate sketch).

## Addendum — v2.0 (the strip-down) validation

After the system was cut to a zero-dependency core (`loop.py`, stdlib-only),
the new code paths were re-validated against a deterministic Ollama-shaped
fake model — six client-side checks plus two server-side assertions, all
passing:

- turn-taking after goal edit · dedup + hot retry · fenced-reply parsing
- `preferences.jsonl` logging · no fence fragments · exact suggestion count
- **server-observed:** `quest.md` content reached the prompt; `!fix` rules
  reached the system prompt; the repeat-skip path fired cleanly

Note: the stripped core's behavior logic is byte-identical to the v1.1 that
survived the real-model live test; the fake-model run validates the rewritten
transport (stdlib `urllib` replacing `requests`) and the new extension point.
