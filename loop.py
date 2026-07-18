#!/usr/bin/env python3
"""
loop.py — the whole product. One file, zero dependencies, Python 3.9+.

A tiny self-improving pair-programming loop:
  workspace.md — YOU own this: goals, code, !commands
  stream.md    — the AGENT owns this: one-line micro-steps
  preferences.jsonl — your corrections, kept forever (Alpaca-shaped)

It talks only to a local model server on localhost. No accounts, no cloud,
no telemetry, no git, no shell tools. It reads/writes exactly the files
listed above (plus quest.md if you create one). Everything else — reflex
completions, test harness, future trainer — is an OPTIONAL separate process
in extras/ that speaks the same file protocol. Run them or don't.

Resource cost: ~15 MB RSS for this process, near-zero CPU (mtime-polled
once a second). The only real cost is the model, and you choose its size.
"""

import collections
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

# ----------------------------- configuration --------------------------------
# Edit these four lines; that's the entire config surface.

MODEL_NAME      = "qwen2.5-coder:3b"   # ollama pull qwen2.5-coder:3b (or :1.5b for minimum RAM)
SERVER_CHAT_URL = "http://localhost:11434/api/chat"
SERVER_TAGS_URL = "http://localhost:11434/api/tags"
NUM_CTX         = 8192                 # Ollama defaults to 2048 — too small for this app

WORKSPACE_FILE = "workspace.md"        # human-owned
STREAM_FILE    = "stream.md"           # agent-owned
QUEST_FILE     = "quest.md"            # OPTIONAL: read only if you create it
MEMORY_FILE    = "preferences.jsonl"

TICK_SECONDS         = 1.0
IDLE_DEBOUNCE        = 2.5             # agent waits for this much typing quiet
WORKSPACE_TAIL_LINES = 80
STREAM_TAIL_LINES    = 20
QUEST_TAIL_LINES     = 40
MAX_FIX_RULES        = 5
REQUEST_TIMEOUT      = 120

SYSTEM_PROMPT = (
    "You are the Architect in a two-file pair-programming loop. "
    "The human owns workspace.md. You may ONLY suggest the next micro-step: "
    "a single line of code, a comment, or a pseudocode step, under 15 words. "
    "No explanations. No markdown fences. Always move the work FORWARD."
)

# --------------------------- tiny http (stdlib) ------------------------------

def _get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def _post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

# ------------------------------- files ---------------------------------------

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def read_tail(path, n):
    return "\n".join(read_text(path).splitlines()[-n:])

def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

def ensure_files():
    if not os.path.exists(WORKSPACE_FILE):
        with open(WORKSPACE_FILE, "w", encoding="utf-8") as f:
            f.write("# workspace.md — you own this file\n\n"
                    "Write a goal below, or a !command:\n\n## Goal: \n")
    if not os.path.exists(STREAM_FILE):
        with open(STREAM_FILE, "w", encoding="utf-8") as f:
            f.write("# stream.md — agent-owned. Suggestions appear here.\n")

def check_server():
    try:
        names = [m.get("name", "") for m in _get_json(SERVER_TAGS_URL).get("models", [])]
        family = MODEL_NAME.split(":")[0]
        if not any(n == MODEL_NAME or n.startswith(family) for n in names):
            print(f"! warning: '{MODEL_NAME}' not found. Run:  ollama pull {MODEL_NAME}")
    except Exception:
        sys.exit("Cannot reach the model server on localhost:11434 — is Ollama running?")

# ---------------------------- memory (the point) -----------------------------

def load_rules():
    rules = []
    for line in read_text(MEMORY_FILE).splitlines():
        try:
            d = json.loads(line)
            rules.append({"rejected": d["input"], "accepted": d["output"]})
        except (json.JSONDecodeError, KeyError):
            continue
    return rules[-MAX_FIX_RULES:]

def log_fix(rejected, accepted):
    record = {"instruction": "Apply the user's style correction in future code.",
              "input": rejected, "output": accepted,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# ------------------------------ the model ------------------------------------

def build_messages(rules, notes):
    system = SYSTEM_PROMPT
    if rules:
        system += "\n\nThe user has corrected your style before. Obey these:"
        for r in rules:
            system += f"\n- instead of `{r['rejected']}`, write `{r['accepted']}`"
    if notes:
        system += "\n\nLive guidance from the user:"
        for n in notes:
            system += f"\n- {n}"

    user = ""
    if os.path.exists(QUEST_FILE):           # the one extension point:
        quest = read_tail(QUEST_FILE, QUEST_TAIL_LINES)   # a Director tool can
        if quest.strip():                                 # drop a quest.md here
            user += f"=== active quest ===\n{quest}\n\n"
    user += (
        f"=== workspace.md (last {WORKSPACE_TAIL_LINES} lines) ===\n"
        f"{read_tail(WORKSPACE_FILE, WORKSPACE_TAIL_LINES)}\n\n"
        f"=== steps already completed (these are DONE — do not redo them) ===\n"
        f"{read_tail(STREAM_FILE, STREAM_TAIL_LINES)}\n\n"
        "Output the single NEXT micro-step that comes AFTER the completed steps:"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

def _first_line(text):
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):      # skip fences BEFORE stripping backticks
            continue
        line = line.strip("`")
        if line:
            return line
    return ""

def _normalize(line):
    return line.strip().lower().lstrip("-*#> `").rstrip(".")

def _already_done(suggestion):
    done = {_normalize(l) for l in read_text(STREAM_FILE).splitlines()}
    return _normalize(suggestion) in done

def ask_architect(rules, notes):
    """One turn, with code-enforced anti-repetition (small models need it)."""
    for attempt, temp in enumerate((0.3, 0.85)):
        payload = {"model": MODEL_NAME, "messages": build_messages(rules, notes),
                   "stream": False,
                   "options": {"temperature": temp, "num_predict": 64, "num_ctx": NUM_CTX}}
        try:
            text = _post_json(SERVER_CHAT_URL, payload, REQUEST_TIMEOUT) \
                       .get("message", {}).get("content", "")
        except Exception as e:
            print(f"! model server unreachable: {e}")
            return ""
        suggestion = _first_line(text)
        if suggestion and not _already_done(suggestion):
            return suggestion
        print(f"— repeat detected (attempt {attempt + 1}); "
              + ("retrying hot" if attempt == 0 else "skipping turn"))
    return ""

# ------------------------------ commands -------------------------------------
# Typed by you, in workspace.md. Each unique command line fires once.

def handle_command(cmd, rules, notes, paused):
    """Returns (paused, force_step)."""
    if cmd == "!pause":
        print("— paused (type !resume in workspace.md to continue)")
        return True, False
    if cmd == "!resume":
        print("— resumed")
        return False, False
    if cmd == "!step":
        return paused, True
    if cmd == "!clear":
        with open(STREAM_FILE, "w", encoding="utf-8") as f:
            f.write("# stream.md — agent-owned. Suggestions appear here.\n")
        print("— stream.md cleared")
        return paused, False
    if cmd.startswith("!note "):
        note = cmd[len("!note "):].strip()
        if note:
            notes.append(note)
            print(f"— noted: {note}")
        return paused, False
    if cmd.startswith("!fix "):
        body = cmd[len("!fix "):]
        if "=>" not in body:
            print("! usage:  !fix <rejected snippet> => <preferred snippet>")
            return paused, False
        rejected, accepted = (s.strip() for s in body.split("=>", 1))
        if rejected and accepted:
            rules.append({"rejected": rejected, "accepted": accepted})
            del rules[:-MAX_FIX_RULES]
            log_fix(rejected, accepted)
            print(f"— learned: `{rejected}` -> `{accepted}` (saved to {MEMORY_FILE})")
        return paused, False
    print(f"! unknown command: {cmd}  (known: !fix !note !step !pause !resume !clear)")
    return paused, False

# -------------------------------- main ----------------------------------------

def main():
    ensure_files()
    check_server()

    rules = load_rules()
    notes = collections.deque(maxlen=3)
    seen_commands = set()
    paused = dirty = force_step = False
    last_edit = time.time()
    workspace_seen = read_text(WORKSPACE_FILE)
    initial_content = workspace_seen
    last_mtime = mtime(WORKSPACE_FILE)

    print(f"* loop up — watching {WORKSPACE_FILE}, writing {STREAM_FILE}")
    print("* commands: !fix a => b | !note txt | !step | !pause | !resume | !clear")

    while True:
        time.sleep(TICK_SECONDS)
        current_mtime = mtime(WORKSPACE_FILE)      # stat first: near-zero CPU
        if current_mtime != last_mtime:
            last_mtime = current_mtime
            workspace = read_text(WORKSPACE_FILE)
            if workspace != workspace_seen:
                workspace_seen = workspace
                last_edit = time.time()
                dirty = True
                for line in workspace.splitlines():
                    cmd = line.strip()
                    if cmd.startswith("!") and cmd not in seen_commands:
                        seen_commands.add(cmd)
                        paused, force_step = handle_command(cmd, rules, notes, paused)

        if workspace_seen == initial_content:
            dirty = False
        if force_step:
            dirty, last_edit, force_step = True, 0.0, False
        if not dirty or paused or time.time() - last_edit < IDLE_DEBOUNCE:
            continue

        suggestion = ask_architect(rules, notes)
        dirty = False                              # one suggestion per human move
        if suggestion:
            with open(STREAM_FILE, "a", encoding="utf-8") as f:
                f.write(suggestion + "\n")
            print(f"-> {suggestion}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n* loop stopped cleanly")
