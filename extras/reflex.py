#!/usr/bin/env python3
"""
extras/reflex.py — OPTIONAL. Local CodeGeeX stand-in ("Reflex Engine").

You do NOT need this if you already use Copilot, Continue, CodeGeeX, or any
other autocomplete — those already do this job better. This exists for people
who want the reflex layer fully local and dependency-free.

Run from the same directory as loop.py. Watches stream.md; when the Architect
drops a new micro-step, expands it into a minimal code block in reflex.md
(its own file — the file protocol is the module API, no coupling to loop.py).
Accept a completion by copying it into workspace.md.

Cost: one extra small Python process (~15 MB) + shares your local model.
"""

import json
import time
import urllib.request

SERVER_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME      = "qwen2.5-coder:3b"   # keep in sync with loop.py
STREAM_FILE     = "stream.md"
REFLEX_FILE     = "reflex.md"
WORKSPACE_FILE  = "workspace.md"
POLL_SECONDS    = 0.5

REFLEX_SYSTEM = (
    "You are a reflex autocomplete engine (like inline ghost-text in an IDE). "
    "Given workspace context and a one-line intent from the Architect, output "
    "ONLY a minimal code block (3-8 lines) implementing that intent. "
    "No prose, no explanation. Wrap in a single ```python fence."
)

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def expand(suggestion):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": REFLEX_SYSTEM},
            {"role": "user", "content":
                f"=== workspace context (tail) ===\n"
                f"{chr(10).join(read_text(WORKSPACE_FILE).splitlines()[-40:])}\n\n"
                f"=== intent to complete ===\n{suggestion}\n\nCode only:"},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 160, "num_ctx": 4096},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER_CHAT_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode("utf-8")) \
                     .get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"# reflex error: {e}"

def main():
    if not read_text(REFLEX_FILE):
        with open(REFLEX_FILE, "w", encoding="utf-8") as f:
            f.write("# reflex.md — Reflex Engine (CodeGeeX stand-in) output.\n")
    seen = set(read_text(STREAM_FILE).splitlines())
    print("* reflex hot — watching stream.md (optional extra, Ctrl-C to stop)")

    while True:
        time.sleep(POLL_SECONDS)
        lines = read_text(STREAM_FILE).splitlines()
        new = [l for l in lines if l not in seen and l.strip()
               and not l.startswith("#") and not l.startswith("!")]
        for suggestion in new:
            seen.add(suggestion)
            print(f"reflex completing: {suggestion[:60]}")
            with open(REFLEX_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n#### intent: `{suggestion}`\n{expand(suggestion)}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
