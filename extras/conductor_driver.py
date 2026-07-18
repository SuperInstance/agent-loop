#!/usr/bin/env python3
"""
conductor_driver.py v2 — scripted human for the live test.

Fixed to respect the loop's real protocol: the Architect takes ONE turn per
human move. So every Conductor action IS the human move that re-arms the
loop: seed goal, steer with !note, accept a reflex completion, teach a !fix,
and force extra turns with !step.
"""

import re
import sys
import time

WORKSPACE = "workspace.md"
STREAM = "stream.md"
REFLEX = "reflex.md"

GOAL = """# workspace.md — you own this file

## Goal: Design v2 of this agent loop — the sandboxed tool layer.
The Architect (you) gets shell tools in v2. We need: an allowlist of safe
commands, a human approval gate, and an audit log. Sketch concrete Python.
"""

NOTE = "!note prefer concrete dataclasses and enums over loose dicts"
FIX = "!fix ALLOWED = ['ls', 'pytest', 'git status'] => class Tool(Enum): LS='ls'; PYTEST='pytest'"

def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def append(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def suggestion_count():
    return len([l for l in read(STREAM).splitlines()
                if l.strip() and not l.startswith("#")])

def wait_for_suggestions(n, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if suggestion_count() >= n:
            return True
        time.sleep(1.0)
    return False

def gate(n, label, timeout=240):
    print(f"[conductor] waiting for {n} suggestion(s) — {label}...", flush=True)
    if not wait_for_suggestions(n, timeout):
        print(f"[conductor] TIMEOUT at {label}", flush=True)
        sys.exit(1)
    time.sleep(2)

def main():
    print("[conductor] seeding goal: v2 sandboxed tool layer (the system ideates about itself)", flush=True)
    with open(WORKSPACE, "w", encoding="utf-8") as f:
        f.write(GOAL)
    time.sleep(1)

    base = suggestion_count()
    print(f"[conductor] boot — stream already has {base} suggestion(s)", flush=True)

    # Move 1: steer with a !note (this edit re-arms the loop)
    print(f"[conductor] steering: {NOTE}", flush=True)
    append(WORKSPACE, NOTE)
    gate(base + 1, "after !note")

    # Move 2: accept the first fenced reflex block into workspace.md (Tab moment)
    print("[conductor] waiting for a reflex completion to accept...", flush=True)
    t0 = time.time()
    while "```python" not in read(REFLEX) and time.time() - t0 < 180:
        time.sleep(2)
    m = re.search(r"```python\n(.*?)```", read(REFLEX), re.DOTALL)
    if m:
        print("[conductor] accepting first reflex completion into workspace.md", flush=True)
        append(WORKSPACE, f"\n# Conductor accepted this reflex completion:\n{m.group(0)}\n")
    else:
        print("[conductor] no reflex block yet; skipping acceptance", flush=True)
        append(WORKSPACE, "# Conductor: keep going, focus on the allowlist data structure")
    gate(base + 2, "after acceptance")

    # Move 3: teach a style correction
    print(f"[conductor] teaching: {FIX}", flush=True)
    append(WORKSPACE, FIX)
    gate(base + 3, "after !fix")

    # Moves 4-5: force one autonomous step with !step, then a plain-edit steer
    # (a repeated identical !step line would be deduped by the orchestrator —
    # commands fire once per unique text, so move 5 uses a normal human edit)
    print("[conductor] forcing !step", flush=True)
    append(WORKSPACE, "!step")
    gate(base + 4, "after !step")
    print("[conductor] final steer via plain edit (approval gate focus)", flush=True)
    append(WORKSPACE, "# Conductor: one more — sketch the human approval gate now")
    gate(base + 5, "after final steer")

    print(f"[conductor] session complete — {suggestion_count()} suggestions total", flush=True)

if __name__ == "__main__":
    main()
