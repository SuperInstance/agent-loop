#!/usr/bin/env python3
"""
autonomous.py — the constantly-thinking local agent.

Extends loop.py with:
  1. Autonomous tick — fires on a timer, not just on human input
  2. Goal queue — works through goals.md independently
  3. Tool sandbox — allowlisted shell commands, file I/O
  4. Working memory — persistent across restarts
  5. Safety fences — autonomous vs needs-approval vs forbidden

Philosophy: the autopilot holds the heading you set.
It does not chart the course. It thinks continuously within the fence.

Zero dependencies. Python 3.9+. Stdlib only.
Resource cost: ~20 MB RSS, near-zero CPU between ticks.
"""

import collections
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

MODEL_NAME      = "qwen2.5-coder:3b"
SERVER_CHAT_URL = "http://localhost:11434/api/chat"
SERVER_TAGS_URL = "http://localhost:11434/api/tags"
NUM_CTX         = 8192

# Files (existing protocol)
WORKSPACE_FILE  = "workspace.md"
STREAM_FILE     = "stream.md"
QUEST_FILE      = "quest.md"
MEMORY_FILE     = "preferences.jsonl"

# New files (autonomous extension)
GOALS_FILE      = "goals.md"
WORKING_MEM_FILE = "working_memory.jsonl"
CONFIG_FILE     = "config.yaml"
TICK_FILE       = "tick.md"
AUDIT_FILE      = "tools/audit.jsonl"
APPROVAL_FILE   = "approval.md"

# Tick settings
TICK_SECONDS         = 1.0      # main loop poll interval
AUTONOMOUS_INTERVAL  = 30.0     # seconds between autonomous ticks
IDLE_DEBOUNCE        = 2.5      # wait for typing quiet
WORKSPACE_TAIL_LINES = 80
STREAM_TAIL_LINES    = 20
QUEST_TAIL_LINES     = 40
MAX_FIX_RULES        = 5
REQUEST_TIMEOUT      = 120
WORKING_MEM_TAIL     = 10

# Default config (overridden by config.yaml if present)
DEFAULT_CONFIG = {
    "tick": {
        "enabled": True,
        "interval_seconds": AUTONOMOUS_INTERVAL,
        "idle_only": False,
    },
    "goals": {
        "max_concurrent": 1,
        "auto_advance": True,
    },
    "safety": {
        "autonomous": [
            "read_file", "write_to_owned_file", "append_to_stream",
            "tick_status_update", "goal_status_update", "working_memory_append",
            "list_files", "search_files",
        ],
        "needs_approval": [
            "shell_execute", "write_to_non_owned_file",
            "file_delete", "git_commit", "network_request",
        ],
        "forbidden": [
            "write_to_config", "write_to_audit_log",
            "modify_core_binaries", "system_configuration",
        ],
    },
    "tools": {
        "allowed_executables": [
            "ls", "cat", "grep", "head", "tail", "wc",
            "pytest", "git", "python3",
        ],
        "allowed_patterns": [
            r"^git (status|diff|log)",
            r"^ls -?\w*\s",
            r"^cat .+\.(md|py|txt|json|ya?ml|toml)$",
            r"^head -\d+ .+",
            r"^tail -\d+ .+",
            r"^wc .+",
            r"^grep -\w* .+",
            r"^pytest (tests/|test_)?",
            r"^python3 -m pytest",
            r"^python3 -c .+",
        ],
        "execution_timeout": 15,
    },
}

SYSTEM_PROMPT = (
    "You are the Autopilot — a local autonomous agent that thinks continuously. "
    "You operate within a conservation law: you hold the heading the human set. "
    "You do not chart a new course without approval.\n\n"
    "Your output is a single JSON object on one line:\n"
    '  {"action": "suggest", "text": "<one-line micro-step>"}\n'
    '  {"action": "tool", "tool": "shell_execute", "command": "<allowlisted cmd>"}\n'
    '  {"action": "memory", "type": "<context|observation|decision>", "content": "<note>"}\n'
    '  {"action": "complete_goal", "goal_id": "<id>", "summary": "<text>"}\n'
    '  {"action": "request_approval", "reason": "<why>", "action_desc": "<what>"}\n'
    '  {"action": "wait", "reason": "<why waiting>"}\n\n'
    "Always move work FORWARD. Never repeat a completed step. "
    "Prefer autonomous actions. Only request approval when truly needed."
)

# ═══════════════════════════════════════════════════════════════
# Tiny HTTP (stdlib only)
# ═══════════════════════════════════════════════════════════════

def _get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def _post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

# ═══════════════════════════════════════════════════════════════
# File helpers
# ═══════════════════════════════════════════════════════════════

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

def append_text(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")

def write_text(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ═══════════════════════════════════════════════════════════════
# Config loading (minimal YAML-ish parser, no dependency)
# ═══════════════════════════════════════════════════════════════

def load_config():
    """Load config.yaml with a minimal parser. Falls back to DEFAULT_CONFIG."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    text = read_text(CONFIG_FILE)
    if not text:
        return config

    section = None
    subsection = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            section = stripped[:-1]
            subsection = None
            if section not in config:
                config[section] = {}
        elif line.startswith("  ") and not line.startswith("    "):
            subsection = stripped.rstrip(":")
            if section and subsection:
                if subsection not in config[section]:
                    config[section][subsection] = []
        elif line.startswith("    ") and stripped.startswith("- "):
            value = stripped[2:].strip()
            if section and subsection and isinstance(config[section].get(subsection), list):
                config[section][subsection].append(value)
        elif ":" in stripped and section:
            key, _, val = stripped.partition(":")
            key, val = key.strip(), val.strip()
            if val and not val.endswith(":"):
                try:
                    if "." in val:
                        config[section][key] = float(val)
                    elif val.lower() in ("true", "false"):
                        config[section][key] = val.lower() == "true"
                    else:
                        config[section][key] = int(val) if val.isdigit() else val
                except ValueError:
                    config[section][key] = val

    return config

# ═══════════════════════════════════════════════════════════════
# File initialization
# ═══════════════════════════════════════════════════════════════

def ensure_files():
    # Existing protocol files
    if not os.path.exists(WORKSPACE_FILE):
        write_text(WORKSPACE_FILE,
            "# workspace.md — you own this file\n\n"
            "Write a goal below, or a !command:\n\n## Goal: \n")
    if not os.path.exists(STREAM_FILE):
        write_text(STREAM_FILE,
            "# stream.md — agent-owned. Suggestions appear here.\n")

    # New autonomous files
    if not os.path.exists(GOALS_FILE):
        write_text(GOALS_FILE,
            "# goals.md — active goals for the autonomous agent\n\n"
            "Add goals in this format:\n\n"
            "## [G001] Goal title\n"
            "### context\n"
            "Describe what needs to be done.\n"
            "### status: pending\n"
            "### priority: 1\n")
    if not os.path.exists(CONFIG_FILE):
        write_text(CONFIG_FILE,
            "# config.yaml — safety fences for the autonomous agent\n"
            "# This file is HUMAN-ONLY. The agent never writes here.\n\n"
            "tick:\n"
            "  enabled: true\n"
            "  interval_seconds: 30.0\n"
            "  idle_only: false\n\n"
            "goals:\n"
            "  max_concurrent: 1\n"
            "  auto_advance: true\n\n"
            "# Uncomment to restrict tools:\n"
            "# tools:\n"
            "#   allowed_executables:\n"
            "#     - ls\n"
            "#     - cat\n"
            "#     - pytest\n")
    if not os.path.exists(TICK_FILE):
        write_text(TICK_FILE,
            "# tick.md — autonomous agent status\n\n"
            "**Status:** starting\n")

    os.makedirs("tools", exist_ok=True)

def check_server():
    try:
        names = [m.get("name", "") for m in _get_json(SERVER_TAGS_URL).get("models", [])]
        family = MODEL_NAME.split(":")[0]
        if not any(n == MODEL_NAME or n.startswith(family) for n in names):
            print(f"! warning: '{MODEL_NAME}' not found. Run:  ollama pull {MODEL_NAME}")
    except Exception:
        sys.exit("Cannot reach model server on localhost:11434 — is Ollama running?")

# ═══════════════════════════════════════════════════════════════
# Memory (style corrections + working memory)
# ═══════════════════════════════════════════════════════════════

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
    append_text(MEMORY_FILE, json.dumps(record))

def load_working_memory(goal_id=None, limit=WORKING_MEM_TAIL):
    records = []
    for line in read_text(WORKING_MEM_FILE).splitlines():
        try:
            record = json.loads(line)
            rid = record.get("goal_id")
            if rid is None or goal_id is None or rid == goal_id:
                records.append(record)
        except json.JSONDecodeError:
            continue
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records[:limit]

def append_working_memory(record_type, content, goal_id=None):
    record = {
        "type": record_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "goal_id": goal_id,
        "content": content,
    }
    append_text(WORKING_MEM_FILE, json.dumps(record))

# ═══════════════════════════════════════════════════════════════
# Goal queue
# ═══════════════════════════════════════════════════════════════

def parse_goals():
    """Parse goals.md into a list of goal dicts."""
    text = read_text(GOALS_FILE)
    goals = []
    current = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## [") and "]" in stripped:
            if current:
                goals.append(current)
            gid = stripped[4:stripped.index("]")]
            title = stripped[stripped.index("]")+1:].strip().lstrip("—").strip()
            current = {"id": gid, "title": title, "context": "",
                       "status": "pending", "priority": 99}
        elif current:
            if stripped.startswith("### context"):
                current["_section"] = "context"
            elif stripped.startswith("### status:"):
                current["status"] = stripped.split(":", 1)[1].strip()
                current["_section"] = None
            elif stripped.startswith("### priority:"):
                try:
                    current["priority"] = int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    pass
                current["_section"] = None
            elif current.get("_section") == "context" and stripped:
                current["context"] += stripped + " "

    if current:
        goals.append(current)
    return goals

def select_active_goal(goals, config):
    candidates = [g for g in goals if g["status"] in ("pending", "in_progress")]
    max_concurrent = config["goals"].get("max_concurrent", 1)
    in_progress = [g for g in candidates if g["status"] == "in_progress"]
    if len(in_progress) >= max_concurrent:
        candidates = in_progress
    candidates.sort(key=lambda g: (g["priority"], g["id"]))
    return candidates[0] if candidates else None

def update_goal_status(goal_id, status, summary=None):
    """Update goal status in goals.md."""
    text = read_text(GOALS_FILE)
    lines = text.splitlines()
    new_lines = []
    current_goal = None

    for line in lines:
        if line.strip().startswith("## [") and "]" in line:
            gid = line.strip()[4:line.strip().index("]")]
            current_goal = gid
        elif current_goal == goal_id and line.strip().startswith("### status:"):
            line = f"### status: {status}"
            if summary:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                new_lines.append(line)
                new_lines.append(f"### status")
                new_lines.append(f"- {ts} — {summary}")
                continue
        new_lines.append(line)

    write_text(GOALS_FILE, "\n".join(new_lines) + "\n")

# ═══════════════════════════════════════════════════════════════
# Tool sandbox
# ═══════════════════════════════════════════════════════════════

class ToolSandbox:
    def __init__(self, config):
        tools_cfg = config.get("tools", DEFAULT_CONFIG["tools"])
        self.allowed_executables = set(tools_cfg.get("allowed_executables", []))
        raw_patterns = tools_cfg.get("allowed_patterns", [])
        self.allowed_patterns = []
        for p in raw_patterns:
            try:
                self.allowed_patterns.append(re.compile(p))
            except re.error:
                pass
        self.execution_timeout = tools_cfg.get("execution_timeout", 15)

    def check_allowed(self, command):
        parts = command.split()
        if not parts:
            return False, "empty command"
        executable = parts[0]
        if executable not in self.allowed_executables:
            return False, f"'{executable}' not in allowlist"
        if not any(p.match(command) for p in self.allowed_patterns):
            return False, "command matches no allowed pattern"
        return True, "ok"

    def execute(self, command):
        allowed, reason = self.check_allowed(command)
        ts = datetime.now(timezone.utc).isoformat()
        req_id = f"REQ-{int(time.time())}"

        # Audit log
        audit = {"id": req_id, "command": command, "timestamp": ts,
                 "allowed": allowed, "reason": reason}
        append_text(AUDIT_FILE, json.dumps(audit))

        if not allowed:
            return {"status": "rejected", "reason": reason}

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=self.execution_timeout
            )
            return {
                "status": "completed",
                "exit_code": result.returncode,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:500],
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "reason": f"exceeded {self.execution_timeout}s"}
        except Exception as e:
            return {"status": "error", "reason": str(e)}

# ═══════════════════════════════════════════════════════════════
# Prompt assembly
# ═══════════════════════════════════════════════════════════════

def build_autonomous_prompt(rules, active_goal, config, notes):
    system = SYSTEM_PROMPT
    if rules:
        system += "\n\nStyle corrections (obey these):"
        for r in rules:
            system += f"\n- instead of `{r['rejected']}`, write `{r['accepted']}`"
    if notes:
        system += "\n\nLive guidance:"
        for n in notes:
            system += f"\n- {n}"

    goal_text = "No active goals — all complete."
    goal_id = None
    if active_goal:
        goal_id = active_goal["id"]
        goal_text = (
            f"## [{active_goal['id']}] {active_goal['title']}\n"
            f"Status: {active_goal['status']}\n"
            f"Priority: {active_goal['priority']}\n"
            f"Context: {active_goal['context'][:500]}"
        )

    mem_records = load_working_memory(goal_id)
    mem_text = "\n".join(
        f"- [{r.get('type', '?')}] {r.get('content', '')[:120]}"
        for r in mem_records
    ) or "(none yet)"

    user = (
        f"=== active goal ===\n{goal_text}\n\n"
        f"=== working memory (last {WORKING_MEM_TAIL}) ===\n{mem_text}\n\n"
        f"=== workspace.md (last {WORKSPACE_TAIL_LINES} lines) ===\n"
        f"{read_tail(WORKSPACE_FILE, WORKSPACE_TAIL_LINES)}\n\n"
        f"=== steps already completed ===\n"
        f"{read_tail(STREAM_FILE, STREAM_TAIL_LINES)}\n\n"
        "Output the next action as a single JSON object:"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

def build_human_move_prompt(rules, notes):
    """Same as original loop.py prompt — for human-triggered turns."""
    system = SYSTEM_PROMPT
    if rules:
        system += "\n\nStyle corrections (obey these):"
        for r in rules:
            system += f"\n- instead of `{r['rejected']}`, write `{r['accepted']}`"
    if notes:
        system += "\n\nLive guidance:"
        for n in notes:
            system += f"\n- {n}"

    user = ""
    if os.path.exists(QUEST_FILE):
        quest = read_tail(QUEST_FILE, QUEST_TAIL_LINES)
        if quest.strip():
            user += f"=== active quest ===\n{quest}\n\n"
    user += (
        f"=== workspace.md (last {WORKSPACE_TAIL_LINES} lines) ===\n"
        f"{read_tail(WORKSPACE_FILE, WORKSPACE_TAIL_LINES)}\n\n"
        f"=== steps already completed ===\n"
        f"{read_tail(STREAM_FILE, STREAM_TAIL_LINES)}\n\n"
        'Output: {"action": "suggest", "text": "<next micro-step>"}'
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

# ═══════════════════════════════════════════════════════════════
# Model call + action parsing
# ═══════════════════════════════════════════════════════════════

def _normalize(line):
    return line.strip().lower().lstrip("-*#> `").rstrip(".")

def _already_done(suggestion):
    done = {_normalize(l) for l in read_text(STREAM_FILE).splitlines()}
    return _normalize(suggestion) in done

def ask_model(messages, temp=0.4):
    payload = {
        "model": MODEL_NAME, "messages": messages,
        "stream": False,
        "options": {"temperature": temp, "num_predict": 200, "num_ctx": NUM_CTX},
    }
    try:
        resp = _post_json(SERVER_CHAT_URL, payload, REQUEST_TIMEOUT)
        return resp.get("message", {}).get("content", "")
    except Exception as e:
        print(f"! model error: {e}")
        return ""

def parse_action(text):
    """Parse model output into an action dict."""
    # Try JSON parse first
    try:
        # Find first JSON object in the text
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Fallback: treat as a suggestion
    for line in text.splitlines():
        line = line.strip().strip("`")
        if line and not line.startswith("```"):
            return {"action": "suggest", "text": line[:200]}

    return {"action": "wait", "reason": "no parseable output"}

# ═══════════════════════════════════════════════════════════════
# Action execution
# ═══════════════════════════════════════════════════════════════

def execute_action(action, rules, notes, sandbox, config, active_goal):
    """Execute an action from the model. Returns (log_line, did_something)."""
    act = action.get("action", "wait")
    goal_id = active_goal["id"] if active_goal else None

    if act == "suggest":
        text = action.get("text", "").strip()
        if not text:
            return ("→ empty suggestion", False)
        if _already_done(text):
            return (f"→ repeat detected: {text[:60]}", False)
        append_text(STREAM_FILE, text)
        return (f"→ {text}", True)

    if act == "tool":
        tool = action.get("tool", "")
        command = action.get("command", "")

        if tool == "shell_execute":
            result = sandbox.execute(command)
            if result["status"] == "completed":
                stdout = result.get("stdout", "")[:300]
                append_working_memory("observation",
                    f"Tool: `{command}` → exit {result['exit_code']}\n{stdout}",
                    goal_id)
                return (f"→ tool: {command} ✓", True)
            elif result["status"] == "rejected":
                return (f"→ tool rejected: {result['reason']}", False)
            else:
                return (f"→ tool failed: {result.get('reason', 'unknown')}", False)
        else:
            return (f"→ unknown tool: {tool}", False)

    if act == "memory":
        mem_type = action.get("type", "context")
        content = action.get("content", "")
        if content:
            append_working_memory(mem_type, content, goal_id)
            return (f"→ memory[{mem_type}]: {content[:60]}", True)
        return ("→ empty memory", False)

    if act == "complete_goal":
        gid = action.get("goal_id", goal_id)
        summary = action.get("summary", "Completed.")
        if gid and config["goals"].get("auto_advance", True):
            update_goal_status(gid, "complete", summary)
            append_working_memory("goal_complete", summary, gid)
            return (f"→ completed goal {gid}: {summary[:60]}", True)
        return (f"→ goal complete signal (no auto-advance)", True)

    if act == "request_approval":
        reason = action.get("reason", "")
        desc = action.get("action_desc", "")
        ts = datetime.now(timezone.utc).isoformat()
        approval = (f"## Approval Request [{ts}]\n\n"
                    f"**Action:** {desc}\n"
                    f"**Reason:** {reason}\n\n"
                    f"Respond in workspace.md: `!approve` or `!reject`\n---\n")
        append_text(APPROVAL_FILE, approval)
        return (f"→ approval requested: {desc[:60]}", True)

    if act == "wait":
        reason = action.get("reason", "unspecified")
        return (f"→ waiting: {reason[:60]}", False)

    return (f"→ unknown action: {act}", False)

# ═══════════════════════════════════════════════════════════════
# Commands (extended from original loop.py)
# ═══════════════════════════════════════════════════════════════

def handle_command(cmd, rules, notes, paused):
    if cmd == "!pause":
        return True, False
    if cmd == "!resume":
        return False, False
    if cmd == "!step":
        return paused, True
    if cmd == "!clear":
        write_text(STREAM_FILE,
            "# stream.md — agent-owned. Suggestions appear here.\n")
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
            print("! usage:  !fix <rejected> => <preferred>")
            return paused, False
        rejected, accepted = (s.strip() for s in body.split("=>", 1))
        if rejected and accepted:
            rules.append({"rejected": rejected, "accepted": accepted})
            del rules[:-MAX_FIX_RULES:]
            log_fix(rejected, accepted)
            print(f"— learned: `{rejected}` -> `{accepted}`")
        return paused, False
    if cmd == "!approve":
        print("— approval granted (stub — full approval flow in v2)")
        return paused, False
    if cmd == "!reject":
        print("— request rejected")
        return paused, False
    if cmd.startswith("!goal "):
        # Quick goal add: !goal Write tests for the reflex engine
        title = cmd[len("!goal "):].strip()
        if title:
            existing = read_text(GOALS_FILE)
            num = existing.count("## [G") + 1
            gid = f"G{num:03d}"
            block = (f"\n## [{gid}] {title}\n"
                     f"### context\n"
                     f"(add context here)\n"
                     f"### status: pending\n"
                     f"### priority: {num}\n")
            append_text(GOALS_FILE, block)
            print(f"— added goal {gid}: {title}")
        return paused, False
    if cmd == "!goals":
        goals = parse_goals()
        for g in goals:
            status_icon = {"pending": "○", "in_progress": "◐",
                          "complete": "●", "failed": "✗"}.get(g["status"], "?")
            print(f"  {status_icon} [{g['id']}] P{g['priority']} {g['title'][:50]}")
        return paused, False
    if cmd == "!status":
        print(f"— tick: {'enabled' if True else 'disabled'}")
        goals = parse_goals()
        active = sum(1 for g in goals if g["status"] in ("pending", "in_progress"))
        done = sum(1 for g in goals if g["status"] == "complete")
        print(f"— goals: {active} active, {done} complete")
        return paused, False
    print(f"! unknown: {cmd}  (known: !fix !note !step !pause !resume !clear !goal !goals !status !approve !reject)")
    return paused, False

# ═══════════════════════════════════════════════════════════════
# Tick status writer
# ═══════════════════════════════════════════════════════════════

def update_tick_status(active_goal, trigger, last_action):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    next_ts = datetime.now(timezone.utc).timestamp() + AUTONOMOUS_INTERVAL
    next_str = datetime.fromtimestamp(next_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    goal_text = "none"
    if active_goal:
        goal_text = f"[{active_goal['id']}] {active_goal['title'][:50]}"

    content = (
        f"# tick.md — autonomous agent status\n\n"
        f"**Last tick:** {ts}\n"
        f"**Next tick:** {next_str}\n"
        f"**Trigger:** {trigger}\n"
        f"**Active goal:** {goal_text}\n"
        f"**Last action:** {last_action[:80]}\n\n"
        f"---\n"
        f"*This file is owned by the autonomous agent. Read-only for humans.*\n"
    )
    write_text(TICK_FILE, content)

# ═══════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════

def main():
    ensure_files()
    check_server()

    config = load_config()
    sandbox = ToolSandbox(config)
    rules = load_rules()
    notes = collections.deque(maxlen=3)
    seen_commands = set()
    paused = dirty = force_step = False
    last_edit = time.time()
    workspace_seen = read_text(WORKSPACE_FILE)
    last_workspace_mtime = mtime(WORKSPACE_FILE)
    last_tick_time = time.time()
    last_action = "starting"

    tick_enabled = config["tick"].get("enabled", True)
    tick_interval = config["tick"].get("interval_seconds", AUTONOMOUS_INTERVAL)
    idle_only = config["tick"].get("idle_only", False)

    print(f"* autonomous loop up — model: {MODEL_NAME}")
    print(f"* tick: {'enabled' if tick_enabled else 'disabled'}, interval: {tick_interval}s")
    print(f"* commands: !fix !note !step !pause !resume !clear !goal !goals !status !approve !reject")
    print(f"* goals file: {GOALS_FILE}")

    while True:
        time.sleep(TICK_SECONDS)

        # ── Human-move detection (existing protocol) ────────────
        current_mtime = mtime(WORKSPACE_FILE)
        if current_mtime != last_workspace_mtime:
            last_workspace_mtime = current_mtime
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

        if force_step:
            dirty, last_edit, force_step = True, 0.0, False

        # ── Human-triggered turn ────────────────────────────────
        if dirty and not paused and time.time() - last_edit >= IDLE_DEBOUNCE:
            messages = build_human_move_prompt(rules, notes)
            text = ask_model(messages, temp=0.3)
            action = parse_action(text)
            log_line, _ = execute_action(action, rules, notes, sandbox, config, None)
            print(f"[human] {log_line}")
            last_action = log_line
            dirty = False
            continue

        # ── Autonomous tick ─────────────────────────────────────
        now = time.time()
        if (tick_enabled and not paused and
                now - last_tick_time >= tick_interval and
                (not idle_only or now - last_edit > IDLE_DEBOUNCE * 4)):

            last_tick_time = now

            goals = parse_goals()
            active_goal = select_active_goal(goals, config)

            update_tick_status(active_goal, "autonomous", last_action)

            if not active_goal:
                # All goals complete — gentle idle
                append_working_memory("context",
                    "All goals complete. Agent idling. Awaiting new goals.")
                print(f"[tick] all goals complete — idling")
                last_action = "idle — all goals complete"
                continue

            # Mark goal as in_progress if pending
            if active_goal["status"] == "pending":
                update_goal_status(active_goal["id"], "in_progress")
                active_goal["status"] = "in_progress"
                print(f"[tick] starting goal [{active_goal['id']}] {active_goal['title'][:50]}")

            # Ask the model for the next action
            messages = build_autonomous_prompt(rules, active_goal, config, notes)
            text = ask_model(messages, temp=0.5)

            if not text.strip():
                print(f"[tick] model returned empty — retrying next tick")
                last_action = "empty model response"
                continue

            action = parse_action(text)
            log_line, did_something = execute_action(
                action, rules, notes, sandbox, config, active_goal)

            print(f"[tick] {log_line}")
            last_action = log_line

            update_tick_status(active_goal, "autonomous", last_action)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n* autonomous loop stopped cleanly")
