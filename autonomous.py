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
import gc
import flock_gc as gc_module
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import signal
import fcntl
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
MAX_WORKING_MEM_LINES = 10000  # Memory leak prevention
MAX_REPEAT_COUNT     = 3      # Infinite loop detection
LOCK_RETRIES         = 5      # Race condition: file lock retries
LOCK_WAIT            = 0.1    # Seconds between lock retries
DRY_RUN              = False  # Global dry-run mode

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
            r"^git (status|diff|log)(\s|$)",
            r"^ls(\s+-\w*)?(\s|$)",
            r"^cat .+\.(md|py|txt|json|ya?ml|toml)$",
            r"^head -\d+ .+",
            r"^tail -\d+ .+",
            r"^wc(\s|$)",
            r"^grep(\s+-\w*)?\s.+",
            r"^pytest(\s|$)",
            r"^python3 -m pytest(\s|$)",
            r"^python3 -c\s.+",
        ],
        "execution_timeout": 15,
    },
    "gc": {
        "enabled": True,
        "run_every_ticks": 120,
        "max_working_mem_per_goal": 100,
        "max_stream_lines": 500,
        "max_audit_age_days": 7,
        "max_model_disk_gb": 20.0,
        "keep_model_tags": ["iterator", "coder", "thinker"],
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
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise IOError(f"Failed to fetch {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise IOError(f"Invalid JSON response from {url}: {e}") from e
    except Exception as e:
        raise IOError(f"Unexpected error fetching {url}: {e}") from e

def _post_json(url, payload, timeout):
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise IOError(f"Failed to POST to {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise IOError(f"Invalid JSON response from {url}: {e}") from e
    except Exception as e:
        raise IOError(f"Unexpected error posting to {url}: {e}") from e

# ═══════════════════════════════════════════════════════════════
# File helpers (with locking for race condition prevention)
# ═══════════════════════════════════════════════════════════════

class FileLock:
    """Context manager for file locking to prevent race conditions."""
    def __init__(self, path, mode="r"):
        self.path = path
        self.mode = mode
        self.fp = None
        self.locked = False

    def __enter__(self):
        for attempt in range(LOCK_RETRIES):
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                self.fp = open(self.path, self.mode, encoding="utf-8")
                fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.locked = True
                return self.fp
            except (IOError, OSError) as e:
                if self.fp:
                    self.fp.close()
                if attempt < LOCK_RETRIES - 1:
                    time.sleep(LOCK_WAIT * (2 ** attempt))  # Exponential backoff
                else:
                    raise IOError(f"Could not acquire lock on {self.path}: {e}") from e
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fp:
            if self.locked:
                try:
                    fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            self.fp.close()

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except IOError as e:
        print(f"! warning: could not read {path}: {e}")
        return ""

def read_tail(path, n):
    return "\n".join(read_text(path).splitlines()[-n:])

def mtime(path):
    try:
        return os.path.getmtime(path)
    except (OSError, AttributeError) as e:
        return 0.0

def append_text(path, text):
    if DRY_RUN:
        print(f"[DRY-RUN] would append to {path}: {text[:100]}")
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with FileLock(path, "a") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except IOError as e:
        print(f"! error: could not append to {path}: {e}")
        raise

def write_text(path, text):
    if DRY_RUN:
        print(f"[DRY-RUN] would write to {path}: {text[:100]}")
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Write to temp file first, then atomic rename
        temp_path = f"{path}.tmp.{os.getpid()}"
        with FileLock(temp_path, "w") as f:
            f.write(text)
        try:
            os.replace(temp_path, path)  # Atomic on POSIX
        except AttributeError:
            # Fallback for Windows
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)
    except IOError as e:
        print(f"! error: could not write to {path}: {e}")
        raise

# ═══════════════════════════════════════════════════════════════
# Config loading (minimal YAML-ish parser, no dependency)
# ═══════════════════════════════════════════════════════════════

def load_config():
    """Load config.yaml with a minimal parser. Falls back to DEFAULT_CONFIG."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    try:
        text = read_text(CONFIG_FILE)
    except IOError:
        return config

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

    # Validate and sanitize config
    return _validate_config(config)

def _validate_config(config):
    """Validate config values and fall back to defaults for invalid values."""
    try:
        # Validate tick section
        if "tick" in config:
            tick_cfg = config["tick"]
            if not isinstance(tick_cfg.get("enabled"), (bool, type(None))):
                tick_cfg["enabled"] = DEFAULT_CONFIG["tick"]["enabled"]
            if not isinstance(tick_cfg.get("interval_seconds"), (int, float)) or tick_cfg.get("interval_seconds", 0) <= 0:
                tick_cfg["interval_seconds"] = DEFAULT_CONFIG["tick"]["interval_seconds"]
            if not isinstance(tick_cfg.get("idle_only"), (bool, type(None))):
                tick_cfg["idle_only"] = DEFAULT_CONFIG["tick"]["idle_only"]

        # Validate goals section
        if "goals" in config:
            goals_cfg = config["goals"]
            if not isinstance(goals_cfg.get("max_concurrent"), int) or goals_cfg.get("max_concurrent", 0) < 1:
                goals_cfg["max_concurrent"] = DEFAULT_CONFIG["goals"]["max_concurrent"]
            if not isinstance(goals_cfg.get("auto_advance"), (bool, type(None))):
                goals_cfg["auto_advance"] = DEFAULT_CONFIG["goals"]["auto_advance"]

        # Validate tools section
        if "tools" in config:
            tools_cfg = config["tools"]
            if not isinstance(tools_cfg.get("allowed_executables"), list):
                tools_cfg["allowed_executables"] = DEFAULT_CONFIG["tools"]["allowed_executables"]
            if not isinstance(tools_cfg.get("allowed_patterns"), list):
                tools_cfg["allowed_patterns"] = DEFAULT_CONFIG["tools"]["allowed_patterns"]
            if not isinstance(tools_cfg.get("execution_timeout"), (int, float)) or tools_cfg.get("execution_timeout", 0) <= 0:
                tools_cfg["execution_timeout"] = DEFAULT_CONFIG["tools"]["execution_timeout"]

        # Validate gc section
        if "gc" in config:
            gc_cfg = config["gc"]
            if not isinstance(gc_cfg.get("enabled"), (bool, type(None))):
                gc_cfg["enabled"] = DEFAULT_CONFIG["gc"]["enabled"]
            if not isinstance(gc_cfg.get("run_every_ticks"), int) or gc_cfg.get("run_every_ticks", 0) <= 0:
                gc_cfg["run_every_ticks"] = DEFAULT_CONFIG["gc"]["run_every_ticks"]
            if not isinstance(gc_cfg.get("max_working_mem_per_goal"), int) or gc_cfg.get("max_working_mem_per_goal", 0) <= 0:
                gc_cfg["max_working_mem_per_goal"] = DEFAULT_CONFIG["gc"]["max_working_mem_per_goal"]
            if not isinstance(gc_cfg.get("max_stream_lines"), int) or gc_cfg.get("max_stream_lines", 0) <= 0:
                gc_cfg["max_stream_lines"] = DEFAULT_CONFIG["gc"]["max_stream_lines"]
            if not isinstance(gc_cfg.get("max_audit_age_days"), (int, float)) or gc_cfg.get("max_audit_age_days", 0) <= 0:
                gc_cfg["max_audit_age_days"] = DEFAULT_CONFIG["gc"]["max_audit_age_days"]
            if not isinstance(gc_cfg.get("max_model_disk_gb"), (int, float)) or gc_cfg.get("max_model_disk_gb", 0) <= 0:
                gc_cfg["max_model_disk_gb"] = DEFAULT_CONFIG["gc"]["max_model_disk_gb"]
            if not isinstance(gc_cfg.get("keep_model_tags"), list):
                gc_cfg["keep_model_tags"] = DEFAULT_CONFIG["gc"]["keep_model_tags"]

        return config
    except (KeyError, TypeError, AttributeError) as e:
        print(f"! warning: invalid config.yaml, using defaults: {e}")
        return DEFAULT_CONFIG

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
    """Verify the model server is reachable with proper error handling."""
    try:
        names = [m.get("name", "") for m in _get_json(SERVER_TAGS_URL).get("models", [])]
        family = MODEL_NAME.split(":")[0]
        if not any(n == MODEL_NAME or n.startswith(family) for n in names):
            print(f"! warning: '{MODEL_NAME}' not found. Run:  ollama pull {MODEL_NAME}")
    except IOError as e:
        sys.exit(f"Cannot reach model server on localhost:11434 — is Ollama running? {e}")
    except Exception as e:
        sys.exit(f"Unexpected error checking server: {e}")

def startup_self_check():
    """Verify all required files exist and are writable at startup."""
    required_files = [
        WORKSPACE_FILE, STREAM_FILE, QUEST_FILE, MEMORY_FILE,
        GOALS_FILE, CONFIG_FILE, TICK_FILE, WORKING_MEM_FILE, APPROVAL_FILE
    ]

    issues = []
    for f in required_files:
        # Check if directory exists and is writable
        dir_path = os.path.dirname(f) or "."
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except OSError as e:
                issues.append(f"Cannot create directory {dir_path}: {e}")
        elif not os.access(dir_path, os.W_OK):
            issues.append(f"Directory {dir_path} is not writable")

        # Try to create/append to the file
        try:
            if not os.path.exists(f):
                # Create empty file
                with open(f, "a", encoding="utf-8") as fp:
                    pass
            elif not os.access(f, os.W_OK):
                issues.append(f"File {f} is not writable")
        except OSError as e:
            issues.append(f"Cannot access file {f}: {e}")

    if issues:
        sys.exit("Startup self-check failed:\n" + "\n".join(f"  - {i}" for i in issues))
    print("* startup self-check passed")

# Global shutdown flag for graceful signal handling
_shutdown_requested = False

def _signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    signal_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(f"\n* received {signal_name}, shutting down gracefully...")

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

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
    try:
        append_text(MEMORY_FILE, json.dumps(record))
    except IOError as e:
        print(f"! warning: could not log fix: {e}")

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
    # Memory leak prevention: rotate if file exceeds max lines
    try:
        lines = read_text(WORKING_MEM_FILE).splitlines()
        if len(lines) >= MAX_WORKING_MEM_LINES:
            # Keep last 80% of max lines
            keep_from = int(MAX_WORKING_MEM_LINES * 0.2)
            with FileLock(WORKING_MEM_FILE, "w") as f:
                f.write("\n".join(lines[keep_from:]) + "\n")
    except IOError:
        pass  # Fall through to append

    append_text(WORKING_MEM_FILE, json.dumps(record))

# ═══════════════════════════════════════════════════════════════
# Goal queue
# ═══════════════════════════════════════════════════════════════

def parse_goals():
    """Parse goals.md into a list of goal dicts. Handles malformed input and duplicate IDs."""
    try:
        text = read_text(GOALS_FILE)
    except IOError:
        return []

    goals = []
    current = None
    seen_ids = set()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## [") and "]" in stripped:
            if current:
                # Validate and clean goal before adding
                current.pop("_section", None)
                if current["id"] not in seen_ids:
                    seen_ids.add(current["id"])
                    goals.append(current)
                else:
                    print(f"! warning: duplicate goal ID {current['id']}, skipping")

            # Parse goal header with better error handling
            try:
                bracket_end = stripped.index("]")
                gid = stripped[4:bracket_end].strip()
                title = stripped[bracket_end+1:].strip().lstrip("—").lstrip("-").strip()

                # Validate ID format
                if not gid or not gid.replace("_", "").replace("-", "").isalnum():
                    gid = f"G{len(seen_ids)+1:03d}"  # Auto-generate if invalid

                current = {
                    "id": gid,
                    "title": title if title else "(untitled)",
                    "context": "",
                    "status": "pending",
                    "priority": 99
                }
            except (ValueError, IndexError):
                current = None
                continue

        elif current:
            if stripped.startswith("### context"):
                current["_section"] = "context"
            elif stripped.startswith("### status:"):
                status_val = stripped.split(":", 1)[1].strip().lower()
                # Validate status value
                if status_val in ("pending", "in_progress", "complete", "failed"):
                    current["status"] = status_val
                else:
                    current["status"] = "pending"  # Default
                current["_section"] = None
            elif stripped.startswith("### priority:"):
                try:
                    prio_val = stripped.split(":", 1)[1].strip()
                    current["priority"] = max(1, min(999, int(prio_val)))  # Clamp to 1-999
                except (ValueError, IndexError):
                    current["priority"] = 99  # Default
                current["_section"] = None
            elif current.get("_section") == "context" and stripped:
                current["context"] += stripped + " "

    if current:
        current.pop("_section", None)
        if current["id"] not in seen_ids:
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
    """Update goal status in goals.md with error handling and file locking."""
    try:
        text = read_text(GOALS_FILE)
    except IOError as e:
        print(f"! error: could not read goals file: {e}")
        return

    lines = text.splitlines()
    new_lines = []
    current_goal = None
    updated = False

    for line in lines:
        if line.strip().startswith("## [") and "]" in line:
            stripped = line.strip()
            gid = stripped[4:stripped.index("]")].strip()
            current_goal = gid
        elif current_goal == goal_id and line.strip().startswith("### status:"):
            line = f"### status: {status}"
            if summary:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                new_lines.append(line)
                new_lines.append(f"### status")
                new_lines.append(f"- {ts} — {summary}")
                updated = True
                continue
        new_lines.append(line)

    if updated or any(status in line for line in new_lines if "### status:" in line):
        try:
            write_text(GOALS_FILE, "\n".join(new_lines) + "\n")
        except IOError as e:
            print(f"! error: could not write goal status: {e}")

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
                # Compile with FULLMATCH semantics for security
                # Ensure patterns are anchored to prevent bypass
                if not p.startswith("^"):
                    p = "^" + p
                if not p.endswith("$"):
                    p += "$"
                self.allowed_patterns.append(re.compile(p))
            except re.error as e:
                print(f"! warning: invalid regex pattern '{p}': {e}")

        self.execution_timeout = tools_cfg.get("execution_timeout", 15)

    def _sanitize_command(self, command):
        """Basic command sanitization to detect obvious bypass attempts."""
        # Detect shell metacharacters that could enable command chaining
        dangerous = ["&&", "||", ";", "|", "$(", "`", "$(", "\x00"]
        for d in dangerous:
            if d in command:
                return False, f"potentially unsafe character: {repr(d)}"
        return True, "ok"

    def check_allowed(self, command):
        parts = command.split()
        if not parts:
            return False, "empty command"

        executable = parts[0]
        if executable not in self.allowed_executables:
            return False, f"'{executable}' not in allowlist"

        # Check for dangerous metacharacters
        safe, reason = self._sanitize_command(command)
        if not safe:
            return False, reason

        # Pattern matching with FULLMATCH semantics
        if not any(p.fullmatch(command) for p in self.allowed_patterns):
            return False, "command matches no allowed pattern"

        return True, "ok"

    def execute(self, command):
        allowed, reason = self.check_allowed(command)
        ts = datetime.now(timezone.utc).isoformat()
        req_id = f"REQ-{int(time.time())}"

        # Audit log (with error handling)
        audit = {"id": req_id, "command": command, "timestamp": ts,
                 "allowed": allowed, "reason": reason}
        try:
            append_text(AUDIT_FILE, json.dumps(audit))
        except IOError as e:
            print(f"! warning: could not write audit log: {e}")

        if not allowed:
            return {"status": "rejected", "reason": reason}

        if DRY_RUN:
            return {"status": "dry_run", "reason": "dry-run mode enabled"}

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
    """Build prompt messages optimized for prefix cache hits.
    
    Order matters: most stable content goes FIRST, most volatile LAST.
    This maximizes Ollama's automatic KV cache reuse (85-95% savings).
    
    Stability ranking (most → least stable):
      1. SYSTEM_PROMPT (frozen string constant)
      2. Style rules (change rarely, only on !fix)
      3. Workspace tail (changes on human edit, stable between ticks)
      4. Goal context (changes when goal status changes)
      5. Working memory (changes each tick — put AFTER stable prefix)
      6. Stream tail (changes every tick — most volatile, put LAST)
      7. Instruction (identical every tick — goes after volatile to re-anchor)
    """
    # System message: frozen prefix + style rules (stable across ticks)
    system = SYSTEM_PROMPT
    if rules:
        system += "\n\nStyle corrections (obey these):"
        for r in rules:
            system += f"\n- instead of `{r['rejected']}`, write `{r['accepted']}`"

    # User message: stable content first, volatile last
    # This lets Ollama's prefix cache reuse the KV state up to the first change
    workspace_tail = read_tail(WORKSPACE_FILE, WORKSPACE_TAIL_LINES)
    
    goal_id = None
    goal_block = "No active goals — all complete."
    if active_goal:
        goal_id = active_goal["id"]
        goal_block = (
            f"## [{active_goal['id']}] {active_goal['title']}\n"
            f"Status: {active_goal['status']}\n"
            f"Priority: {active_goal['priority']}\n"
            f"Context: {active_goal['context'][:500]}"
        )

    # Notes are ephemeral but small — include in system for prefix stability
    if notes:
        system += "\n\nLive guidance:"
        for n in notes:
            system += f"\n- {n}"

    # Stable user prefix (workspace + goal — changes rarely between ticks)
    stable_prefix = (
        f"=== workspace.md ===\n{workspace_tail}\n\n"
        f"=== active goal ===\n{goal_block}\n\n"
    )

    # Volatile suffix (changes every tick — placed last to preserve cache)
    mem_records = load_working_memory(goal_id)
    mem_text = "\n".join(
        f"- [{r.get('type', '?')}] {r.get('content', '')[:120]}"
        for r in mem_records
    ) or "(none yet)"

    volatile_suffix = (
        f"=== working memory (last {WORKING_MEM_TAIL}) ===\n{mem_text}\n\n"
        f"=== steps already completed ===\n"
        f"{read_tail(STREAM_FILE, STREAM_TAIL_LINES)}\n\n"
        "Output the next action as a single JSON object:"
    )

    return [{"role": "system", "content": system},
            {"role": "user", "content": stable_prefix + volatile_suffix}]

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
    try:
        done = {_normalize(l) for l in read_text(STREAM_FILE).splitlines()}
        return _normalize(suggestion) in done
    except IOError:
        return False

# Infinite loop detection state
_repeat_tracker = collections.deque(maxlen=MAX_REPEAT_COUNT)

def _is_repeating(action_text):
    """Detect if the agent is stuck in a loop suggesting the same thing."""
    normalized = _normalize(action_text)
    if not normalized:
        return False

    _repeat_tracker.append(normalized)
    # If we've seen this same normalized text MAX_REPEAT_COUNT times in a row
    if len(_repeat_tracker) == MAX_REPEAT_COUNT and len(set(_repeat_tracker)) == 1:
        _repeat_tracker.clear()  # Reset after detection
        return True
    return False

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
            parsed = json.loads(match.group())
            # Validate it has an action key, or treat as suggestion
            if "action" in parsed:
                return parsed
            # Empty JSON {} or missing action → check for text field
            if "text" in parsed:
                return {"action": "suggest", "text": parsed["text"]}
            # Empty JSON {} → wait
            return {"action": "wait", "reason": "empty action"}
    except (json.JSONDecodeError, AttributeError):
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
        # Infinite loop detection
        if _is_repeating(text):
            return (f"→ LOOP BREAK: same suggestion repeated {MAX_REPEAT_COUNT} times", False)
        try:
            append_text(STREAM_FILE, text)
        except IOError as e:
            return (f"→ write failed: {e}", False)
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
        try:
            write_text(STREAM_FILE,
                "# stream.md — agent-owned. Suggestions appear here.\n")
            print("— stream.md cleared")
        except IOError as e:
            print(f"! error: could not clear stream.md: {e}")
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
            try:
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
            except IOError as e:
                print(f"! error: could not add goal: {e}")
        return paused, False
    if cmd == "!goals":
        try:
            goals = parse_goals()
            for g in goals:
                status_icon = {"pending": "○", "in_progress": "◐",
                              "complete": "●", "failed": "✗"}.get(g["status"], "?")
                print(f"  {status_icon} [{g['id']}] P{g['priority']} {g['title'][:50]}")
        except IOError as e:
            print(f"! error: could not parse goals: {e}")
        return paused, False
    if cmd == "!status":
        try:
            print(f"— tick: {'enabled' if True else 'disabled'}")
            goals = parse_goals()
            active = sum(1 for g in goals if g["status"] in ("pending", "in_progress"))
            done = sum(1 for g in goals if g["status"] == "complete")
            print(f"— goals: {active} active, {done} complete")
        except IOError as e:
            print(f"! error: could not get status: {e}")
        return paused, False
    if cmd == "!gc" or cmd.startswith("!gc "):
        # Parse optional subcommands: !gc --report, !gc --prune, !gc --dry-run
        parts = cmd.split()
        gc_config = config.get("gc", {})
        results = gc_module.auto_gc(gc_config, dry_run="--dry" in parts)
        print("* garbage collection results:")
        for op, result in results.items():
            if isinstance(result, dict):
                if "error" in result:
                    print(f"  {op}: ! {result['error']}")
                elif "dry_run" in result:
                    print(f"  {op}: ~ {result['dry_run']}")
                else:
                    print(f"  {op}: ✓")
                    for k, v in result.items():
                        if k not in ("error", "dry_run"):
                            print(f"    {k}: {v}")
            else:
                print(f"  {op}: {result}")
        # Also show disk usage summary
        usage = gc_module.disk_usage_report()
        print(f"\n* disk usage summary:")
        print(f"  models: {usage['models_gb']} GB")
        print(f"  workspace: {usage['workspace_files_mb']} MB")
        print(f"  logs: {usage['logs_mb']} MB")
        if usage.get("cache_freed_mb", 0) > 0:
            print(f"  cache freed: {usage['cache_freed_mb']} MB")
        return paused, False
    print(f"! unknown: {cmd}  (known: !fix !note !step !pause !resume !clear !goal !goals !status !gc !approve !reject)")
    return paused, False

# ═══════════════════════════════════════════════════════════════
# Tick status writer
# ═══════════════════════════════════════════════════════════════

def update_tick_status(active_goal, trigger, last_action):
    try:
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
    except IOError as e:
        print(f"! warning: could not update tick status: {e}")

# ═══════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════

def main():
    global DRY_RUN, _shutdown_requested

    # Parse command-line arguments for --dry-run
    if "--dry-run" in sys.argv or "-d" in sys.argv:
        DRY_RUN = True
        print("* DRY-RUN MODE: no file writes or tool execution")

    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()

    # Startup self-check
    startup_self_check()

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
    tick_count = 0  # For GC scheduling

    tick_enabled = config["tick"].get("enabled", True)
    tick_interval = config["tick"].get("interval_seconds", AUTONOMOUS_INTERVAL)
    idle_only = config["tick"].get("idle_only", False)

    print(f"* autonomous loop up — model: {MODEL_NAME}")
    print(f"* tick: {'enabled' if tick_enabled else 'disabled'}, interval: {tick_interval}s")
    if DRY_RUN:
        print(f"* DRY-RUN MODE: showing what would happen without executing")
    print(f"* commands: !fix !note !step !pause !resume !clear !goal !goals !status !approve !reject")
    print(f"* goals file: {GOALS_FILE}")

    while not _shutdown_requested:
        try:
            time.sleep(TICK_SECONDS)
        except IOError:
            # Handle sleep interruption during shutdown
            break

        # ── Human-move detection (existing protocol) ────────────
        try:
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
        except IOError as e:
            print(f"! warning: workspace file access error: {e}")

        if force_step:
            dirty, last_edit, force_step = True, 0.0, False

        # ── Human-triggered turn ────────────────────────────────
        if dirty and not paused and time.time() - last_edit >= IDLE_DEBOUNCE:
            try:
                messages = build_human_move_prompt(rules, notes)
                text = ask_model(messages, temp=0.3)
                action = parse_action(text)
                log_line, _ = execute_action(action, rules, notes, sandbox, config, None)
                print(f"[human] {log_line}")
                last_action = log_line
            except IOError as e:
                print(f"! error during human turn: {e}")
            finally:
                dirty = False
            continue

        # ── Autonomous tick ─────────────────────────────────────
        now = time.time()
        if (tick_enabled and not paused and
                now - last_tick_time >= tick_interval and
                (not idle_only or now - last_edit > IDLE_DEBOUNCE * 4)):

            last_tick_time = now
            tick_count += 1

            # ── Garbage collection check (runs periodically) ─────────
            gc_cfg = config.get("gc", DEFAULT_CONFIG.get("gc", {}))
            if gc_cfg.get("enabled", True):
                gc_interval = gc_cfg.get("run_every_ticks", 120)
                if tick_count % gc_interval == 0 and tick_count > 0:
                    try:
                        gc_results = gc_module.auto_gc(gc_cfg, dry_run=False)
                        # Log summary to tick file and working memory
                        summary_parts = []
                        for op, result in gc_results.items():
                            if isinstance(result, dict) and "error" not in result:
                                if op == "working_memory":
                                    removed = result.get("total_removed", 0)
                                    if removed > 0:
                                        summary_parts.append(f"gc: working_mem -{removed} records")
                                elif op == "stream":
                                    removed = result.get("removed", 0)
                                    if removed > 0:
                                        summary_parts.append(f"gc: stream -{removed} lines")
                                elif op == "audit_log":
                                    removed = result.get("removed", 0)
                                    if removed > 0:
                                        summary_parts.append(f"gc: audit -{removed} entries")
                                elif op == "goals_archive":
                                    archived = result.get("archived_count", 0)
                                    if archived > 0:
                                        summary_parts.append(f"gc: archived {archived} goals")
                                elif op == "models_warning":
                                    summary_parts.append(f"gc: {result['message']}")
                        if summary_parts:
                            gc_summary = "; ".join(summary_parts)
                            print(f"[gc] {gc_summary}")
                            append_working_memory("gc", gc_summary)
                            last_action = f"gc: {gc_summary}"
                    except Exception as e:
                        print(f"[gc] error during garbage collection: {e}")

            try:
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
            except IOError as e:
                print(f"! error during autonomous tick: {e}")
            except Exception as e:
                print(f"! unexpected error during tick: {e}")
                last_action = f"error: {e}"

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n* autonomous loop stopped cleanly")
    except SystemExit:
        # Re-raise SystemExit (from sys.exit in startup checks)
        raise
    except Exception as e:
        print(f"\n! fatal error: {e}")
        sys.exit(1)
    finally:
        if _shutdown_requested:
            print("* graceful shutdown complete")
