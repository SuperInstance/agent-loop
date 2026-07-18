# EXTENSION_DESIGN.md — Autonomous Agent Extension

**Version:** 1.0  
**Date:** 2026-07-18  
**Status:** Design specification (implementation pending)

---

## Overview

This document specifies how to extend the agent-loop from a reactive pair-programming system to an autonomous thinking agent while preserving all six core mechanisms (M1-M6) from WHITEPAPER.md. The design maintains the zero-dependency, stdlib-only philosophy and one-file-per-concern structure.

**Key constraint:** The core loop.py remains unchanged. Extensions are additive, new files only.

---

## 1. File Protocol Additions

### 1.1 New Files

| File | Writer (owner) | Readers | Created by | Format | Purpose |
|------|----------------|---------|------------|--------|---------|
| `goals.md` | human or Director | core v2 | core v2 (template) | Markdown with goal blocks | Multiple active goals the agent works through |
| `config.yaml` | human only | core v2, tools | human | YAML configuration | Safety fences, tool allowlists, tick settings |
| `working_memory.jsonl` | core v2 (append-only) | core v2 (boot), trainers | core v2 | JSONL, one record per thought | Persistent working memory across restarts |
| `tools/` directory | tools layer | core v2 | human | README.md + per-tool .md | Tool specifications and audit logs |
| `tools/audit.jsonl` | tools layer (append) | human, auditors | tools layer | JSONL | Audit trail of all tool invocations |
| `tick.md` | core v2 | human, diagnostics | core v2 | Markdown | Tick status, last fire time, current goal |

### 1.2 Ownership Rules (extends PROTOCOL.md I1)

**New invariants:**
- **I7 — Tick ownership.** `tick.md` is owned by core v2; humans may read it but never write.
- **I8 — Goal immutability during execution.** While a goal is `in_progress`, core v2 may append `### status` sub-headers to goals.md but never modifies the goal text itself.
- **I9 — Tool audit trail.** Every tool invocation writes exactly one line to `tools/audit.jsonl` before execution, with `status: "pending"`, updated to `status: "completed"` or `"failed"` after.
- **I10 — Config is human-only.** `config.yaml` is never written by any agent process. Runtime reconfiguration requires a human edit and process restart.

### 1.3 File Formats

#### goals.md
```markdown
# Active Goals

## [G001] Write a reflex engine test
### context
The reflex engine needs unit tests for the expansion logic.
### status: pending
### priority: 2
### assigned: architect

## [G002] Design tool sandbox
### context
We need an allowlist-based tool layer before exposing shell access.
### status: in_progress
### priority: 1
### assigned: architect
### status
- 2026-07-18T10:23:00Z — started, analyzing attack surface
```

#### config.yaml
```yaml
# Autonomous tick settings
tick:
  enabled: true
  interval_seconds: 30.0  # time between autonomous ticks
  idle_only: false        # if true, only tick when no human activity

# Goal queue settings
goals:
  max_concurrent: 2
  auto_advance: true       # move to next goal when current completes
  completion_threshold: 0.8  # agent must declare 80% "done" to mark complete

# Safety classification
safety:
  # Operations that can run autonomously
  autonomous:
    - read_file
    - write_to_owned_file
    - append_to_stream
    - tick_status_update
    - goal_status_update
    - working_memory_append
  
  # Operations requiring human approval
  needs_approval:
    - write_to_non_owned_file
    - shell_execute
    - network_request
    - file_delete
    - git_commit
  
  # Operations that are never allowed
  forbidden:
    - write_to_config
    - write_to_audit_log
    - modify_core_binaries
    - system_configuration

# Tool allowlist
tools:
  allowed_executables:
    - ls
    - cat
    - grep
    - pytest
    - git status
    - git diff
  
  allowed_patterns:
    - "^git (status|diff|log)"
    - "^ls -"
    - "^cat .+\\.md$"
    - "^pytest (tests/|test_)"
    - "^grep -r"

  # Timeouts (seconds)
  execution_timeout: 10
  network_timeout: 30
```

#### working_memory.jsonl
```json
{"type": "context", "timestamp": "2026-07-18T10:00:00Z", "goal_id": "G002", "content": "The tool sandbox needs an allowlist pattern matcher. Simple regex is sufficient for v1."}
{"type": "observation", "timestamp": "2026-07-18T10:05:00Z", "goal_id": "G002", "content": "Reflex.py uses urllib.request - we should standardize on that for network ops."}
{"type": "decision", "timestamp": "2026-07-18T10:10:00Z", "goal_id": "G002", "content": "Decision: Use subprocess.run with shell=False for exec, capture stdout/stderr for audit."}
{"type": "goal_complete", "timestamp": "2026-07-18T10:30:00Z", "goal_id": "G002", "summary": "Tool sandbox v1 designed with allowlist and audit trail."}
```

---

## 2. Autonomous Tick Mechanism

### 2.1 Tick Architecture

The autonomous tick is NOT a separate process. It is an additional clock source in the core loop's main event loop.

**New state variables (core v2):**
```python
last_tick_time = time.time()
tick_interval = config["tick"]["interval_seconds"]
autonomous_enabled = config["tick"]["enabled"]
```

**Extended main loop:**
```python
while True:
    time.sleep(TICK_SECONDS)
    
    # Existing: human-move clocking (M2)
    current_mtime = mtime(WORKSPACE_FILE)
    if current_mtime != last_mtime:
        # ... existing human move handling ...
    
    # NEW: autonomous tick
    now = time.time()
    if autonomous_enabled and now - last_tick_time >= tick_interval:
        if not config["tick"]["idle_only"] or now - last_edit > IDLE_DEBOUNCE * 2:
            last_tick_time = now
            handle_autonomous_tick(rules, notes)
    
    # Existing: dirty check and suggestion
    if dirty and not paused:
        # ... existing architect call ...
```

### 2.2 Tick Prompt Assembly

When firing on autonomous tick, the prompt differs from human-move prompts:

```python
def build_tick_prompt(rules, active_goals, working_memory_tail):
    system = SYSTEM_PROMPT + (
        "\n\nYou are operating in autonomous mode. "
        "Review the active goals and working memory, then output "
        "the NEXT micro-step for the highest-priority incomplete goal. "
        "If all goals are complete, output: !all_goals_complete"
    )
    # + style rules from preferences.jsonl
    
    user = (
        f"=== active goals ===\n{active_goals}\n\n"
        f"=== working memory (last 10 records) ===\n{working_memory_tail}\n\n"
        f"=== workspace.md (last {WORKSPACE_TAIL_LINES} lines) ===\n"
        f"{read_tail(WORKSPACE_FILE, WORKSPACE_TAIL_LINES)}\n\n"
        f"=== stream.md (last {STREAM_TAIL_LINES} lines) ===\n"
        f"{read_tail(STREAM_FILE, STREAM_TAIL_LINES)}\n\n"
        "Output the next micro-step (or !all_goals_complete):"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
```

### 2.3 Tick.md Output

The tick process writes its state to `tick.md` for observability:

```markdown
# Tick Status

**Last tick:** 2026-07-18T10:23:15Z  
**Next tick:** 2026-07-18T10:23:45Z  
**Trigger:** autonomous  
**Active goal:** G002  
**Suggestion:** "Add allowlist pattern matcher to tool sandbox"

---
*This file is owned by core v2. Read-only for humans.*
```

---

## 3. Goal Queue Architecture

### 3.1 Goal Lifecycle

```
pending → in_progress → complete → archived
          ↓              ↓
       paused         failed
```

**Transitions:**
- `pending → in_progress`: Agent selects goal (highest priority first)
- `in_progress → complete`: Agent declares completion OR human marks complete
- `in_progress → failed`: Agent declares impasse (requires human intervention)
- `in_progress → paused`: Human pauses via `!pause_goal <id>`
- `complete → archived`: Human moves to archive (or auto-archived after N days)

### 3.2 Goal Selection Algorithm

```python
def select_active_goal(goals):
    """Select the highest-priority unblocked goal."""
    candidates = [g for g in goals if g["status"] in ("pending", "in_progress")]
    
    # Filter by capacity (max_concurrent from config.yaml)
    in_progress_count = sum(1 for g in candidates if g["status"] == "in_progress")
    if in_progress_count >= config["goals"]["max_concurrent"]:
        candidates = [g for g in candidates if g["status"] == "in_progress"]
    
    # Sort by priority (1 = highest), then by creation time
    candidates.sort(key=lambda g: (g["priority"], g.get("created", "")))
    
    return candidates[0] if candidates else None
```

### 3.3 Goal Completion Detection

The agent signals completion by suggesting `!complete_goal <id> <summary>`. When this appears in stream.md:

```python
def handle_completion_signal(suggestion):
    """Parse and record goal completion."""
    if suggestion.startswith("!complete_goal"):
        parts = suggestion.split()
        goal_id = parts[1]
        summary = " ".join(parts[2:])
        
        # Update goals.md status
        update_goal_status(goal_id, "complete", summary)
        
        # Log to working memory
        append_working_memory({
            "type": "goal_complete",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "goal_id": goal_id,
            "summary": summary
        })
        
        # Move to next goal if auto_advance enabled
        if config["goals"]["auto_advance"]:
            advance_to_next_goal(goal_id)
```

---

## 4. Tool Sandbox Design

### 4.1 Tool Layer Architecture

The tool layer is a SEPARATE process (`tools/tool_layer.py`), speaking the file protocol:

**Files:**
- `tools/request.md` — written by core v2, tool invocations appear here
- `tools/response.md` — written by tool layer, results appear here
- `tools/audit.jsonl` — audit trail (append-only)
- `tools/spec/TOOL_NAME.md` — per-tool specifications

**Request format:**
```markdown
## [REQ-001] shell_execute

### command
ls -la

### safety_class
needs_approval

### status: pending

### timestamp
2026-07-18T10:00:00Z
```

**Response format:**
```markdown
## [REQ-001] shell_execute

### status
completed

### exit_code
0

### stdout
total 24
drwxr-xr-x 6 user user 4096 Jul 18 10:00 .

### stderr
(empty)

### duration_seconds
0.02
```

### 4.2 Allowlist Engine

```python
import subprocess
import re
from pathlib import Path

class ToolSandbox:
    def __init__(self, config):
        self.allowed_executables = set(config["tools"]["allowed_executables"])
        self.allowed_patterns = [re.compile(p) for p in config["tools"]["allowed_patterns"]]
        self.execution_timeout = config["tools"]["execution_timeout"]
    
    def check_allowed(self, command):
        """Return (allowed, reason) tuple."""
        parts = command.split()
        executable = parts[0]
        
        # Check executable allowlist
        if executable not in self.allowed_executables:
            return False, f"executable '{executable}' not in allowlist"
        
        # Check pattern allowlist
        full_match = any(p.match(command) for p in self.allowed_patterns)
        if not full_match:
            return False, "command does not match any allowed pattern"
        
        return True, "ok"
    
    def execute(self, command):
        """Execute with timeout, return result dict."""
        allowed, reason = self.check_allowed(command)
        if not allowed:
            return {"status": "rejected", "reason": reason}
        
        try:
            result = subprocess.run(
                command,
                shell=True,  # Safe because of allowlist
                capture_output=True,
                text=True,
                timeout=self.execution_timeout
            )
            return {
                "status": "completed",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_seconds": 0.0  # measured
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "reason": f"exceeded {self.execution_timeout}s"}
```

### 4.3 Safety Classification

Tool operations fall into three classes:

| Class | Approval | Audit | Examples |
|-------|----------|-------|----------|
| `autonomous` | None | Yes | read_file, tick_status_update, working_memory_append |
| `needs_approval` | Human | Yes | shell_execute, write_to_non_owned_file, network_request |
| `forbidden` | Never | N/A | write_to_config, modify_core_binaries |

**Approval mechanism:**
```python
def handle_approval_required(tool_request):
    """Write to approval.md and pause until human response."""
    with open("approval.md", "a") as f:
        f.write(f"## Approve? {tool_request['type']}\n\n")
        f.write(f"### Command\n{tool_request['command']}\n\n")
        f.write(f"### Risk\n{tool_request['risk_assessment']}\n\n")
        f.write("### Response\nType `!approve {id}` or `!reject {id}` in workspace.md\n")
    
    # Wait for human !approve or !reject command
    # (poll workspace.md, similar to existing command handling)
```

### 4.4 Audit Log

Every tool invocation writes to `tools/audit.jsonl`:

```json
{"id": "REQ-001", "type": "shell_execute", "command": "ls -la", "safety_class": "needs_approval", "status": "approved", "timestamp": "2026-07-18T10:00:00Z", "duration_seconds": 0.02}
{"id": "REQ-002", "type": "read_file", "path": "workspace.md", "safety_class": "autonomous", "status": "completed", "timestamp": "2026-07-18T10:01:00Z", "duration_seconds": 0.001}
```

---

## 5. Working Memory Architecture

### 5.1 Memory Types

| Type | Purpose | Retention |
|------|---------|-----------|
| `context` | Facts learned about the codebase | Indefinite |
| `observation` | Notes from execution/testing | Indefinite |
| `decision` | Architectural decisions made | Indefinite |
| `goal_complete` | Goal completion summaries | Indefinite |
| `temp` | Ephemeral scratch | 24 hours (auto-cleanup) |

### 5.2 Memory Lookup

During tick and human-move turns, relevant memory is loaded:

```python
def load_relevant_memory(goal_id, limit=10):
    """Load memory records relevant to the current goal."""
    records = []
    for line in read_text("working_memory.jsonl").splitlines():
        try:
            record = json.loads(line)
            # Include if no goal_id (general) or matching goal_id
            if record.get("goal_id") in (None, goal_id):
                records.append(record)
        except json.JSONDecodeError:
            continue
    
    # Return last N records by timestamp
    records.sort(key=lambda r: r["timestamp"], reverse=True)
    return records[:limit]
```

### 5.3 Memory Cleanup

A separate `tools/memory_janitor.py` process (optional) cleans up temp records:

```python
def cleanup_temp_records(max_age_hours=24):
    """Remove temp records older than max_age_hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    new_lines = []
    
    for line in read_text("working_memory.jsonl").splitlines():
        record = json.loads(line)
        if record.get("type") == "temp":
            ts = datetime.fromisoformat(record["timestamp"])
            if ts > cutoff:
                new_lines.append(line)
        else:
            new_lines.append(line)  # Keep non-temp forever
    
    with open("working_memory.jsonl", "w") as f:
        f.write("\n".join(new_lines))
```

---

## 6. Safety Fences

### 6.1 Configuration Validation

`config.yaml` is validated on startup:

```python
def validate_config(config):
    """Return (valid, errors) tuple."""
    errors = []
    
    # Check required sections
    required_sections = ["tick", "goals", "safety", "tools"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing section: {section}")
    
    # Check safety classes are known
    valid_classes = {"autonomous", "needs_approval", "forbidden"}
    for op in config["safety"]["autonomous"] + config["safety"]["needs_approval"]:
        if op not in valid_classes:
            errors.append(f"Unknown safety class: {op}")
    
    # Check tick interval is reasonable (min 5 seconds)
    if config["tick"]["interval_seconds"] < 5:
        errors.append("tick.interval_seconds must be >= 5")
    
    # Check tool allowlist is not empty (if autonomous tools enabled)
    if any("shell" in op for op in config["safety"]["autonomous"]):
        if not config["tools"]["allowed_executables"]:
            errors.append("Cannot have autonomous shell ops with empty allowlist")
    
    return len(errors) == 0, errors
```

### 6.2 Operation Approval Flow

```
┌─────────────┐
│ Core v2     │  Wants to execute operation
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Safety Check    │  What safety class?
└──────┬──────────┘
       │
   ┌───┴────┬──────────┐
   ▼        ▼          ▼
┌──────┐ ┌──────┐ ┌──────────┐
│ auto │ │need  │ │forbidden │
│ mous │ │approv│ │          │
│  al  │ │  al  │ │  BLOCK   │
└──┬───┘ └───┬──┘ └──────────┘
   │         │
   ▼         ▼
Execute   Human
Now      Approve?
```

### 6.3 Forbidden Operations

These operations NEVER execute, regardless of approval:

```python
FORBIDDEN_OPERATIONS = {
    "write_to_config",           # config.yaml is human-only
    "write_to_audit_log",        # audit.jsonl is append-only by tool layer
    "modify_core_binaries",      # loop.py, core v2 binaries
    "system_configuration",      # /etc, system settings
    "privilege_escalation",      # sudo, pkexec, etc.
}

def check_forbidden(operation):
    """Return (allowed, reason)."""
    if operation in FORBIDDEN_OPERATIONS:
        return False, f"Operation {operation} is forbidden by policy"
    return True, "ok"
```

---

## 7. Configuration Surface

### 7.1 Complete config.yaml Template

```yaml
# ============================================
# agent-loop v2 Configuration
# ============================================

# === Tick Settings ===
tick:
  enabled: true
  interval_seconds: 30.0
  idle_only: false  # Only tick when human idle

# === Goal Queue Settings ===
goals:
  max_concurrent: 2
  auto_advance: true
  completion_threshold: 0.8

# === Safety Classification ===
safety:
  autonomous:
    - read_file
    - write_to_owned_file
    - append_to_stream
    - tick_status_update
    - goal_status_update
    - working_memory_append
    - tool_status_query
  
  needs_approval:
    - write_to_non_owned_file
    - shell_execute
    - network_request
    - file_delete
    - git_commit
    - goal_create
    - goal_delete
  
  forbidden:
    - write_to_config
    - write_to_audit_log
    - modify_core_binaries
    - system_configuration
    - privilege_escalation

# === Tool Allowlist ===
tools:
  allowed_executables:
    - ls
    - cat
    - grep
    - pytest
    - git
    - python3
  
  allowed_patterns:
    - "^git (status|diff|log|log --oneline)"
    - "^ls -"
    - "^cat .+\\.md$"
    - "^grep -r"
    - "^pytest (tests/|test_)"
    - "^python3 -m pytest"
  
  execution_timeout: 10
  network_timeout: 30

# === Memory Settings ===
memory:
  max_temp_age_hours: 24
  max_records_loaded: 10

# === Model Settings ===
model:
  name: "qwen2.5-coder:3b"
  server_url: "http://localhost:11434/api/chat"
  num_ctx: 8192

# === File Paths ===
files:
  workspace: "workspace.md"
  stream: "stream.md"
  goals: "goals.md"
  config: "config.yaml"
  working_memory: "working_memory.jsonl"
  tick_status: "tick.md"
```

### 7.2 Runtime Config Loading

```python
import yaml

def load_config(path="config.yaml"):
    """Load and validate config.yaml."""
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("! config.yaml not found, using defaults")
        config = DEFAULT_CONFIG
    except yaml.YAMLError as e:
        sys.exit(f"! config.yaml parse error: {e}")
    
    valid, errors = validate_config(config)
    if not valid:
        sys.exit(f"! config.yaml validation errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return config
```

---

## 8. Implementation Modules

### 8.1 Module Breakdown

| Module | File | Purpose | Lines (est.) |
|--------|------|---------|--------------|
| Core v2 | `loop_v2.py` | Extended main loop with tick, goals, memory | ~450 |
| Tool Layer | `tools/tool_layer.py` | Sandbox, execution, audit | ~280 |
| Tool Spec | `tools/spec/SPEC.md` | Tool interface contract | ~120 |
| Config Loader | `config.py` | YAML load, validation | ~80 |
| Memory Janitor | `tools/memory_janitor.py` | Cleanup temp records | ~60 |
| Approval UI | `extras/approval_ui.py` | Optional approval TUI | ~150 |
| Tests | `tests/test_*.py` | Comprehensive test suite | ~800 |

**Total new code:** ~1,940 lines (est.)  
**Total including core v1:** ~2,200 lines

### 8.2 Zero-Dependency Strategy

- **YAML:** Use `yaml.safe_load` from stdlib (Python 3.6+)
- **Subprocess:** Use `subprocess.run` (stdlib)
- **File watching:** Use `os.stat` mtime polling (already in core v1)
- **Time:** Use `time`, `datetime` (stdlib)
- **HTTP:** Use `urllib.request` (already in core v1)

**NO external deps:** No `pyyaml`, no `watchdog`, no `click`, no `rich`.

---

## 9. Migration Path

### 9.1 Backwards Compatibility

Core v1 (loop.py) continues to work EXACTLY as before. The v2 system is opt-in:

```bash
# Existing v1 usage (unchanged)
python3 loop.py

# New v2 usage
python3 loop_v2.py     # Core with autonomous tick
python3 tools/tool_layer.py  &  # Tool layer (optional)
python3 tools/memory_janitor.py  &  # Janitor (optional)
```

### 9.2 Progressive Rollout

1. **Phase 1:** Deploy `config.yaml` and `loop_v2.py` (tick + goals)
2. **Phase 2:** Deploy `tools/tool_layer.py` (sandbox)
3. **Phase 3:** Deploy `tools/memory_janitor.py` (maintenance)
4. **Phase 4:** Deploy `extras/approval_ui.py` (UX improvement)

Each phase is independently valuable and can be adopted without later phases.

---

## 10. Security Model

### 10.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| Model hallucinates dangerous command | Allowlist + pattern matching |
| Tool layer compromised | Audit log + human approval for dangerous ops |
| Config poisoned | Config is human-only, validated on load |
| Privilege escalation | Forbidden ops list + no sudo in allowlist |
| Resource exhaustion | Execution timeouts + tick interval floor |
| Memory injection | JSONL schema validation + size limits |

### 10.2 Audit Trail

All actions are logged to `tools/audit.jsonl`:

```json
{"id": "REQ-001", "type": "shell_execute", "command": "ls -la", "safety_class": "needs_approval", "status": "approved", "approver": "human", "timestamp": "2026-07-18T10:00:00Z"}
```

This enables:
- Post-mortem analysis
- Compliance reporting
- Pattern detection (e.g., frequent rejected commands)

---

## 11. Future Extensions (Out of Scope for v1)

These are noted for future consideration but NOT part of this design:

- **Multi-file goals:** Goals that span multiple files/projects
- **Goal dependencies:** Explicit blocking/unblocking relationships
- **Tool composition:** Chaining multiple tools in one request
- **Remote execution:** Tool layer on a different machine
- **Encryption:** Encrypting working memory for sensitive contexts
- **Multi-user:** Multiple humans collaborating with one agent

---

## 12. Verification

### 12.1 Test Strategy

Each module has comprehensive tests:

```bash
# Unit tests
pytest tests/test_config.py
pytest tests/test_sandbox.py
pytest tests/test_memory.py

# Integration tests
pytest tests/test_tick_flow.py
pytest tests/test_goal_lifecycle.py

# Safety tests
pytest tests/test_forbidden_ops.py
pytest tests/test_approval_flow.py
```

### 12.2 Manual Verification Checklist

- [ ] Autonomous tick fires at configured interval
- [ ] Goal queue advances correctly on completion
- [ ] Tool allowlist blocks disallowed commands
- [ ] Approval flow pauses for human input
- [ ] Audit log records all tool invocations
- [ ] Working memory persists across restarts
- [ ] Forbidden operations are always blocked
- [ ] Config validation rejects invalid configs

---

## Appendix A: File Dependencies

```
config.yaml (human-only)
    ↓
loop_v2.py (reads config)
    ↓ reads/writes
goals.md ↔ tick.md ↔ working_memory.jsonl
    ↓ writes to
tools/request.md
    ↓ reads
tools/tool_layer.py
    ↓ writes to
tools/response.md, tools/audit.jsonl
```

## Appendix B: Command Extensions (v2)

New commands in `workspace.md`:

| Command | Args | Effect |
|---------|------|--------|
| `!tick_enable` | none | Enable autonomous tick |
| `!tick_disable` | none | Disable autonomous tick |
| `!goal_create` | `<priority> <title>` | Create new goal |
| `!goal_pause` | `<id>` | Pause goal |
| `!goal_resume` | `<id>` | Resume paused goal |
| `!approve` | `<id>` | Approve pending tool request |
| `!reject` | `<id>` | Reject pending tool request |
| `!memory_recall` | `<pattern>` | Search working memory |

---

**End of EXTENSION_DESIGN.md**
