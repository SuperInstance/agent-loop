#!/usr/bin/env python3
"""
gc.py — garbage collector for the autonomous agent workspace.

Manages disk usage on constrained SSDs (256-512GB). Prunes old logs,
archives completed goals, and manages Ollama model footprint.

Zero dependencies. Python 3.9+. Stdlib only.

Usage:
  # Import and use programmatically
  import gc
  gc.auto_gc(config)

  # Run standalone
  python3 gc.py --report
  python3 gc.py --prune
  python3 gc.py --ollama-prune --keep-tags iterator,coder,thinker
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone, timedelta

# ═══════════════════════════════════════════════════════════════
# File paths (should match autonomous.py)
# ═══════════════════════════════════════════════════════════════

WORKSPACE_FILE  = "workspace.md"
STREAM_FILE     = "stream.md"
QUEST_FILE      = "quest.md"
MEMORY_FILE     = "preferences.jsonl"
GOALS_FILE      = "goals.md"
GOALS_ARCHIVE   = "goals_archive.md"
WORKING_MEM_FILE = "working_memory.jsonl"
AUDIT_FILE      = "tools/audit.jsonl"
CONFIG_FILE     = "config.yaml"

# Default thresholds (can be overridden by config from autonomous.py)
DEFAULT_CONFIG = {
    "enabled": True,
    "run_every_ticks": 120,
    "max_working_mem_per_goal": 100,
    "max_stream_lines": 500,
    "max_audit_age_days": 7,
    "max_model_disk_gb": 20.0,
    "keep_model_tags": ["iterator", "coder", "thinker"],
}

# ═══════════════════════════════════════════════════════════════
# Ollama API helpers
# ═══════════════════════════════════════════════════════════════

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"


def _get_json(url, timeout=5):
    """Fetch JSON from URL with error handling."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return None


def ollama_disk_usage():
    """
    Query Ollama API for model sizes on disk.

    Returns:
        dict with {total_gb, models: [{name, size_gb, digest}]} or None on error.
    """
    data = _get_json(OLLAMA_TAGS_URL)
    if not data or "models" not in data:
        return None

    total = 0.0
    models = []
    for m in data["models"]:
        size_bytes = m.get("size", 0)
        size_gb = size_bytes / (1024**3)
        total += size_gb
        models.append({
            "name": m.get("name", "unknown"),
            "size_gb": size_gb,
            "digest": m.get("digest", "")[:16],
        })

    return {"total_gb": total, "models": models}


def prune_ollama_models(keep_tags=None, dry_run=False):
    """
    Remove Ollama models not matching keep tags.

    Args:
        keep_tags: List of model tags to preserve (e.g., ['iterator', 'coder'])
        dry_run: If True, report what would be removed without removing

    Returns:
        dict with {removed, kept, skipped, total_freed_gb}
    """
    if keep_tags is None:
        keep_tags = DEFAULT_CONFIG["keep_model_tags"]

    # Fetch available models
    data = _get_json(OLLAMA_TAGS_URL)
    if not data or "models" not in data:
        return {"error": "Could not fetch model list from Ollama"}

    keep_tags_set = set(t.lower() for t in keep_tags)

    removed = []
    kept = []
    skipped = []
    total_freed_gb = 0.0

    for m in data["models"]:
        name = m.get("name", "")
        size_bytes = m.get("size", 0)
        size_gb = size_bytes / (1024**3)

        # Check if model matches any keep tag
        # Match by model name (contains tag) or by specialty
        should_keep = False
        for tag in keep_tags_set:
            if tag in name.lower():
                should_keep = True
                break

        if should_keep:
            kept.append({"name": name, "size_gb": size_gb})
        else:
            if dry_run:
                removed.append({"name": name, "size_gb": size_gb})
                total_freed_gb += size_gb
            else:
                # Execute ollama rm
                try:
                    result = subprocess.run(
                        ["ollama", "rm", name],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        removed.append({"name": name, "size_gb": size_gb})
                        total_freed_gb += size_gb
                    else:
                        skipped.append({
                            "name": name,
                            "size_gb": size_gb,
                            "reason": result.stderr or "remove failed"
                        })
                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    skipped.append({
                        "name": name,
                        "size_gb": size_gb,
                        "reason": str(e)
                    })

    return {
        "removed": removed,
        "kept": kept,
        "skipped": skipped,
        "total_freed_gb": total_freed_gb,
    }


# ═══════════════════════════════════════════════════════════════
# File operations
# ═══════════════════════════════════════════════════════════════

def _read_lines(path):
    """Read file lines safely, returning empty list on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except (FileNotFoundError, IOError):
        return []


def _write_lines(path, lines):
    """Write lines to file atomically."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        temp_path = f"{path}.tmp.{os.getpid()}"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(temp_path, path)
        return True
    except IOError:
        return False


def _get_file_age_days(path):
    """Get file age in days, or None if file doesn't exist."""
    try:
        mtime = os.path.getmtime(path)
        age = (time.time() - mtime) / 86400
        return age
    except (OSError, AttributeError):
        return None


import time


def prune_working_memory(max_per_goal=100, max_age_days=30):
    """
    Remove old records from working_memory.jsonl.

    Keeps:
    - At most max_per_goal records per goal_id
    - Records younger than max_age_days

    Args:
        max_per_goal: Max records to keep per goal ID
        max_age_days: Max age of records to keep

    Returns:
        dict with {total_before, total_after, removed_by_goal, removed_by_age}
    """
    path = WORKING_MEM_FILE
    lines = _read_lines(path)

    if not lines:
        return {"total_before": 0, "total_after": 0, "removed": 0}

    now = time.time()
    cutoff_time = now - (max_age_days * 86400)

    # Group by goal_id
    by_goal = {}
    removed_age = 0
    removed_goal = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            goal_id = record.get("goal_id", "none")
            timestamp = record.get("timestamp", "")

            # Parse timestamp
            try:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                record_time = ts.timestamp()
            except (ValueError, AttributeError):
                record_time = now  # Unknown timestamp, treat as recent

            if record_time < cutoff_time:
                removed_age += 1
                continue

            if goal_id not in by_goal:
                by_goal[goal_id] = []
            by_goal[goal_id].append((record_time, line))
        except (json.JSONDecodeError, ValueError):
            continue

    # Prune to max_per_goal per goal (keep newest)
    kept_lines = []
    for goal_id, records in by_goal.items():
        # Sort by timestamp descending (newest first)
        records.sort(key=lambda x: x[0], reverse=True)
        kept = records[:max_per_goal]
        removed_goal += len(records) - len(kept)
        # Extract original lines (sort back by timestamp for consistency)
        kept.sort(key=lambda x: x[0])
        for _, line in kept:
            kept_lines.append(line + "\n")

    if _write_lines(path, kept_lines):
        return {
            "total_before": len(lines),
            "total_after": len(kept_lines),
            "removed_by_goal": removed_goal,
            "removed_by_age": removed_age,
            "total_removed": removed_goal + removed_age,
        }
    return {"error": "Failed to write pruned working memory"}


def prune_stream(max_lines=500):
    """
    Keep only the last N lines in stream.md.

    Preserves the header line starting with '#'.

    Args:
        max_lines: Maximum number of content lines to keep

    Returns:
        dict with {lines_before, lines_after, removed}
    """
    path = STREAM_FILE
    lines = _read_lines(path)

    if not lines:
        return {"lines_before": 0, "lines_after": 0, "removed": 0}

    # Preserve header
    header = []
    content = []
    for line in lines:
        if line.strip().startswith("#"):
            header.append(line)
        else:
            content.append(line)

    # Keep last max_lines of content
    kept_content = content[-max_lines:] if len(content) > max_lines else content

    if _write_lines(path, header + kept_content):
        return {
            "lines_before": len(lines),
            "lines_after": len(header) + len(kept_content),
            "removed": len(lines) - len(header) - len(kept_content),
        }
    return {"error": "Failed to write pruned stream"}


def prune_audit_log(max_age_days=7):
    """
    Remove old audit entries from tools/audit.jsonl.

    Args:
        max_age_days: Maximum age of audit entries to keep

    Returns:
        dict with {entries_before, entries_after, removed}
    """
    path = AUDIT_FILE
    lines = _read_lines(path)

    if not lines:
        return {"entries_before": 0, "entries_after": 0, "removed": 0}

    now = time.time()
    cutoff_time = now - (max_age_days * 86400)
    kept_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            timestamp = entry.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                entry_time = ts.timestamp()
            except (ValueError, AttributeError):
                entry_time = now

            if entry_time >= cutoff_time:
                kept_lines.append(line + "\n")
        except (json.JSONDecodeError, ValueError):
            # Keep malformed lines
            kept_lines.append(line + "\n")

    if _write_lines(path, kept_lines):
        return {
            "entries_before": len(lines),
            "entries_after": len(kept_lines),
            "removed": len(lines) - len(kept_lines),
        }
    return {"error": "Failed to write pruned audit log"}


def archive_completed_goals():
    """
    Move completed goals from goals.md to goals_archive.md.

    Returns:
        dict with {archived_count, remaining_active}
    """
    goals_path = GOALS_FILE
    archive_path = GOALS_ARCHIVE

    text = "".join(_read_lines(goals_path))
    if not text:
        return {"archived_count": 0, "remaining_active": 0}

    # Parse goals into blocks
    goal_blocks = []
    current_block = []
    current_id = None
    current_status = "pending"

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## [") and "]" in stripped:
            # Save previous block
            if current_block:
                goal_blocks.append({
                    "id": current_id,
                    "status": current_status,
                    "lines": current_block,
                })
            # Start new block
            try:
                bracket_end = stripped.index("]")
                current_id = stripped[4:bracket_end].strip()
            except ValueError:
                current_id = "unknown"
            current_status = "pending"
            current_block = [line + "\n"]
        elif current_block:
            current_block.append(line + "\n")
            if stripped.startswith("### status:"):
                status_val = stripped.split(":", 1)[1].strip().lower()
                if status_val in ("complete", "completed", "failed"):
                    current_status = "complete"

    # Save last block
    if current_block:
        goal_blocks.append({
            "id": current_id,
            "status": current_status,
            "lines": current_block,
        })

    # Separate active and completed
    active_lines = []
    archive_lines = []

    for block in goal_blocks:
        if block["status"] == "complete":
            archive_lines.extend(block["lines"])
            archive_lines.append("\n")
        else:
            active_lines.extend(block["lines"])

    # Write active goals back
    _write_lines(goals_path, active_lines)

    # Append completed to archive
    existing_archive = _read_lines(archive_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    archive_header = f"# Archived Goals — {timestamp}\n\n"
    _write_lines(archive_path, existing_archive + [archive_header] + archive_lines)

    return {
        "archived_count": len([b for b in goal_blocks if b["status"] == "complete"]),
        "remaining_active": len([b for b in goal_blocks if b["status"] != "complete"]),
    }


def clean_cache():
    """
    Remove __pycache__, .pyc files, and common temp directories.

    Returns:
        dict with {files_removed, dirs_removed, space_freed_bytes}
    """
    files_removed = 0
    dirs_removed = 0
    space_freed = 0

    # Patterns to clean
    cache_dirs = ["__pycache__", ".pytest_cache", ".mypy_cache", ".hypothesis"]
    temp_patterns = ["*.pyc", "*.pyo", "*.tmp", "*.tmp.*"]

    # Walk current directory
    for root, dirs, files in os.walk(".", topdown=True):
        # Remove cache directories
        for d in list(dirs):
            if d in cache_dirs or d.startswith(".tmp"):
                full_path = os.path.join(root, d)
                try:
                    # Calculate size before removal
                    dir_size = sum(
                        os.path.getsize(os.path.join(dirpath, f))
                        for dirpath, _, filenames in os.walk(full_path)
                        for f in filenames
                        if os.path.isfile(os.path.join(dirpath, f))
                    )
                    shutil.rmtree(full_path)
                    dirs_removed += 1
                    space_freed += dir_size
                    dirs.remove(d)  # Don't walk into removed dir
                except (OSError, shutil.Error):
                    pass

        # Remove temp files
        for f in files:
            if f.endswith((".pyc", ".pyo", ".tmp")) or f.startswith(".tmp."):
                full_path = os.path.join(root, f)
                try:
                    size = os.path.getsize(full_path)
                    os.remove(full_path)
                    files_removed += 1
                    space_freed += size
                except OSError:
                    pass

    return {
        "files_removed": files_removed,
        "dirs_removed": dirs_removed,
        "space_freed_bytes": space_freed,
        "space_freed_mb": space_freed / (1024 * 1024),
    }


# ═══════════════════════════════════════════════════════════════
# Disk usage reporting
# ═══════════════════════════════════════════════════════════════

def disk_usage_report():
    """
    Generate a comprehensive disk usage report.

    Returns:
        dict with {models_gb, workspace_files_gb, logs_gb, cache_gb, total_gb}
    """
    report = {
        "models": ollama_disk_usage(),
        "workspace_files": {},
        "logs": {},
        "cache": clean_cache() if "--clean" in sys.argv else {"space_freed_mb": 0},
    }

    # Calculate workspace file sizes
    workspace_files = [
        WORKSPACE_FILE, STREAM_FILE, QUEST_FILE, MEMORY_FILE,
        GOALS_FILE, GOALS_ARCHIVE, WORKING_MEM_FILE, CONFIG_FILE,
        "tick.md", "approval.md",
    ]

    total_workspace = 0.0
    for f in workspace_files:
        if os.path.exists(f):
            size = os.path.getsize(f) / (1024**2)  # MB
            total_workspace += size
            report["workspace_files"][f] = size

    # Log file sizes
    log_files = [AUDIT_FILE, "tools/*.jsonl", "*.log"]
    total_logs = 0.0
    if os.path.exists(AUDIT_FILE):
        size = os.path.getsize(AUDIT_FILE) / (1024**2)
        total_logs += size
        report["logs"][AUDIT_FILE] = size

    # Ollama models
    models_gb = 0.0
    if report["models"]:
        models_gb = report["models"]["total_gb"]

    return {
        "models_gb": round(models_gb, 2),
        "workspace_files_mb": round(total_workspace, 2),
        "logs_mb": round(total_logs, 2),
        "cache_freed_mb": round(report["cache"]["space_freed_mb"], 2),
        "details": report,
    }


# ═══════════════════════════════════════════════════════════════
# Auto GC (main entry point for autonomous.py)
# ═══════════════════════════════════════════════════════════════

def auto_gc(config=None, dry_run=False):
    """
    Run all garbage collection based on config thresholds.

    Args:
        config: Dict of gc config (uses DEFAULT_CONFIG if None)
        dry_run: If True, report what would be done without doing it

    Returns:
        dict with {operation: result} for each GC operation
    """
    if config is None:
        config = DEFAULT_CONFIG

    if not config.get("enabled", True):
        return {"status": "disabled"}

    results = {}

    # Working memory pruning
    max_per_goal = config.get("max_working_mem_per_goal", 100)
    if dry_run:
        results["working_memory"] = {"dry_run": f"Would keep max {max_per_goal} per goal"}
    else:
        results["working_memory"] = prune_working_memory(max_per_goal=max_per_goal)

    # Stream pruning
    max_stream = config.get("max_stream_lines", 500)
    if dry_run:
        results["stream"] = {"dry_run": f"Would keep last {max_stream} lines"}
    else:
        results["stream"] = prune_stream(max_lines=max_stream)

    # Audit log pruning
    max_audit_age = config.get("max_audit_age_days", 7)
    if dry_run:
        results["audit_log"] = {"dry_run": f"Would prune entries older than {max_audit_age} days"}
    else:
        results["audit_log"] = prune_audit_log(max_age_days=max_audit_age)

    # Goal archiving
    if dry_run:
        results["goals_archive"] = {"dry_run": "Would archive completed goals"}
    else:
        results["goals_archive"] = archive_completed_goals()

    # Cache cleaning
    if dry_run:
        results["cache"] = {"dry_run": "Would clean __pycache__ and temp files"}
    else:
        results["cache"] = clean_cache()

    # Ollama model disk check (warn only, don't auto-remove)
    max_model_gb = config.get("max_model_disk_gb", 20.0)
    ollama_usage = ollama_disk_usage()
    if ollama_usage:
        model_gb = ollama_usage["total_gb"]
        if model_gb > max_model_gb:
            results["models_warning"] = {
                "current_gb": round(model_gb, 2),
                "max_gb": max_model_gb,
                "excess_gb": round(model_gb - max_model_gb, 2),
                "message": f"Ollama models ({model_gb:.1f}GB) exceed threshold ({max_model_gb}GB)",
            }

    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Garbage collector for autonomous agent workspace"
    )
    parser.add_argument("--report", action="store_true",
                       help="Show disk usage report")
    parser.add_argument("--prune", action="store_true",
                       help="Run all pruning operations")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be pruned without pruning")
    parser.add_argument("--ollama-prune", action="store_true",
                       help="Remove unused Ollama models")
    parser.add_argument("--keep-tags", default="iterator,coder,thinker",
                       help="Comma-separated model tags to keep (default: iterator,coder,thinker)")
    parser.add_argument("--models", action="store_true",
                       help="Show Ollama model disk usage")
    parser.add_argument("--clean-cache", action="store_true",
                       help="Remove __pycache__, .pyc, and temp files")
    args = parser.parse_args()

    if args.report:
        report = disk_usage_report()
        print("=== Disk Usage Report ===")
        print(f"Ollama models: {report['models_gb']} GB")
        print(f"Workspace files: {report['workspace_files_mb']} MB")
        print(f"Logs: {report['logs_mb']} MB")
        print(f"Cache cleaned: {report['cache_freed_mb']} MB")
        return

    if args.models:
        usage = ollama_disk_usage()
        if usage:
            print(f"=== Ollama Models: {usage['total_gb']:.2f} GB ===")
            for m in usage["models"]:
                print(f"  {m['name']}: {m['size_gb']:.2f} GB")
        else:
            print("Could not fetch Ollama model list")
        return

    if args.ollama_prune:
        keep_tags = [t.strip() for t in args.keep_tags.split(",")]
        result = prune_ollama_models(keep_tags=keep_tags, dry_run=args.dry_run)
        if args.dry_run:
            print("=== Dry Run: Would remove these models ===")
        else:
            print("=== Pruned Ollama Models ===")
        print(f"Kept: {len(result['kept'])} models")
        print(f"Removed: {len(result['removed'])} models")
        print(f"Freed: {result['total_freed_gb']:.2f} GB")
        if result["removed"]:
            print("\nRemoved:")
            for m in result["removed"]:
                print(f"  - {m['name']} ({m['size_gb']:.2f} GB)")
        if result["skipped"]:
            print("\nSkipped:")
            for m in result["skipped"]:
                print(f"  - {m['name']} ({m['size_gb']:.2f} GB): {m['reason']}")
        return

    if args.clean_cache:
        result = clean_cache()
        print("=== Cache Cleaned ===")
        print(f"Files removed: {result['files_removed']}")
        print(f"Dirs removed: {result['dirs_removed']}")
        print(f"Space freed: {result['space_freed_mb']:.2f} MB")
        return

    if args.prune or args.dry_run:
        results = auto_gc(dry_run=args.dry_run)
        print("=== Garbage Collection Results ===")
        for op, result in results.items():
            print(f"\n{op}:")
            if isinstance(result, dict):
                for k, v in result.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {result}")
        return

    # Default: show brief report
    print("Use --help for options. Try: python3 gc.py --report")


if __name__ == "__main__":
    main()
