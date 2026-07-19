"""Unit tests for autonomous.py.

Covers:
  1. Config loading (load_config)
  2. Goal parsing (parse_goals)
  3. Goal selection (select_active_goal)
  4. Goal status updates (update_goal_status)
  5. Working memory (append_working_memory / load_working_memory)
  6. Style rule loading (load_rules)
  7. Action parsing (parse_action)
  8. Action execution (execute_action — every action type)
  9. Command handling (handle_command — every command)
 10. Tick status writer (update_tick_status)
 11. File initialization (ensure_files)
 12. Already-done detection (_already_done)

All tests run with a temporary directory fixture so no real files are touched.
"""

import collections
import io
import json
import os
import re
import subprocess
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import autonomous


# ════════════════════════════════════════════════════════════════
# 1. load_config — default / empty / custom / missing / malformed
# ════════════════════════════════════════════════════════════════


class TestLoadConfig:
    """config loading — parser is intentionally minimal."""

    def test_load_config_default_when_no_file(self, in_tmp_path):
        """With no config.yaml present, returns a deep copy of DEFAULT_CONFIG."""
        assert not (in_tmp_path / "config.yaml").exists()
        config = autonomous.load_config()
        # Top-level sections match DEFAULT_CONFIG
        assert set(config.keys()) == set(autonomous.DEFAULT_CONFIG.keys())
        # Specific values
        assert config["tick"]["enabled"] is True
        assert config["goals"]["max_concurrent"] == 1
        assert config["goals"]["auto_advance"] is True
        assert "python3" in config["tools"]["allowed_executables"]
        # Default safety fences
        assert "read_file" in config["safety"]["autonomous"]
        assert "shell_execute" in config["safety"]["needs_approval"]

    def test_load_config_returns_deep_copy(self, in_tmp_path):
        """Mutating the result must not affect the module-level DEFAULT_CONFIG."""
        config = autonomous.load_config()
        config["goals"]["max_concurrent"] = 999
        config["safety"]["autonomous"].append("rogue_action")
        # Originals unchanged
        assert autonomous.DEFAULT_CONFIG["goals"]["max_concurrent"] == 1
        assert "rogue_action" not in autonomous.DEFAULT_CONFIG["safety"]["autonomous"]

    def test_load_config_empty_file_returns_default(self, in_tmp_path):
        """Empty file behaves like no file (parser sees no content)."""
        (in_tmp_path / "config.yaml").write_text("")
        config = autonomous.load_config()
        assert config["tick"]["interval_seconds"] == autonomous.AUTONOMOUS_INTERVAL
        assert config["goals"]["max_concurrent"] == 1

    def test_load_config_missing_file_returns_default(self, in_tmp_path):
        """Removing config.yaml after creation falls back to defaults."""
        # Set up files via ensure_files, then remove config.yaml
        autonomous.ensure_files()
        (in_tmp_path / "config.yaml").unlink()
        config = autonomous.load_config()
        assert config["goals"]["max_concurrent"] == 1
        assert config["tick"]["enabled"] is True

    def test_load_config_custom_subsection_list_appends(self, in_tmp_path):
        """Custom YAML with section/subsection/items — items are appended to defaults."""
        (in_tmp_path / "config.yaml").write_text(
            "safety:\n"
            "  needs_approval:\n"
            "    - my_custom_tool\n"
            "    - another_tool\n"
        )
        config = autonomous.load_config()
        # Default items preserved
        assert "shell_execute" in config["safety"]["needs_approval"]
        # Custom items appended
        assert "my_custom_tool" in config["safety"]["needs_approval"]
        assert "another_tool" in config["safety"]["needs_approval"]

    def test_load_config_new_section(self, in_tmp_path):
        """A section not in the defaults is added."""
        (in_tmp_path / "config.yaml").write_text(
            "experimental:\n"
            "  feature_one:\n"
            "    - flag_a\n"
            "    - flag_b\n"
        )
        config = autonomous.load_config()
        assert "experimental" in config
        assert config["experimental"]["feature_one"] == ["flag_a", "flag_b"]
        # Defaults still accessible
        assert config["goals"]["max_concurrent"] == 1

    def test_load_config_with_comments_ignored(self, in_tmp_path):
        """Comment lines (#) are skipped during parsing."""
        (in_tmp_path / "config.yaml").write_text(
            "# Top-level comment\n"
            "tick:\n"
            "  # Subsection comment\n"
            "  enabled: true\n"
        )
        # This won't crash; config is parsed to the extent supported.
        config = autonomous.load_config()
        # Defaults still present
        assert config["goals"]["max_concurrent"] == 1

    def test_load_config_malformed_keeps_defaults(self, in_tmp_path):
        """Garbage content does not crash and defaults remain accessible."""
        (in_tmp_path / "config.yaml").write_text(
            "this is :: not : valid : yaml :: @@@\n"
            "!!! garbage ###\n"
            "\x00\x01\x02\n"  # binary noise
        )
        config = autonomous.load_config()
        # Should not raise; defaults still there
        assert config["tick"]["enabled"] is True
        assert "python3" in config["tools"]["allowed_executables"]

    def test_load_config_unicode_and_blank_lines(self, in_tmp_path):
        """Mixed unicode / blank lines parse without errors."""
        (in_tmp_path / "config.yaml").write_text(
            "\n"
            "\n"
            "tick:\n"
            "  enabled: true\n"
            "\n"
            "  # comment line\n"
        )
        config = autonomous.load_config()
        assert config["goals"]["max_concurrent"] == 1


# ════════════════════════════════════════════════════════════════
# 2. parse_goals — empty / single / multiple / malformed
# ════════════════════════════════════════════════════════════════


class TestParseGoals:
    """Goal parser is permissive — only well-formed ## [Gxxx] headers count."""

    def test_parse_goals_empty_returns_empty_list(self, in_workspace):
        (in_workspace / "goals.md").write_text("")
        assert autonomous.parse_goals() == []

    def test_parse_goals_only_header_lines(self, in_workspace):
        """Top-level docstring / comments / blank lines are skipped."""
        (in_workspace / "goals.md").write_text(
            "# goals.md — nothing yet\n"
            "\n"
            "Add goals here:\n"
            "\n"
        )
        goals = autonomous.parse_goals()
        assert goals == []

    def test_parse_goals_single(self, in_workspace):
        (in_workspace / "goals.md").write_text(
            "## [G001] Write the tests\n"
            "### context\n"
            "Write a comprehensive test suite for the agent loop.\n"
            "### status: pending\n"
            "### priority: 1\n"
        )
        goals = autonomous.parse_goals()
        assert len(goals) == 1
        g = goals[0]
        assert g["id"] == "G001"
        assert g["title"] == "Write the tests"
        assert g["status"] == "pending"
        assert g["priority"] == 1
        assert "comprehensive test suite" in g["context"]

    def test_parse_goals_multiple(self, in_workspace):
        (in_workspace / "goals.md").write_text(
            "## [G001] First goal\n"
            "### status: pending\n"
            "### priority: 1\n"
            "\n"
            "## [G002] Second goal\n"
            "### status: in_progress\n"
            "### priority: 3\n"
            "\n"
            "## [G003] Third goal\n"
            "### status: complete\n"
            "### priority: 2\n"
        )
        goals = autonomous.parse_goals()
        assert len(goals) == 3
        assert [g["id"] for g in goals] == ["G001", "G002", "G003"]
        assert [g["title"] for g in goals] == [
            "First goal", "Second goal", "Third goal"
        ]
        assert [g["priority"] for g in goals] == [1, 3, 2]
        assert [g["status"] for g in goals] == [
            "pending", "in_progress", "complete"
        ]

    def test_parse_goals_malformed_missing_bracket_skipped(self, in_workspace):
        """Header without closing ] is not parsed as a goal."""
        (in_workspace / "goals.md").write_text(
            "## [G001 broken\n"
            "some content\n"
            "## [G002] Valid goal\n"
            "### status: pending\n"
            "### priority: 5\n"
        )
        goals = autonomous.parse_goals()
        assert len(goals) == 1
        assert goals[0]["id"] == "G002"

    def test_parse_goals_priority_defaults_high_when_missing(self, in_workspace):
        """When priority isn't specified, the default of 99 is used."""
        (in_workspace / "goals.md").write_text(
            "## [G001] No priority set\n"
            "### status: pending\n"
        )
        goals = autonomous.parse_goals()
        assert goals[0]["priority"] == 99

    def test_parse_goals_invalid_priority_keeps_default(self, in_workspace):
        """Non-numeric priority falls back to the default of 99."""
        (in_workspace / "goals.md").write_text(
            "## [G001] Bad priority\n"
            "### status: pending\n"
            "### priority: NaN\n"
        )
        goals = autonomous.parse_goals()
        # int("NaN") raises ValueError → excepted → priority untouched (default 99)
        assert goals[0]["priority"] == 99

    def test_parse_goals_status_in_progress(self, in_workspace):
        (in_workspace / "goals.md").write_text(
            "## [G001] Doing it now\n"
            "### status: in_progress\n"
            "### priority: 1\n"
        )
        goals = autonomous.parse_goals()
        assert goals[0]["status"] == "in_progress"


# ════════════════════════════════════════════════════════════════
# 3. select_active_goal — priority, max_concurrent, tiebreaks
# ════════════════════════════════════════════════════════════════


class TestSelectActiveGoal:
    """Active goal = lowest-priority-numbered pending/in_progress goal, capped by max_concurrent."""

    def _config(self, max_concurrent=1):
        cfg = json.loads(json.dumps(autonomous.DEFAULT_CONFIG))
        cfg["goals"]["max_concurrent"] = max_concurrent
        return cfg

    def _g(self, gid, status, priority, title="T"):
        return {
            "id": gid,
            "title": title,
            "context": "",
            "status": status,
            "priority": priority,
        }

    def test_select_empty_returns_none(self):
        assert autonomous.select_active_goal([], self._config()) is None

    def test_select_all_complete_returns_none(self):
        goals = [
            self._g("G001", "complete", 1),
            self._g("G002", "failed", 1),
        ]
        assert autonomous.select_active_goal(goals, self._config()) is None

    def test_select_priority_ordering(self):
        """Lowest priority number wins."""
        goals = [
            self._g("G001", "pending", 5, "A"),
            self._g("G002", "pending", 1, "B"),
            self._g("G003", "pending", 3, "C"),
        ]
        selected = autonomous.select_active_goal(goals, self._config())
        assert selected["id"] == "G002"

    def test_select_max_concurrent_1_keeps_in_progress(self):
        """With max_concurrent=1 and one in_progress, that one wins."""
        goals = [
            self._g("G001", "in_progress", 5, "running"),
            self._g("G002", "pending", 1, "queued-but-priority-higher"),
        ]
        # Wait — priorities: in_progress has 5, pending has 1. With 1 in_progress
        # at max_concurrent, candidates narrows to in_progress only → G001 wins.
        selected = autonomous.select_active_goal(goals, self._config(1))
        assert selected["id"] == "G001"

    def test_select_max_concurrent_allows_multiple(self):
        """With max_concurrent=2 and only 1 in_progress, pending joins the pool."""
        goals = [
            self._g("G001", "in_progress", 5),
            self._g("G002", "pending", 1),
        ]
        # 1 in_progress < 2 → all candidates stay; lowest priority wins
        selected = autonomous.select_active_goal(goals, self._config(2))
        assert selected["id"] == "G002"

    def test_select_tiebreak_by_id(self):
        """Same priority → alphabetic id wins."""
        goals = [
            self._g("G010", "pending", 1, "B"),
            self._g("G001", "pending", 1, "A"),
            self._g("G005", "pending", 1, "C"),
        ]
        selected = autonomous.select_active_goal(goals, self._config())
        assert selected["id"] == "G001"

    def test_select_skips_complete_and_failed(self):
        goals = [
            self._g("G001", "complete", 1, "DONE"),
            self._g("G002", "failed", 1, "DONE"),
            self._g("G003", "pending", 5, "STILL HERE"),
        ]
        selected = autonomous.select_active_goal(goals, self._config())
        assert selected["id"] == "G003"


# ════════════════════════════════════════════════════════════════
# 4. update_goal_status
# ════════════════════════════════════════════════════════════════


class TestUpdateGoalStatus:
    """Updates the ### status: line for the matching goal_id."""

    def test_update_status_simple_no_summary(self, in_workspace):
        (in_workspace / "goals.md").write_text(
            "## [G001] Goal one\n"
            "### status: pending\n"
            "### priority: 1\n"
            "\n"
            "## [G002] Goal two\n"
            "### status: pending\n"
            "### priority: 2\n"
        )
        autonomous.update_goal_status("G001", "in_progress")
        text = (in_workspace / "goals.md").read_text()
        # G001 updated
        assert "## [G001] Goal one\n### status: in_progress\n" in text
        # G002 untouched
        assert "## [G002] Goal two\n### status: pending\n" in text

    def test_update_status_with_summary(self, in_workspace):
        (in_workspace / "goals.md").write_text(
            "## [G001] Goal one\n"
            "### status: pending\n"
            "### priority: 1\n"
        )
        autonomous.update_goal_status("G001", "complete", "All tests pass")
        text = (in_workspace / "goals.md").read_text()
        # Status line updated
        assert "### status: complete" in text
        # Summary appended with timestamp
        assert "All tests pass" in text
        # Summary line includes "—" separator and an ISO-like timestamp pattern
        lines = text.splitlines()
        summary_line = next(l for l in lines if "All tests pass" in l)
        # Should match pattern like: - 2026-... — All tests pass
        assert re.match(r"^- \d{4}-\d{2}-\d{2}T.+ — All tests pass$", summary_line)

    def test_update_status_unknown_id_does_not_modify(self, in_workspace):
        original = (
            "## [G001] Goal one\n"
            "### status: pending\n"
            "### priority: 1\n"
        )
        (in_workspace / "goals.md").write_text(original)
        autonomous.update_goal_status("G999", "complete")
        text = (in_workspace / "goals.md").read_text()
        # No status line changed
        assert "### status: pending" in text
        assert "complete" not in text.split("### status")[1].split("\n", 1)[0]

    def test_update_status_preserves_other_lines(self, in_workspace):
        (in_workspace / "goals.md").write_text(
            "## [G001] Goal one\n"
            "### context\n"
            "Important context that must survive.\n"
            "### status: pending\n"
            "### priority: 1\n"
        )
        autonomous.update_goal_status("G001", "in_progress")
        text = (in_workspace / "goals.md").read_text()
        assert "Important context that must survive." in text
        assert "### priority: 1" in text


# ════════════════════════════════════════════════════════════════
# 5. Working memory — append, load, filter, limit
# ════════════════════════════════════════════════════════════════


class TestWorkingMemory:
    """append_working_memory / load_working_memory — JSONL persistent store."""

    def test_append_creates_jsonl_with_required_fields(self, in_workspace):
        autonomous.append_working_memory("observation", "File opened", "G001")
        autonomous.append_working_memory("decision", "Use pytest")

        text = (in_workspace / "working_memory.jsonl").read_text()
        lines = [l for l in text.splitlines() if l.strip()]
        assert len(lines) == 2

        records = [json.loads(l) for l in lines]
        assert records[0]["type"] == "observation"
        assert records[0]["goal_id"] == "G001"
        assert records[0]["content"] == "File opened"
        assert "timestamp" in records[0]
        # Second record has no goal_id
        assert records[1]["type"] == "decision"
        assert records[1]["goal_id"] is None
        assert records[1]["content"] == "Use pytest"

    def test_append_timestamp_is_iso(self, in_workspace):
        autonomous.append_working_memory("context", "test", "G001")
        text = (in_workspace / "working_memory.jsonl").read_text()
        record = json.loads(text.strip().splitlines()[-1])
        # Round-trip the timestamp through datetime to verify ISO format
        datetime.fromisoformat(record["timestamp"])

    def test_load_filter_by_goal_id(self, in_workspace):
        autonomous.append_working_memory("observation", "G001 step 1", "G001")
        autonomous.append_working_memory("observation", "G002 step 1", "G002")
        autonomous.append_working_memory("observation", "G001 step 2", "G001")

        g1 = autonomous.load_working_memory(goal_id="G001")
        assert len(g1) == 2
        for r in g1:
            assert r["goal_id"] == "G001"

        g2 = autonomous.load_working_memory(goal_id="G002")
        assert len(g2) == 1
        assert g2[0]["goal_id"] == "G002"

    def test_load_untyped_records_included_with_goal_query(self, in_workspace):
        """Records with goal_id=None are also included when querying a specific goal.

        Per the implementation: `if rid is None or goal_id is None or rid == goal_id`
        — records without a goal_id (e.g. global context) are always returned.
        """
        autonomous.append_working_memory("observation", "G001", "G001")
        autonomous.append_working_memory("context", "global note", None)
        records = autonomous.load_working_memory(goal_id="G001")
        # Both records come back (the G001 and the untyped global)
        assert len(records) == 2
        goal_ids = {r.get("goal_id") for r in records}
        assert goal_ids == {"G001", None}

    def test_load_with_no_goal_id_returns_all_with_untyped(self, in_workspace):
        autonomous.append_working_memory("observation", "G001 note", "G001")
        autonomous.append_working_memory("context", "Global note", None)

        records = autonomous.load_working_memory()
        # All records (goal_id and untyped) come back
        assert len(records) == 2
        goal_ids = {r.get("goal_id") for r in records}
        assert goal_ids == {"G001", None}


    def test_load_respects_limit(self, in_workspace):
        for i in range(15):
            autonomous.append_working_memory("observation", f"step {i}", "G001")

        records = autonomous.load_working_memory(goal_id="G001", limit=5)
        assert len(records) == 5

    def test_load_default_limit(self, in_workspace):
        """Default limit matches WORKING_MEM_TAIL."""
        for i in range(autonomous.WORKING_MEM_TAIL + 5):
            autonomous.append_working_memory("observation", f"step {i}", "G001")

        records = autonomous.load_working_memory(goal_id="G001")
        assert len(records) == autonomous.WORKING_MEM_TAIL

    def test_load_returns_newest_first(self, in_workspace):
        import time as _time
        autonomous.append_working_memory("observation", "First", "G001")
        _time.sleep(0.02)
        autonomous.append_working_memory("observation", "Second", "G001")
        _time.sleep(0.02)
        autonomous.append_working_memory("observation", "Third", "G001")

        records = autonomous.load_working_memory(goal_id="G001")
        assert records[0]["content"] == "Third"
        assert records[-1]["content"] == "First"

    def test_load_skips_malformed_lines(self, in_workspace):
        valid = json.dumps({"type": "obs", "content": "valid", "goal_id": "G1",
                            "timestamp": "2026-01-01T00:00:00+00:00"})
        also = json.dumps({"type": "obs", "content": "also valid", "goal_id": "G1",
                           "timestamp": "2026-01-02T00:00:00+00:00"})
        (in_workspace / "working_memory.jsonl").write_text(
            valid + "\n"
            + "not json at all\n"
            + also + "\n"
        )
        records = autonomous.load_working_memory(goal_id="G1")
        assert len(records) == 2

    def test_load_empty_file(self, in_workspace):
        # Fresh file (never written to) — read_text returns ""
        records = autonomous.load_working_memory()
        assert records == []


# ════════════════════════════════════════════════════════════════
# 6. load_rules — empty, populated, malformed
# ════════════════════════════════════════════════════════════════


class TestLoadRules:
    """Style rules loaded from preferences.jsonl, capped to MAX_FIX_RULES."""

    def test_load_rules_empty_file(self, in_workspace):
        # overwrite default
        (in_workspace / "preferences.jsonl").write_text("")
        assert autonomous.load_rules() == []

    def test_load_rules_missing_file(self, in_workspace):
        # File doesn't exist yet (defaults only created by ensure_files for some)
        if (in_workspace / "preferences.jsonl").exists():
            (in_workspace / "preferences.jsonl").unlink()
        assert autonomous.load_rules() == []

    def test_load_rules_populated(self, in_workspace):
        records = [
            {
                "instruction": "Apply style fix.",
                "input": "def foo():\n  return 1",
                "output": "def foo():\n    return 1",
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "instruction": "Apply style fix.",
                "input": "import os, sys",
                "output": "import os\nimport sys",
                "timestamp": "2026-01-02T00:00:00+00:00",
            },
        ]
        with open(in_workspace / "preferences.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        rules = autonomous.load_rules()
        assert len(rules) == 2
        assert rules[0]["rejected"] == "def foo():\n  return 1"
        assert rules[0]["accepted"] == "def foo():\n    return 1"
        assert rules[1]["rejected"] == "import os, sys"
        assert rules[1]["accepted"] == "import os\nimport sys"

    def test_load_rules_skips_malformed_json(self, in_workspace):
        valid = '{"instruction": "fix", "input": "a", "output": "b", "timestamp": "x"}'
        (in_workspace / "preferences.jsonl").write_text(
            valid + "\n"
            + "not valid json\n"
            + 'this is not even close\n'
        )
        rules = autonomous.load_rules()
        assert len(rules) == 1
        assert rules[0] == {"rejected": "a", "accepted": "b"}

    def test_load_rules_skips_missing_fields(self, in_workspace):
        """Records missing input/output are skipped (KeyError caught)."""
        (in_workspace / "preferences.jsonl").write_text(
            '{"instruction": "fix", "input": "valid", "output": "ok", "timestamp": "x"}\n'
            '{"instruction": "fix", "timestamp": "y"}\n'  # missing input/output
            '{"input": "x", "output": "y", "timestamp": "z"}\n'  # missing instruction is OK
        )
        rules = autonomous.load_rules()
        # 2 valid (input+output both present)
        assert len(rules) == 2
        rejected = {r["rejected"] for r in rules}
        assert rejected == {"valid", "x"}

    def test_load_rules_caps_at_max_fix(self, in_workspace):
        """Returns at most the last MAX_FIX_RULES entries."""
        with open(in_workspace / "preferences.jsonl", "w", encoding="utf-8") as f:
            for i in range(10):
                r = {
                    "instruction": "fix",
                    "input": f"x{i:03d}",
                    "output": f"y{i:03d}",
                    "timestamp": f"2026-01-{i+1:02d}T00:00:00+00:00",
                }
                f.write(json.dumps(r) + "\n")

        rules = autonomous.load_rules()
        assert len(rules) == autonomous.MAX_FIX_RULES
        # The last MAX_FIX_RULES should be the highest-numbered entries
        rejected = [r["rejected"] for r in rules]
        assert rejected[-1] == "x009"


# ════════════════════════════════════════════════════════════════
# 7. parse_action — JSON, fallback, wait
# ════════════════════════════════════════════════════════════════


class TestParseAction:
    """parse_action tries a JSON regex first, then falls back to first non-empty line."""

    def test_parse_action_valid_json_pure(self):
        action = autonomous.parse_action(
            '{"action": "suggest", "text": "do this"}'
        )
        assert action == {"action": "suggest", "text": "do this"}

    def test_parse_action_json_in_surrounding_text(self):
        text = (
            "Here is my response:\n"
            '{"action": "wait", "reason": "thinking"}\n'
            "The end."
        )
        action = autonomous.parse_action(text)
        assert action == {"action": "wait", "reason": "thinking"}

    def test_parse_action_complete_goal_json(self):
        action = autonomous.parse_action(
            '{"action": "complete_goal", "goal_id": "G001", "summary": "ok"}'
        )
        assert action == {
            "action": "complete_goal",
            "goal_id": "G001",
            "summary": "ok",
        }

    def test_parse_action_invalid_json_falls_back_to_suggest(self):
        text = "Just plain text with no JSON."
        action = autonomous.parse_action(text)
        assert action["action"] == "suggest"
        assert "Just plain text" in action["text"]

    def test_parse_action_broken_json_then_text(self):
        """Broken JSON falls back to first non-empty line."""
        text = (
            'preamble: "{not real json"\n'
            "Real suggestion line here.\n"
            "another line\n"
        )
        action = autonomous.parse_action(text)
        assert action["action"] == "suggest"
        # Should grab the first clean line
        assert "Real suggestion" in action["text"]

    def test_parse_action_empty_returns_wait(self):
        action = autonomous.parse_action("")
        assert action == {"action": "wait", "reason": "no parseable output"}

    def test_parse_action_only_whitespace_returns_wait(self):
        action = autonomous.parse_action("   \n\n   \t  \n")
        assert action == {"action": "wait", "reason": "no parseable output"}

    def test_parse_action_only_code_fence_returns_wait(self):
        """Nothing but code fences → fallback gets the fence line, but it's marked ``` → skip."""
        text = "```\n```\n```\n"
        action = autonomous.parse_action(text)
        # The fallback skips lines starting with ``` so we fall through to wait.
        assert action == {"action": "wait", "reason": "no parseable output"}

    def test_parse_action_suggest_text_truncated_to_200(self):
        """Fallback truncates the suggest text to 200 chars."""
        long_line = "x" * 500
        action = autonomous.parse_action(long_line)
        assert action["action"] == "suggest"
        assert len(action["text"]) == 200

    def test_parse_action_request_approval(self):
        action = autonomous.parse_action(
            '{"action": "request_approval", "reason": "why", "action_desc": "what"}'
        )
        assert action["action"] == "request_approval"
        assert action["reason"] == "why"


# ════════════════════════════════════════════════════════════════
# 8. execute_action — every action type
# ════════════════════════════════════════════════════════════════


class TestExecuteAction:
    """execute_action dispatches on the action field and returns (log, did_something)."""

    def _ctx(self):
        return (
            [],  # rules
            collections.deque(maxlen=3),  # notes
            autonomous.ToolSandbox(json.loads(json.dumps(autonomous.DEFAULT_CONFIG))),
            json.loads(json.dumps(autonomous.DEFAULT_CONFIG)),
        )

    def _capture(self, func):
        """Run func and capture stdout (for tools that print)."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            return func()

    # ── suggest ──────────────────────────────────────────────────
    def test_suggest_appends_to_stream(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        buf = io.StringIO()
        with redirect_stdout(buf):
            log, did = autonomous.execute_action(
                {"action": "suggest", "text": "Run the test"},
                [], collections.deque(maxlen=3), sandbox, config, None,
            )
        assert did is True
        assert "Run the test" in log
        assert "Run the test" in (in_workspace / "stream.md").read_text()

    def test_suggest_empty_noop(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        log, did = autonomous.execute_action(
            {"action": "suggest", "text": ""},
            [], collections.deque(maxlen=3), sandbox, config, None,
        )
        assert did is False
        assert "empty" in log.lower()

    def test_suggest_already_done_noop(self, in_workspace):
        # Pre-populate stream.md
        (in_workspace / "stream.md").write_text(
            "# stream.md — agent-owned. Suggestions appear here.\n"
            "Run the test\n"
        )
        _, _, sandbox, config = self._ctx()
        log, did = autonomous.execute_action(
            {"action": "suggest", "text": "Run the test"},
            [], collections.deque(maxlen=3), sandbox, config, None,
        )
        assert did is False
        assert "repeat" in log.lower()

    # ── tool ──────────────────────────────────────────────────────
    def test_tool_shell_execute_success(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        active_goal = {"id": "G001", "title": "Test"}
        log, did = autonomous.execute_action(
            {"action": "tool", "tool": "shell_execute", "command": "ls"},
            [], collections.deque(maxlen=3), sandbox, config, active_goal,
        )
        assert did is True
        assert "tool: ls" in log
        # Audit logged
        audit_lines = (in_workspace / "tools" / "audit.jsonl").read_text().splitlines()
        assert any("ls" in l for l in audit_lines)
        # Working memory got an observation entry
        records = autonomous.load_working_memory(goal_id="G001")
        assert any("Tool: `ls`" in r.get("content", "") for r in records)

    def test_tool_shell_execute_rejected(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        active_goal = {"id": "G001", "title": "Test"}
        log, did = autonomous.execute_action(
            {"action": "tool", "tool": "shell_execute", "command": "rm -rf /"},
            [], collections.deque(maxlen=3), sandbox, config, active_goal,
        )
        assert did is False
        assert "rejected" in log.lower()

    def test_tool_unknown_tool_name(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        active_goal = {"id": "G001", "title": "Test"}
        log, did = autonomous.execute_action(
            {"action": "tool", "tool": "frobnicate", "command": "ls"},
            [], collections.deque(maxlen=3), sandbox, config, active_goal,
        )
        assert did is False
        assert "unknown tool" in log.lower()

    def test_tool_with_no_active_goal_uses_None_for_memory(self, in_workspace):
        """Memory should still write even when active_goal is None."""
        _, _, sandbox, config = self._ctx()
        log, did = autonomous.execute_action(
            {"action": "tool", "tool": "shell_execute", "command": "ls"},
            [], collections.deque(maxlen=3), sandbox, config, None,
        )
        assert did is True
        # No goal_id → untyped record
        records = autonomous.load_working_memory()
        assert any("Tool: `ls`" in r.get("content", "") for r in records)

    # ── memory ────────────────────────────────────────────────────
    def test_memory_appends_record(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        active_goal = {"id": "G001", "title": "Test"}
        log, did = autonomous.execute_action(
            {"action": "memory", "type": "decision", "content": "Use pytest"},
            [], collections.deque(maxlen=3), sandbox, config, active_goal,
        )
        assert did is True
        assert "memory" in log.lower()
        records = autonomous.load_working_memory(goal_id="G001")
        assert any(r.get("content") == "Use pytest" for r in records)
        assert any(r.get("type") == "decision" for r in records)

    def test_memory_empty_content_noop(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        active_goal = {"id": "G001", "title": "Test"}
        log, did = autonomous.execute_action(
            {"action": "memory", "type": "decision", "content": ""},
            [], collections.deque(maxlen=3), sandbox, config, active_goal,
        )
        assert did is False
        assert "empty" in log.lower()

    # ── complete_goal ─────────────────────────────────────────────
    def test_complete_goal_updates_status(self, in_workspace):
        (in_workspace / "goals.md").write_text(
            "## [G001] Test goal\n"
            "### status: in_progress\n"
            "### priority: 1\n"
        )
        _, _, sandbox, config = self._ctx()
        active_goal = {"id": "G001", "title": "Test"}
        log, did = autonomous.execute_action(
            {"action": "complete_goal", "goal_id": "G001",
             "summary": "All tests pass"},
            [], collections.deque(maxlen=3), sandbox, config, active_goal,
        )
        assert did is True
        assert "completed" in log.lower()
        # Status updated in goals.md
        text = (in_workspace / "goals.md").read_text()
        assert "### status: complete" in text
        # Working memory got a goal_complete entry
        records = autonomous.load_working_memory(goal_id="G001")
        assert any(r.get("type") == "goal_complete" for r in records)

    def test_complete_goal_uses_active_goal_id_fallback(self, in_workspace):
        """If action omits goal_id, falls back to the active_goal's id."""
        (in_workspace / "goals.md").write_text(
            "## [G005] In progress\n"
            "### status: in_progress\n"
            "### priority: 1\n"
        )
        _, _, sandbox, config = self._ctx()
        active_goal = {"id": "G005", "title": "Test"}
        log, did = autonomous.execute_action(
            {"action": "complete_goal", "summary": "Done."},
            [], collections.deque(maxlen=3), sandbox, config, active_goal,
        )
        assert did is True
        text = (in_workspace / "goals.md").read_text()
        assert "complete" in text

    def test_complete_goal_no_auto_advance(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        config["goals"]["auto_advance"] = False
        active_goal = {"id": "G001", "title": "Test"}
        log, did = autonomous.execute_action(
            {"action": "complete_goal", "goal_id": "G001", "summary": "ok"},
            [], collections.deque(maxlen=3), sandbox, config, active_goal,
        )
        assert did is True
        assert "no auto-advance" in log.lower()
        # goals.md should NOT have been touched
        goals_text = (in_workspace / "goals.md").read_text()
        assert "complete" not in goals_text

    # ── wait ──────────────────────────────────────────────────────
    def test_wait_is_noop(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        log, did = autonomous.execute_action(
            {"action": "wait", "reason": "thinking"},
            [], collections.deque(maxlen=3), sandbox, config, None,
        )
        assert did is False
        assert "waiting" in log.lower()

    # ── request_approval ──────────────────────────────────────────
    def test_request_approval_writes_to_approval_md(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        log, did = autonomous.execute_action(
            {"action": "request_approval",
             "reason": "external API call",
             "action_desc": "Call Anthropic API"},
            [], collections.deque(maxlen=3), sandbox, config, None,
        )
        assert did is True
        assert "approval" in log.lower()
        assert (in_workspace / "approval.md").exists()
        text = (in_workspace / "approval.md").read_text()
        assert "Anthropic API" in text
        assert "external API call" in text

    # ── unknown ───────────────────────────────────────────────────
    def test_unknown_action_returns_error(self, in_workspace):
        _, _, sandbox, config = self._ctx()
        log, did = autonomous.execute_action(
            {"action": "frobnicate"},
            [], collections.deque(maxlen=3), sandbox, config, None,
        )
        assert did is False
        assert "unknown action" in log.lower()

    def test_no_action_defaults_to_wait(self, in_workspace):
        """Missing 'action' key defaults to 'wait'."""
        _, _, sandbox, config = self._ctx()
        log, did = autonomous.execute_action(
            {},  # empty dict
            [], collections.deque(maxlen=3), sandbox, config, None,
        )
        assert did is False
        assert "waiting" in log.lower()


# ════════════════════════════════════════════════════════════════
# 9. handle_command — every command
# ════════════════════════════════════════════════════════════════


class TestHandleCommand:
    """Returns (paused, force_step). Mutates notes/rules via shared mutable lists."""

    def _ctx(self):
        return [], collections.deque(maxlen=3)

    def _silent(self, cmd, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            return autonomous.handle_command(cmd, *args)

    # ── pause / resume ────────────────────────────────────────────
    def test_pause(self, in_workspace):
        paused, force = self._silent("!pause", *self._ctx(), False)
        assert paused is True
        assert force is False

    def test_resume(self, in_workspace):
        paused, force = self._silent("!resume", *self._ctx(), True)
        assert paused is False
        assert force is False

    # ── step ──────────────────────────────────────────────────────
    def test_step_when_paused(self, in_workspace):
        paused, force = self._silent("!step", *self._ctx(), True)
        assert paused is True
        assert force is True

    def test_step_when_running(self, in_workspace):
        paused, force = self._silent("!step", *self._ctx(), False)
        assert paused is False
        assert force is True

    # ── clear ─────────────────────────────────────────────────────
    def test_clear_resets_stream(self, in_workspace):
        (in_workspace / "stream.md").write_text("past suggestions here\nmore\n")
        paused, force = self._silent("!clear", *self._ctx(), False)
        assert paused is False
        assert force is False
        text = (in_workspace / "stream.md").read_text()
        assert text.startswith("# stream.md")
        assert "past suggestions" not in text

    # ── note ──────────────────────────────────────────────────────
    def test_note_appends_to_notes(self, in_workspace):
        _, notes = self._ctx()
        paused, force = self._silent("!note Always run tests", *self._ctx(), False)
        # rebuild with notes so we can check
        notes_local = []
        buf = io.StringIO()
        with redirect_stdout(buf):
            paused, force = autonomous.handle_command(
                "!note second note", [], notes_local, False
            )
        assert paused is False
        assert "second note" in notes_local

    def test_note_empty_noop(self, in_workspace):
        paused, force = self._silent("!note ", *self._ctx(), False)
        assert paused is False
        # Empty note is not appended (the function checks `if note`)

    # ── fix ───────────────────────────────────────────────────────
    def test_fix_adds_rule_and_persists(self, in_workspace):
        rules = []
        paused, force = self._silent(
            "!fix bad_code => good_code", *self._ctx(), False
        )
        # rerun to capture rules list
        rules_local = []
        buf = io.StringIO()
        with redirect_stdout(buf):
            paused, force = autonomous.handle_command(
                "!fix bad => good", [], rules_local, False
            )
        assert paused is False
        assert rules_local == [{"rejected": "bad", "accepted": "good"}]
        # Persisted to preferences.jsonl
        prefs = (in_workspace / "preferences.jsonl").read_text()
        assert "bad" in prefs and "good" in prefs

    def test_fix_without_arrow_prints_usage(self, in_workspace):
        buf = io.StringIO()
        with redirect_stdout(buf):
            paused, force = autonomous.handle_command(
                "!fix noarrow here", [], [], False
            )
        assert paused is False
        assert "usage" in buf.getvalue().lower()

    def test_fix_caps_at_max_fix(self, in_workspace):
        """!fix deletes oldest rules beyond MAX_FIX_RULES."""
        rules = [
            {"rejected": f"old{i}", "accepted": f"new{i}"}
            for i in range(autonomous.MAX_FIX_RULES + 3)
        ]
        paused, force = self._silent("!fix latest => latest2", rules, [])
        assert len(rules) == autonomous.MAX_FIX_RULES
        # Newest at end
        assert rules[-1] == {"rejected": "latest", "accepted": "latest2"}

    # ── approve / reject ──────────────────────────────────────────
    def test_approve_returns_current_state(self, in_workspace):
        paused, force = self._silent("!approve", *self._ctx(), False)
        assert paused is False
        assert force is False

    def test_reject_returns_current_state(self, in_workspace):
        paused, force = self._silent("!reject", *self._ctx(), True)
        assert paused is True
        assert force is False

    # ── goal ──────────────────────────────────────────────────────
    def test_goal_adds_to_goals_file(self, in_workspace):
        # Empty goals.md so the count starts at 0
        (in_workspace / "goals.md").write_text("")
        paused, force = self._silent(
            "!goal Write tests for autonomous", *self._ctx(), False
        )
        assert paused is False
        text = (in_workspace / "goals.md").read_text()
        assert "## [G001]" in text
        assert "Write tests for autonomous" in text
        assert "### status: pending" in text

    def test_goal_increments_id(self, in_workspace):
        (in_workspace / "goals.md").write_text(
            "## [G001] Existing\n"
            "### status: pending\n"
        )
        paused, force = self._silent("!goal Next goal", *self._ctx(), False)
        text = (in_workspace / "goals.md").read_text()
        assert "## [G002]" in text

    def test_goal_empty_noop(self, in_workspace):
        (in_workspace / "goals.md").write_text("")
        paused, force = self._silent("!goal ", *self._ctx(), False)
        text = (in_workspace / "goals.md").read_text()
        # No goal block added
        assert "## [G" not in text

    # ── !goals ────────────────────────────────────────────────────
    def test_goals_lists_parsed(self, in_workspace):
        # Empty goals.md so !goals lists nothing
        (in_workspace / "goals.md").write_text("")
        buf = io.StringIO()
        with redirect_stdout(buf):
            paused, force = autonomous.handle_command(
                "!goals", [], [], False
            )
        assert paused is False
        out = buf.getvalue()
        # Empty file → no goal lines printed
        assert "[" not in out or "Pending" not in out

    def test_goals_list_prints_pending_markers(self, in_workspace):
        (in_workspace / "goals.md").write_text(
            "## [G001] First\n"
            "### status: pending\n"
            "### priority: 1\n"
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            paused, force = autonomous.handle_command(
                "!goals", [], [], False
            )
        out = buf.getvalue()
        assert "G001" in out
        assert "First" in out
        # The status icon for pending is "○"
        assert "○" in out

    # ── !status ───────────────────────────────────────────────────
    def test_status_prints_summary(self, in_workspace):
        buf = io.StringIO()
        with redirect_stdout(buf):
            paused, force = autonomous.handle_command(
                "!status", [], [], False
            )
        out = buf.getvalue()
        assert "tick" in out.lower()
        assert "goals" in out.lower()

    # ── unknown ───────────────────────────────────────────────────
    def test_unknown_command_prints_help(self, in_workspace):
        buf = io.StringIO()
        with redirect_stdout(buf):
            paused, force = autonomous.handle_command(
                "!frobnicate", [], [], False
            )
        assert paused is False
        assert force is False
        out = buf.getvalue()
        assert "unknown" in out.lower()


# ════════════════════════════════════════════════════════════════
# 10. update_tick_status
# ════════════════════════════════════════════════════════════════


class TestUpdateTickStatus:
    """Writes tick.md with current/next timestamps and active-goal summary."""

    def test_update_tick_status_with_goal(self, in_workspace):
        goal = {"id": "G001", "title": "Run the test suite"}
        autonomous.update_tick_status(goal, "autonomous", "Did something")
        text = (in_workspace / "tick.md").read_text()
        assert "[G001] Run the test suite" in text
        assert "autonomous" in text
        assert "Did something" in text
        # Both timestamps present
        assert "Last tick:" in text
        assert "Next tick:" in text

    def test_update_tick_status_no_goal(self, in_workspace):
        autonomous.update_tick_status(None, "human", "Human triggered turn")
        text = (in_workspace / "tick.md").read_text()
        assert "Active goal:** none" in text
        assert "human" in text

    def test_update_tick_status_truncates_long_title(self, in_workspace):
        long_title = "X" * 200
        goal = {"id": "G001", "title": long_title}
        autonomous.update_tick_status(goal, "autonomous", "ok")
        text = (in_workspace / "tick.md").read_text()
        # Title is truncated to 50 chars
        assert "X" * 50 in text
        # Full title not present
        assert long_title not in text

    def test_update_tick_status_truncates_long_action(self, in_workspace):
        long_action = "Y" * 200
        autonomous.update_tick_status(None, "autonomous", long_action)
        text = (in_workspace / "tick.md").read_text()
        assert "Y" * 80 in text
        # Full action truncated
        assert long_action not in text

    def test_update_tick_status_timestamps_are_iso(self, in_workspace):
        autonomous.update_tick_status(None, "autonomous", "ok")
        text = (in_workspace / "tick.md").read_text()
        m = re.search(r"Last tick:\*\* (\S+)", text)
        assert m, "Last tick timestamp not found"
        ts = m.group(1)
        # Should parse as ISO with Z suffix (UTC)
        datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


# ════════════════════════════════════════════════════════════════
# 11. ensure_files — create everything when missing
# ════════════════════════════════════════════════════════════════


class TestEnsureFiles:
    """Creates workspace, stream, goals, config, tick files + tools/ directory."""

    def test_creates_all_required_files(self, in_tmp_path):
        # Sanity: nothing exists yet
        for fname in ("workspace.md", "stream.md", "goals.md",
                      "config.yaml", "tick.md", "tools"):
            assert not (in_tmp_path / fname).exists()

        autonomous.ensure_files()

        expected_files = [
            "workspace.md", "stream.md", "goals.md",
            "config.yaml", "tick.md",
        ]
        for fname in expected_files:
            assert (in_tmp_path / fname).exists(), f"{fname} not created"
        # tools/ directory created (not a file)
        assert (in_tmp_path / "tools").is_dir()

    def test_goals_template_has_example(self, in_tmp_path):
        autonomous.ensure_files()
        text = (in_tmp_path / "goals.md").read_text()
        # Template includes a sample ## [G001] block
        assert "## [G001]" in text
        assert "### status: pending" in text

    def test_config_template_has_sections(self, in_tmp_path):
        autonomous.ensure_files()
        text = (in_tmp_path / "config.yaml").read_text()
        assert "tick:" in text
        assert "goals:" in text

    def test_idempotent_does_not_overwrite_existing(self, in_tmp_path):
        autonomous.ensure_files()
        # User wrote their own workspace content
        (in_tmp_path / "workspace.md").write_text("# my custom workspace\n")
        # Run again
        autonomous.ensure_files()
        # Custom content preserved
        assert (in_tmp_path / "workspace.md").read_text() == "# my custom workspace\n"

    def test_creates_tools_audit_on_demand(self, in_tmp_path):
        """tools/audit.jsonl is created by ToolSandbox.execute, not ensure_files."""
        autonomous.ensure_files()
        assert (in_tmp_path / "tools").is_dir()
        # audit.jsonl not yet
        assert not (in_tmp_path / "tools" / "audit.jsonl").exists()

        sandbox = autonomous.ToolSandbox(json.loads(json.dumps(autonomous.DEFAULT_CONFIG)))
        sandbox.check_allowed("ls")  # noop
        # Audit only written on actual execute, not check_allowed
        assert not (in_tmp_path / "tools" / "audit.jsonl").exists()


# ════════════════════════════════════════════════════════════════
# 12. _already_done — exact & normalized match
# ════════════════════════════════════════════════════════════════


class TestAlreadyDone:
    """Normalized comparison: strip, lowercase, lstrip('-*#> `'), rstrip('.')."""

    def test_no_match_when_stream_empty(self, in_workspace):
        # Default stream.md only has a header
        assert autonomous._already_done("Run pytest") is False

    def test_no_match_for_different_text(self, in_workspace):
        (in_workspace / "stream.md").write_text(
            "# stream.md\nRun pytest\n"
        )
        assert autonomous._already_done("Write docs") is False

    def test_exact_match(self, in_workspace):
        (in_workspace / "stream.md").write_text(
            "# stream.md\nRun pytest\n"
        )
        assert autonomous._already_done("Run pytest") is True

    def test_case_insensitive(self, in_workspace):
        (in_workspace / "stream.md").write_text(
            "# stream.md\nRun pytest\n"
        )
        assert autonomous._already_done("run pytest") is True
        assert autonomous._already_done("RUN PYTEST") is True
        assert autonomous._already_done("rUn PyTeSt") is True

    def test_strips_leading_bullets(self, in_workspace):
        """Leading -, *, #, >, ` chars are stripped."""
        (in_workspace / "stream.md").write_text(
            "# stream.md\n- Run pytest\n* Run pytest\n# Run pytest\n> Run pytest\n"
        )
        assert autonomous._already_done("Run pytest") is True

    def test_strips_trailing_period(self, in_workspace):
        (in_workspace / "stream.md").write_text(
            "# stream.md\nRun pytest.\n"
        )
        assert autonomous._already_done("Run pytest") is True

    def test_strips_whitespace(self, in_workspace):
        (in_workspace / "stream.md").write_text(
            "# stream.md\n   Run pytest   \n"
        )
        assert autonomous._already_done("Run pytest") is True

    def test_normalized_match_with_bullet_and_period(self, in_workspace):
        (in_workspace / "stream.md").write_text(
            "# stream.md\n- Run pytest.\n"
        )
        # Suggestion without bullet/period still matches
        assert autonomous._already_done("Run pytest") is True

    def test_completely_different_still_false(self, in_workspace):
        (in_workspace / "stream.md").write_text(
            "# stream.md\nFoo bar baz\n"
        )
        assert autonomous._already_done("Run pytest") is False


# ════════════════════════════════════════════════════════════════
# Bonus: ToolSandbox
# ════════════════════════════════════════════════════════════════


class TestToolSandbox:
    """Tool sandbox allowlist checks + execution + audit logging."""

    def test_check_allowed_accepts_known_executable(self):
        cfg = json.loads(json.dumps(autonomous.DEFAULT_CONFIG))
        sandbox = autonomous.ToolSandbox(cfg)
        ok, reason = sandbox.check_allowed("ls")
        assert ok is True
        assert reason == "ok"

    def test_check_allowed_rejects_unknown_executable(self):
        cfg = json.loads(json.dumps(autonomous.DEFAULT_CONFIG))
        sandbox = autonomous.ToolSandbox(cfg)
        ok, reason = sandbox.check_allowed("curl")
        # curl is not in allowed_executables
        assert ok is False
        assert "curl" in reason

    def test_check_allowed_rejects_command_not_matching_pattern(self):
        cfg = json.loads(json.dumps(autonomous.DEFAULT_CONFIG))
        sandbox = autonomous.ToolSandbox(cfg)
        ok, reason = sandbox.check_allowed("pytest something_random")
        # pytest matches allowed_executables but not the pytest pattern
        assert ok is False

    def test_check_allowed_empty_command(self):
        cfg = json.loads(json.dumps(autonomous.DEFAULT_CONFIG))
        sandbox = autonomous.ToolSandbox(cfg)
        ok, reason = sandbox.check_allowed("")
        assert ok is False
        assert "empty" in reason.lower()

    def test_execute_logs_to_audit(self, in_workspace):
        cfg = json.loads(json.dumps(autonomous.DEFAULT_CONFIG))
        sandbox = autonomous.ToolSandbox(cfg)
        sandbox.execute("ls")
        audit = (in_workspace / "tools" / "audit.jsonl").read_text()
        lines = [l for l in audit.splitlines() if l.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["command"] == "ls"
        assert "id" in rec
        assert "timestamp" in rec
        assert rec["allowed"] is True

    def test_execute_rejected_writes_audit(self, in_workspace):
        cfg = json.loads(json.dumps(autonomous.DEFAULT_CONFIG))
        sandbox = autonomous.ToolSandbox(cfg)
        result = sandbox.execute("curl https://example.com")
        assert result["status"] == "rejected"
        # Still audited
        audit = (in_workspace / "tools" / "audit.jsonl").read_text()
        rec = json.loads(audit.strip().splitlines()[-1])
        assert rec["allowed"] is False

    def test_log_fix_appends_to_memory_file(self, in_workspace):
        autonomous.log_fix("foo()", "bar()")
        text = (in_workspace / "preferences.jsonl").read_text()
        assert "foo()" in text and "bar()" in text

    def test_invalid_regex_pattern_silently_skipped(self):
        cfg = json.loads(json.dumps(autonomous.DEFAULT_CONFIG))
        cfg["tools"]["allowed_patterns"].append("[invalid(")  # bad regex
        sandbox = autonomous.ToolSandbox(cfg)
        # Should not raise; bad pattern dropped silently
        assert len(sandbox.allowed_patterns) == len(cfg["tools"]["allowed_patterns"]) - 1
