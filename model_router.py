#!/usr/bin/env python3
"""
model_router.py — the flock dispatcher for local multi-model orchestration.

A collection of tiny models that load and unload like dogs entering and leaving
the pen. The smallest model runs permanently — it's the heartbeat, the iterator,
the dispatcher. When it's stumped, it calls a specialist. The specialist loads,
solves the problem, gets unloaded. VRAM is the pen.

Designed to run alongside autonomous.py. Zero dependencies. Stdlib only.

Usage:
  from model_router import ModelRouter
  router = ModelRouter()
  result = router.dispatch(task_type="coding", prompt="...", context="...")

Or standalone as an escalation service:
  python model_router.py  # watches escalation.md, dispatches specialists

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │  VRAM BUDGET (e.g. 8GB)                              │
  │                                                      │
  │  ┌─────────────┐  always loaded                      │
  │  │ Iterator    │  qwen2.5:0.5b (~400MB)             │
  │  │ (sheepdog)  │  runs the tick loop                 │
  │  └──────┬──────┘                                     │
  │         │ escalates                                  │
  │  ┌──────▼──────┐  loaded on demand                   │
  │  │ Specialist  │  one at a time (budget permitting)  │
  │  │ (big dog)   │  unloaded after use                 │
  │  └─────────────┘                                     │
  └──────────────────────────────────────────────────────┘
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SERVER_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
SERVER_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
SERVER_PS_URL   = f"{OLLAMA_HOST}/api/ps"  # loaded models
REQUEST_TIMEOUT = 180

# The iterator — always loaded, always running
ITERATOR_MODEL = "qwen2.5:0.5b"

# VRAM budget in GB (auto-detected or configured)
VRAM_BUDGET_GB = float(os.environ.get("VRAM_BUDGET_GB", "8"))

# ═══════════════════════════════════════════════════════════════
# Model Roster — the flock
# ═══════════════════════════════════════════════════════════════

ROSTER = [
    # ── Tier 0: Always loaded (the iterator) ──────────────────
    {
        "name": "qwen2.5:0.5b",
        "tag": "iterator",
        "specialty": "dispatch",
        "size_gb": 0.4,
        "vram_gb": 0.5,
        "speed_rank": 1,        # 1 = fastest
        "always_loaded": True,
        "description": "The heartbeat. Runs the tick loop, dispatches to specialists.",
        "strengths": ["fast_iteration", "simple_suggestions", "routing", "pattern_matching"],
        "weaknesses": ["complex_reasoning", "long_context", "code_generation", "vision"],
        "ollama_pull": "ollama pull qwen2.5:0.5b",
    },
    # ── Tier 1: Fast specialists (load in <3s) ────────────────
    {
        "name": "qwen2.5-coder:1.5b",
        "tag": "fast-coder",
        "specialty": "coding",
        "size_gb": 1.0,
        "vram_gb": 1.2,
        "speed_rank": 2,
        "always_loaded": False,
        "description": "Fast code generation and review. Good for boilerplate and simple fixes.",
        "strengths": ["code_generation", "code_review", "syntax_fixes", "test_writing"],
        "weaknesses": ["architecture", "complex_logic", "long_files"],
        "ollama_pull": "ollama pull qwen2.5-coder:1.5b",
    },
    {
        "name": "deepseek-r1:1.5b",
        "tag": "fast-thinker",
        "specialty": "reasoning",
        "size_gb": 1.0,
        "vram_gb": 1.2,
        "speed_rank": 3,
        "always_loaded": False,
        "description": "Chain-of-thought reasoning. Thinks before it speaks. Good for puzzles.",
        "strengths": ["step_by_step_reasoning", "math", "logic_puzzles", "debugging"],
        "weaknesses": ["creative_writing", "code_generation", "speed"],
        "ollama_pull": "ollama pull deepseek-r1:1.5b",
    },
    {
        "name": "gemma2:2b",
        "tag": "generalist",
        "specialty": "general",
        "size_gb": 1.5,
        "vram_gb": 1.8,
        "speed_rank": 4,
        "always_loaded": False,
        "description": "Google's small model. Well-rounded. Good fallback for general tasks.",
        "strengths": ["general_qa", "summarization", "explanation", "translation"],
        "weaknesses": ["code_generation", "vision", "deep_reasoning"],
        "ollama_pull": "ollama pull gemma2:2b",
    },
    # ── Tier 2: Medium specialists (~2GB, load in ~5s) ────────
    {
        "name": "qwen2.5-coder:3b",
        "tag": "coder",
        "specialty": "coding",
        "size_gb": 2.0,
        "vram_gb": 2.5,
        "speed_rank": 5,
        "always_loaded": False,
        "description": "The workhorse coder. Good architecture sense, solid implementations.",
        "strengths": ["code_generation", "architecture", "refactoring", "code_review", "debugging"],
        "weaknesses": ["vision", "creative_writing"],
        "ollama_pull": "ollama pull qwen2.5-coder:3b",
    },
    {
        "name": "qwen2.5:3b",
        "tag": "thinker",
        "specialty": "reasoning",
        "size_gb": 2.0,
        "vram_gb": 2.5,
        "speed_rank": 6,
        "always_loaded": False,
        "description": "General reasoning and planning. Good for breaking down complex goals.",
        "strengths": ["planning", "analysis", "reasoning", "writing", "explanation"],
        "weaknesses": ["code_generation", "vision"],
        "ollama_pull": "ollama pull qwen2.5:3b",
    },
    {
        "name": "phi3:3.8b",
        "tag": "scholar",
        "specialty": "reasoning",
        "size_gb": 2.5,
        "vram_gb": 3.0,
        "speed_rank": 7,
        "always_loaded": False,
        "description": "Microsoft's Phi-3. Punches above its weight on reasoning and math.",
        "strengths": ["math", "logic", "reasoning", "structured_analysis"],
        "weaknesses": ["code_generation", "creative", "vision"],
        "ollama_pull": "ollama pull phi3",
    },
    # ── Tier 3: Heavy specialists (~4.5GB, load in ~10s) ──────
    {
        "name": "qwen2.5-coder:7b",
        "tag": "architect",
        "specialty": "coding",
        "size_gb": 4.5,
        "vram_gb": 5.0,
        "speed_rank": 9,
        "always_loaded": False,
        "description": "The senior engineer. Complex architecture, multi-file reasoning, deep debugging.",
        "strengths": ["complex_architecture", "multi_file_reasoning", "deep_debugging", "system_design"],
        "weaknesses": ["speed", "vision"],
        "ollama_pull": "ollama pull qwen2.5-coder:7b",
    },
    {
        "name": "deepseek-r1:7b",
        "tag": "sage",
        "specialty": "reasoning",
        "size_gb": 4.5,
        "vram_gb": 5.0,
        "speed_rank": 10,
        "always_loaded": False,
        "description": "Deep chain-of-thought reasoning. The wise elder. Slow but thorough.",
        "strengths": ["deep_reasoning", "math_proofs", "complex_planning", "edge_case_analysis"],
        "weaknesses": ["speed", "code_generation", "vision"],
        "ollama_pull": "ollama pull deepseek-r1:7b",
    },
    {
        "name": "mistral:7b",
        "tag": "general-heavy",
        "specialty": "general",
        "size_gb": 4.5,
        "vram_gb": 5.0,
        "speed_rank": 9,
        "always_loaded": False,
        "description": "Mistral 7B. Reliable general-purpose model. Good at following instructions.",
        "strengths": ["instruction_following", "general_qa", "summarization", "writing", "analysis"],
        "weaknesses": ["vision", "code_generation"],
        "ollama_pull": "ollama pull mistral",
    },
    # ── Tier 4: Multimodal specialists ────────────────────────
    {
        "name": "qwen2.5-vl:3b",
        "tag": "eyes",
        "specialty": "vision",
        "size_gb": 2.0,
        "vram_gb": 2.5,
        "speed_rank": 6,
        "always_loaded": False,
        "description": "Vision-language model. Can see images, screenshots, diagrams.",
        "strengths": ["image_understanding", "screenshot_analysis", "diagram_reading", "ui_inspection"],
        "weaknesses": ["code_generation", "long_context", "deep_reasoning"],
        "ollama_pull": "ollama pull qwen2.5-vl:3b",
    },
    {
        "name": "llava:7b",
        "tag": "watchman",
        "specialty": "vision",
        "size_gb": 4.5,
        "vram_gb": 5.0,
        "speed_rank": 10,
        "always_loaded": False,
        "description": "Heavy vision model. Detailed image analysis, OCR, complex visual reasoning.",
        "strengths": ["detailed_image_analysis", "ocr", "visual_reasoning", "chart_reading"],
        "weaknesses": ["speed", "code_generation"],
        "ollama_pull": "ollama pull llava:7b",
    },
]

# Build lookup indexes
BY_NAME = {m["name"]: m for m in ROSTER}
BY_TAG = {m["tag"]: m for m in ROSTER}
BY_SPECIALTY = {}
for m in ROSTER:
    BY_SPECIALTY.setdefault(m["specialty"], []).append(m)

# ═══════════════════════════════════════════════════════════════
# HTTP helpers (stdlib)
# ═══════════════════════════════════════════════════════════════

def _get_json(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}

def _post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

# ═══════════════════════════════════════════════════════════════
# Ollama model management
# ═══════════════════════════════════════════════════════════════

def list_available_models():
    """List all models available in Ollama (already pulled)."""
    data = _get_json(SERVER_TAGS_URL)
    return {m["name"]: m for m in data.get("models", [])}

def list_loaded_models():
    """List currently loaded (resident in VRAM) models."""
    data = _get_json(SERVER_PS_URL)
    return {m["name"]: m for m in data.get("models", [])}

def unload_model(model_name):
    """Unload a model from VRAM."""
    try:
        body = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False

def chat(model_name, messages, temperature=0.4, max_tokens=200, num_ctx=8192):
    """Send a chat completion to a specific model."""
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    resp = _post_json(SERVER_CHAT_URL, payload, REQUEST_TIMEOUT)
    return resp.get("message", {}).get("content", "")

# ═══════════════════════════════════════════════════════════════
# ModelRouter — the flock dispatcher
# ═══════════════════════════════════════════════════════════════

class ModelRouter:
    """
    Manages the flock of local models. The iterator runs permanently.
    Specialists are loaded on demand within the VRAM budget.
    """

    def __init__(self, vram_budget_gb=None):
        self.vram_budget = vram_budget_gb or VRAM_BUDGET_GB
        self.available = list_available_models()
        self.loaded = list_loaded_models()

        if ITERATOR_MODEL not in self.available:
            print(f"⚠ Iterator model '{ITERATOR_MODEL}' not found.")
            print(f"  Run: ollama pull {ITERATOR_MODEL}")
            sys.exit(1)

        # Ensure iterator is loaded
        if ITERATOR_MODEL not in self.loaded:
            print(f"* loading iterator ({ITERATOR_MODEL})...")
            chat(ITERATOR_MODEL, [{"role": "user", "content": "ready"}],
                 max_tokens=5, temperature=0.1)
            self.loaded = list_loaded_models()

        print(f"* flock ready — iterator: {ITERATOR_MODEL}")
        print(f"* VRAM budget: {self.vram_budget:.1f} GB")
        self._report_status()

    def _current_vram_usage(self):
        """Estimate current VRAM usage from loaded models."""
        total = 0.0
        loaded = list_loaded_models()
        for name in loaded:
            model = BY_NAME.get(name)
            if model:
                total += model["vram_gb"]
            else:
                # Unknown model — estimate from Ollama size
                info = loaded.get(name, {})
                size = info.get("size", 0) / (1024**3)  # bytes → GB
                total += size
        return total

    def _make_room(self, needed_gb):
        """Unload models to free VRAM. Never unloads the iterator."""
        current = self._current_vram_usage()
        if current + needed_gb <= self.vram_budget:
            return True

        # Sort loaded models by size (biggest first), excluding iterator
        loaded = list_loaded_models()
        candidates = []
        for name in loaded:
            if name == ITERATOR_MODEL:
                continue
            model = BY_NAME.get(name)
            if model:
                candidates.append(model)

        # Sort by vram_gb descending — unload biggest first for maximum space
        candidates.sort(key=lambda m: m["vram_gb"], reverse=True)

        for model in candidates:
            if current + needed_gb <= self.vram_budget:
                return True
            print(f"* unloading {model['name']} ({model['vram_gb']:.1f}GB) to make room...")
            unload_model(model["name"])
            current -= model["vram_gb"]
            time.sleep(0.5)

        return current + needed_gb <= self.vram_budget

    def select_specialist(self, task_type, hint=None):
        """
        Select the best specialist for a task type.
        Prefers smaller/faster models when possible.
        """
        candidates = BY_SPECIALTY.get(task_type, [])
        if not candidates:
            # Unknown task type — use general
            candidates = BY_SPECIALTY.get("general", [])

        # Filter to models that are actually pulled
        pulled = [m for m in candidates if m["name"] in self.available]
        if not pulled:
            print(f"⚠ No {task_type} specialists pulled. Available:")
            for m in candidates:
                print(f"    {m['ollama_pull']}")
            return None

        # Sort by speed_rank (fastest first) — prefer cheap escalation
        pulled.sort(key=lambda m: m["speed_rank"])

        # If hint provided, try to find a match in strengths
        if hint:
            hint_lower = hint.lower()
            for m in pulled:
                if any(h in s for s in m["strengths"] for h in [hint_lower]):
                    return m

        return pulled[0]  # Fastest available specialist

    def dispatch(self, task_type, prompt, context=None, system=None,
                 temperature=0.4, max_tokens=300):
        """
        Load a specialist, run the task, return the result.
        The specialist is kept loaded briefly for follow-ups (Ollama default).
        """
        specialist = self.select_specialist(task_type)
        if not specialist:
            return {"error": f"No specialist available for '{task_type}'"}

        needed = specialist["vram_gb"]
        if not self._make_room(needed):
            return {"error": f"Cannot fit {specialist['name']} ({needed}GB) in VRAM budget"}

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
            messages.append({"role": "assistant", "content": "Understood."})
        messages.append({"role": "user", "content": prompt})

        print(f"* dispatching to {specialist['tag']} ({specialist['name']})...")

        try:
            result = chat(specialist["name"], messages, temperature, max_tokens)
            self.loaded = list_loaded_models()
            return {
                "model": specialist["name"],
                "tag": specialist["tag"],
                "result": result,
            }
        except Exception as e:
            return {"error": str(e), "model": specialist["name"]}

    def iterate(self, prompt, context=None, system=None):
        """
        Run the iterator model. This is the fast, always-loaded heartbeat.
        Should be used for every tick. Only escalate with dispatch() when needed.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
            messages.append({"role": "assistant", "content": "Understood."})
        messages.append({"role": "user", "content": prompt})

        try:
            result = chat(ITERATOR_MODEL, messages,
                         temperature=0.3, max_tokens=150, num_ctx=4096)
            return result
        except Exception as e:
            return f"[iterator error: {e}]"

    def escalate_check(self, iterator_output):
        """
        Check if the iterator's output signals escalation needed.
        Returns (needs_escalation, task_type, hint) or (False, None, None).
        """
        output = iterator_output.lower().strip()

        # Direct escalation signals
        escalation_markers = [
            ("escalate:coding", "coding"),
            ("escalate:reasoning", "reasoning"),
            ("escalate:vision", "vision"),
            ("escalate:general", "general"),
            ("need_specialist:", None),
            ("i'm not sure", "general"),
            ("i don't know", "general"),
            ("this is complex", "reasoning"),
            ("requires deeper analysis", "reasoning"),
        ]

        for marker, task_type in escalation_markers:
            if marker in output:
                # Try to extract hint after marker
                hint = None
                if ":" in output:
                    _, _, after = output.partition(marker)
                    hint = after.strip()[:100] if after.strip() else None

                if task_type is None:
                    # Parse task type from the signal
                    if "coding" in output or "code" in output:
                        task_type = "coding"
                    elif "vision" in output or "image" in output:
                        task_type = "vision"
                    elif "reason" in output or "think" in output:
                        task_type = "reasoning"
                    else:
                        task_type = "general"

                return True, task_type, hint

        # Check for repeated identical outputs (stuck signal)
        return False, None, None

    def _report_status(self):
        """Print a status report of the flock."""
        loaded = list_loaded_models()
        usage = self._current_vram_usage()
        print(f"* loaded models: {list(loaded.keys())}")
        print(f"* VRAM: {usage:.1f} / {self.vram_budget:.1f} GB")
        pulled = [m["name"] for m in ROSTER if m["name"] in self.available]
        missing = [m["name"] for m in ROSTER if m["name"] not in self.available]
        print(f"* roster: {len(pulled)} pulled, {len(missing)} available to pull")
        if missing:
            print(f"* not yet pulled: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")

    def roster_report(self):
        """Return a formatted roster report."""
        lines = ["# Model Flock Roster", ""]
        for tier_start, tier_end, tier_name in [
            (0, 1, "Tier 0 — Always Loaded (Iterator)"),
            (1, 4, "Tier 1 — Fast Specialists (<3s load)"),
            (4, 7, "Tier 2 — Medium Specialists (~5s load)"),
            (7, 10, "Tier 3 — Heavy Specialists (~10s load)"),
            (10, 12, "Tier 4 — Multimodal Specialists"),
        ]:
            lines.append(f"## {tier_name}")
            lines.append("")
            for m in ROSTER[tier_start:tier_end]:
                status = "✓ pulled" if m["name"] in self.available else "○ not pulled"
                loaded = " (loaded)" if m["name"] in self.loaded else ""
                lines.append(f"### {m['tag']} — `{m['name']}` {status}{loaded}")
                lines.append(f"- **Specialty:** {m['specialty']}")
                lines.append(f"- **VRAM:** {m['vram_gb']:.1f} GB | **Speed:** rank {m['speed_rank']}")
                lines.append(f"- **Strengths:** {', '.join(m['strengths'])}")
                lines.append(f"- **Weaknesses:** {', '.join(m['weaknesses'])}")
                lines.append(f"- {m['description']}")
                lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Escalation protocol — the iterator calls for help
# ═══════════════════════════════════════════════════════════════

ESCALATION_SYSTEM = (
    "You are the Iterator — a tiny, fast model that runs continuously. "
    "Your job is to handle simple steps and ESCALATE complex ones.\n\n"
    "For each task, output ONE line:\n"
    "  - The answer if you can handle it (simple, fast)\n"
    "  - `ESCALATE:<type>:<hint>` if you need a specialist\n\n"
    "Types: coding, reasoning, vision, general\n"
    "Example: ESCALATE:coding:multi-file refactor of auth system\n"
    "Example: ESCALATE:reasoning:complex edge case in timeout logic\n\n"
    "Escalate generously. You are small. Pride costs more than VRAM."
)

def run_with_escalation(router, prompt, context=None):
    """
    Full escalation flow: iterator tries → escalates if needed → specialist solves.
    """
    # Step 1: Iterator tries
    iterator_output = router.iterate(
        prompt, context=context, system=ESCALATION_SYSTEM)

    # Step 2: Check for escalation
    needs_escalation, task_type, hint = router.escalate_check(iterator_output)

    if not needs_escalation:
        return {
            "handled_by": "iterator",
            "result": iterator_output,
            "escalated": False,
        }

    # Step 3: Dispatch to specialist
    print(f"* iterator escalated → {task_type}" + (f" ({hint})" if hint else ""))
    result = router.dispatch(task_type, prompt, context=context,
                            system=f"You are a {task_type} specialist. Solve this precisely.")

    return {
        "handled_by": result.get("model", task_type),
        "result": result.get("result", result.get("error", "unknown")),
        "escalated": True,
        "task_type": task_type,
        "hint": hint,
    }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local multi-model flock dispatcher")
    parser.add_argument("--roster", action="store_true", help="Print the model roster")
    parser.add_argument("--status", action="store_true", help="Print current flock status")
    parser.add_argument("--task", help="Task type: coding, reasoning, vision, general")
    parser.add_argument("--prompt", help="Prompt for the task")
    parser.add_argument("--context", help="Additional context")
    parser.add_argument("--escalate", action="store_true",
                       help="Run with escalation (iterator tries first)")
    args = parser.parse_args()

    router = ModelRouter()

    if args.roster:
        print(router.roster_report())
        return

    if args.status:
        router._report_status()
        return

    if args.prompt:
        if args.escalate:
            result = run_with_escalation(router, args.prompt, args.context)
        elif args.task:
            result = router.dispatch(args.task, args.prompt, args.context)
        else:
            result = {"result": router.iterate(args.prompt, args.context)}

        print(f"\n[handled by: {result.get('handled_by', 'unknown')}]")
        print(result.get("result", result.get("error", "no output")))
        return

    # Default: print roster
    print(router.roster_report())


if __name__ == "__main__":
    main()
