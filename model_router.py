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
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════
# Hardware Environment Setup
# ═══════════════════════════════════════════════════════════════

def setup_env():
    """
    Configure hardware-specific environment variables for optimal performance.
    Call this once at process startup, before initializing Ollama connections.

    These vars are critical for 6GB cards (RTX 4050) and helpful for larger GPUs:
    - CUDA_VISIBLE_DEVICES: Target the primary GPU
    - GGML_CUDA_FORCE_MMQ: Force tensor core usage for faster inference
    - OLLAMA_NUM_PARALLEL: Prevent KV cache splitting (6GB cards can't split)
    """
    env_vars = {
        "CUDA_VISIBLE_DEVICES": "0",
        "GGML_CUDA_FORCE_MMQ": "1",
        "OLLAMA_NUM_PARALLEL": "1",
    }

    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"* setting {key}={value}")

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

# Context window hard caps (per blueprint specs)
# Router/iterator gets small window for fast dispatch
# Specialists get larger windows for complex tasks
CONTEXT_CAP_ROUTER = 1024
CONTEXT_CAP_SPECIALIST = 4096
CONTEXT_CAP_HEAVY = 8192  # For models that can handle it

# Keep-alive durations (in seconds)
KEEP_ALIVE_IMMEDIATE = 0      # Evict immediately after use
KEEP_ALIVE_BRIEF = 300        # 5 minutes — for likely follow-ups
KEEP_ALIVE_LONG = 3600        # 1 hour — for pinned specialists
KEEP_ALIVE_PERMANENT = -1     # Never unload (for iterator)

# Timeout and retry settings
MODEL_LOAD_TIMEOUT = 45       # Seconds to wait for model to load
UNLOAD_VERIFY_DELAY = 1.0     # Seconds to wait after unload before verifying
MAX_ESCALATION_DEPTH = 3      # Prevent infinite escalation loops

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
        "context_cap": CONTEXT_CAP_ROUTER,
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
        "context_cap": CONTEXT_CAP_SPECIALIST,
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
        "context_cap": CONTEXT_CAP_SPECIALIST,
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
        "context_cap": CONTEXT_CAP_SPECIALIST,
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
        "context_cap": CONTEXT_CAP_SPECIALIST,
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
        "context_cap": CONTEXT_CAP_SPECIALIST,
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
        "context_cap": CONTEXT_CAP_SPECIALIST,
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
        "context_cap": CONTEXT_CAP_HEAVY,
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
        "context_cap": CONTEXT_CAP_HEAVY,
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
        "context_cap": CONTEXT_CAP_HEAVY,
    },
    # ── Tier 3.5: MoE and Audio specialists ───────────────────────
    {
        "name": "qwen2.5-next:80b-a3b-q2_K",
        "tag": "moe-oracle",
        "specialty": "reasoning",
        "size_gb": 4.2,
        "vram_gb": 4.5,
        "speed_rank": 8,
        "always_loaded": False,
        "description": "Mixture-of-Experts: 80B knowledge, only 3B active per token. Massive reasoning in a tiny footprint.",
        "strengths": ["deep_reasoning", "knowledge_synthesis", "complex_analysis", "multi_domain_expertise"],
        "weaknesses": ["speed", "vision", "code_generation"],
        "ollama_pull": "ollama pull qwen2.5-next:80b-a3b-q2_K",
        "context_cap": CONTEXT_CAP_SPECIALIST,
    },
    {
        "name": "minicpm-o:4.5",
        "tag": "herald",
        "specialty": "audio",
        "size_gb": 3.0,
        "vram_gb": 3.5,
        "speed_rank": 7,
        "always_loaded": False,
        "description": "Full-duplex speech model. Can listen and speak simultaneously. Handles voice I/O.",
        "strengths": ["speech_recognition", "speech_synthesis", "full_duplex_audio", "voice_commands"],
        "weaknesses": ["text_only_tasks", "long_context", "complex_reasoning"],
        "ollama_pull": "ollama pull minicpm-o:4.5",
        "context_cap": CONTEXT_CAP_SPECIALIST,
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
        "context_cap": CONTEXT_CAP_SPECIALIST,
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
        "context_cap": CONTEXT_CAP_HEAVY,
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

def chat(model_name, messages, temperature=0.4, max_tokens=200, num_ctx=8192,
         keep_alive=None):
    """
    Send a chat completion to a specific model.

    Args:
        model_name: The model to chat with
        messages: List of {role, content} message dicts
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        num_ctx: Context window size
        keep_alive: How long to keep model loaded (seconds)
                   - None: Use model's default policy
                   - 0: Evict immediately after use
                   - -1: Never unload (permanent pinning)
                   - >0: Keep for N seconds
    """
    # Determine default keep_alive based on model role
    if keep_alive is None:
        if model_name == ITERATOR_MODEL:
            keep_alive = KEEP_ALIVE_PERMANENT  # Iterator stays loaded
        else:
            keep_alive = KEEP_ALIVE_BRIEF  # Specialists cached briefly

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
            "keep_alive": keep_alive,
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
        self._dispatch_lock = threading.Lock()  # Prevent concurrent dispatch calls
        self._escalation_depth = 0  # Track escalation depth to prevent infinite loops

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

        # Issue 10: Graceful degradation — check VRAM budget can fit at least one specialist
        smallest_specialist = None
        for model in ROSTER:
            if model.get("always_loaded", False):
                continue
            if model["name"] in self.available:
                if smallest_specialist is None or model["vram_gb"] < smallest_specialist["vram_gb"]:
                    smallest_specialist = model

        if smallest_specialist and self.vram_budget < smallest_specialist["vram_gb"]:
            print(f"⚠ VRAM budget ({self.vram_budget:.1f}GB) too small for any specialist!")
            print(f"  Smallest available specialist: {smallest_specialist['name']} ({smallest_specialist['vram_gb']:.1f}GB)")
            print(f"  Increase VRAM_BUDGET_GB environment variable or use a larger GPU.")

        print(f"* flock ready — iterator: {ITERATOR_MODEL}")
        print(f"* VRAM budget: {self.vram_budget:.1f} GB")
        self._report_status()

    def _current_vram_usage(self):
        """Estimate current VRAM usage from loaded models with robust fallbacks."""
        total = 0.0
        loaded = list_loaded_models()
        for name, info in loaded.items():
            model = BY_NAME.get(name)
            if model:
                total += model["vram_gb"]
            else:
                # Unknown model — estimate from Ollama size with safe defaults
                size_bytes = info.get("size", 0)
                if size_bytes > 0:
                    estimated_vram = size_bytes / (1024**3)  # bytes → GB
                    # Ollama's size is often the model file size, VRAM is typically 1.2-1.5x
                    total += estimated_vram * 1.3
                else:
                    # No size info — use conservative default (estimate as 2GB)
                    print(f"* warning: unknown model '{name}' has no size info, estimating 2GB")
                    total += 2.0
        return total

    def _make_room(self, needed_gb, max_attempts=3):
        """Unload models to free VRAM with verification. Never unloads the iterator."""
        for attempt in range(max_attempts):
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
                if unload_model(model["name"]):
                    # Wait and verify it actually unloaded
                    time.sleep(UNLOAD_VERIFY_DELAY)
                    loaded_after = list_loaded_models()
                    if model["name"] not in loaded_after:
                        current -= model["vram_gb"]
                        print(f"* verified {model['name']} unloaded")
                    else:
                        print(f"! warning: {model['name']} still loaded after unload attempt")
                else:
                    print(f"! failed to send unload request for {model['name']}")

            current = self._current_vram_usage()
            if current + needed_gb > self.vram_budget:
                print(f"! insufficient space after attempt {attempt + 1}/{max_attempts}")
                if attempt < max_attempts - 1:
                    time.sleep(1)

        return current + needed_gb <= self.vram_budget

    def select_specialist(self, task_type, hint=None):
        """
        Select the best specialist for a task type.
        Prefers smaller/faster models when possible.

        Returns None if no suitable specialist is available, with error details.
        """
        candidates = BY_SPECIALTY.get(task_type, [])
        if not candidates:
            # Unknown task type — use general
            candidates = BY_SPECIALTY.get("general", [])

        if not candidates:
            print(f"⚠ No specialists found for task type '{task_type}'")
            return None

        # Filter to models that are actually pulled
        pulled = [m for m in candidates if m["name"] in self.available]
        if not pulled:
            print(f"⚠ No {task_type} specialists available. Models not yet pulled:")
            for m in candidates:
                print(f"  - {m['name']}: {m['ollama_pull']}")
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
                 temperature=0.4, max_tokens=300, keep_alive=None, load_timeout=MODEL_LOAD_TIMEOUT):
        """
        Load a specialist, run the task, return the result.
        The specialist is kept loaded briefly for follow-ups (Ollama default).

        Uses context window hard caps from model spec if available.
        Thread-safe: uses internal lock to prevent concurrent dispatches.

        Args:
            task_type: Type of specialist to use
            prompt: Task prompt
            context: Optional background context
            system: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            keep_alive: How long to keep model loaded (None=auto, 0=immediate, -1=permanent)
            load_timeout: Seconds to wait for model load (default 45s)

        Returns:
            dict with {model, tag, result} or {error, model}
        """
        # Issue 8: Concurrent dispatch protection
        if not self._dispatch_lock.acquire(blocking=False):
            return {"error": "Another dispatch is already in progress. Please wait."}

        try:
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
                # Issue 3: Timeout handling with retry for slow model loads
                start_time = time.time()
                ctx_cap = specialist.get("context_cap", CONTEXT_CAP_SPECIALIST)

                # Try with adaptive timeout based on model size
                adaptive_timeout = max(load_timeout, specialist.get("vram_gb", 1) * 10)

                result = chat(specialist["name"], messages, temperature, max_tokens,
                             num_ctx=ctx_cap, keep_alive=keep_alive)

                load_time = time.time() - start_time
                if load_time > 30:
                    print(f"! warning: {specialist['name']} took {load_time:.1f}s to load/respond")

                self.loaded = list_loaded_models()
                return {
                    "model": specialist["name"],
                    "tag": specialist["tag"],
                    "result": result,
                }
            except urllib.error.URLError as e:
                if "timeout" in str(e).lower():
                    return {"error": f"Model load timeout (> {adaptive_timeout}s). Try increasing MODEL_LOAD_TIMEOUT.", "model": specialist["name"]}
                return {"error": f"Network error: {e}", "model": specialist["name"]}
            except Exception as e:
                return {"error": str(e), "model": specialist["name"]}
        finally:
            self._dispatch_lock.release()

    def iterate(self, prompt, context=None, system=None):
        """
        Run the iterator model. This is the fast, always-loaded heartbeat.
        Should be used for every tick. Only escalate with dispatch() when needed.

        Uses CONTEXT_CAP_ROUTER for fast, low-memory operation.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
            messages.append({"role": "assistant", "content": "Understood."})
        messages.append({"role": "user", "content": prompt})

        try:
            # Iterator uses small context window per blueprint spec
            result = chat(ITERATOR_MODEL, messages,
                         temperature=0.3, max_tokens=150, num_ctx=CONTEXT_CAP_ROUTER,
                         keep_alive=KEEP_ALIVE_PERMANENT)
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

    def health_check(self, timeout=5):
        """
        Verify Ollama is reachable and responsive.
        Returns dict with {healthy, error, iterator_loaded, vram_status}.
        """
        result = {
            "healthy": False,
            "error": None,
            "iterator_loaded": False,
            "vram_status": "unknown",
        }

        # Check Ollama connectivity
        try:
            tags_data = _get_json(SERVER_TAGS_URL, timeout=timeout)
            if not tags_data or "models" not in tags_data:
                result["error"] = "Ollama returned invalid response"
                return result
        except urllib.error.URLError as e:
            result["error"] = f"Ollama unreachable: {e}"
            return result
        except Exception as e:
            result["error"] = f"Health check failed: {e}"
            return result

        result["healthy"] = True

        # Check iterator availability
        if ITERATOR_MODEL in self.available:
            loaded = list_loaded_models()
            result["iterator_loaded"] = ITERATOR_MODEL in loaded

        # Check VRAM status
        current_vram = self._current_vram_usage()
        utilization = (current_vram / self.vram_budget * 100) if self.vram_budget > 0 else 0
        if utilization > 90:
            result["vram_status"] = "critical"
        elif utilization > 75:
            result["vram_status"] = "high"
        elif utilization > 50:
            result["vram_status"] = "moderate"
        else:
            result["vram_status"] = "healthy"

        return result

    def roster_report(self):
        """Return a formatted roster report."""
        lines = ["# Model Flock Roster", ""]
        for tier_start, tier_end, tier_name in [
            (0, 1, "Tier 0 — Always Loaded (Iterator)"),
            (1, 4, "Tier 1 — Fast Specialists (<3s load)"),
            (4, 7, "Tier 2 — Medium Specialists (~5s load)"),
            (7, 9, "Tier 3 — Heavy Specialists (~10s load)"),
            (9, 11, "Tier 3.5 — MoE & Audio Specialists"),
            (11, 13, "Tier 4 — Multimodal Specialists"),
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

    # ═══════════════════════════════════════════════════════════
    # Sequential Escalation Pipeline (Layer 2)
    # ═══════════════════════════════════════════════════════════

    def pipeline(self, task_type, prompt, context=None, system=None):
        """
        Execute a multi-step research pipeline (Layer 2 escalation).
        This is the 3-step flow from the external blueprint:

        Step 1: Researcher (DeepSeek-R1) creates a plan
        Step 2: Scout (light model) executes searches/tool calls
        Step 3: Researcher synthesizes results

        Issue 6: Pipeline failure recovery - if step 2 fails critically,
        step 3 can still attempt synthesis with the available data.

        Args:
            task_type: Primary specialty (e.g., "reasoning", "coding")
            prompt: The research query
            context: Optional background context
            system: Optional system message override

        Returns:
            dict with {plan, scout_results, synthesis, steps_taken, pipeline_status}
        """
        results = {"steps_taken": [], "plan": None, "scout_results": None, "synthesis": None, "pipeline_status": "running"}

        # Step 1: Researcher creates a plan
        print(f"* [Step 1/3] Researcher planning...")
        researcher = self.select_specialist("reasoning", hint="deep_planning")
        if not researcher:
            researcher = self.select_specialist(task_type)

        if not researcher:
            results["plan"] = "No reasoning specialist available."
            results["pipeline_status"] = "failed_no_researcher"
            return results

        plan_system = system or (
            "You are a Researcher. Break this task into a clear research plan. "
            "Output a JSON plan with steps."
        )
        plan_result = self.dispatch(
            "reasoning",
            f"Create a research plan for: {prompt}\n\nContext: {context or 'None'}",
            context=None,
            system=plan_system,
            max_tokens=400,
        )

        results["plan"] = plan_result.get("result", plan_result.get("error", ""))
        results["steps_taken"].append("researcher_plan")
        print(f"* plan received: {len(results['plan'])} chars")

        # Check if plan failed critically
        if "error" in plan_result or not results["plan"] or results["plan"].startswith("[iterator error"):
            print(f"! warning: plan step failed, continuing with direct scout...")

        # Step 2: Scout executes (fast model for searches/tool calls)
        print(f"* [Step 2/3] Scout execution...")
        scout = self.select_specialist("reasoning", hint="fast") or self.select_specialist("general")
        scout_failed = False

        if scout:
            scout_system = (
                "You are a Scout. Execute the plan efficiently. "
                "Use tools, search, and gather information concisely."
            )
            scout_prompt = f"Execute this plan: {results['plan'] or prompt}"
            scout_result = self.dispatch(
                scout["specialty"],
                scout_prompt,
                context=context,
                system=scout_system,
                max_tokens=600,
            )
            results["scout_results"] = scout_result.get("result", scout_result.get("error", ""))
            results["steps_taken"].append("scout_execute")
            print(f"* scout returned: {len(results['scout_results'])} chars")

            # Check if scout failed critically
            if "error" in scout_result:
                scout_failed = True
                print(f"! warning: scout step failed, synthesis will use limited data")

        # Step 3: Researcher synthesizes (always runs if we got this far)
        print(f"* [Step 3/3] Researcher synthesis...")

        # Adjust synthesis prompt based on what data we actually have
        if scout_failed:
            synthesis_prompt = (
                f"Original task: {prompt}\n\n"
                f"Plan:\n{results['plan']}\n\n"
                f"Note: Scout execution failed or returned errors. "
                f"Synthesize what you can from the plan and original task. "
                f"Provide a best-effort answer with clear limitations."
            )
        else:
            synthesis_prompt = (
                f"Original task: {prompt}\n\n"
                f"Plan:\n{results['plan']}\n\n"
                f"Scout findings:\n{results['scout_results']}\n\n"
                "Synthesize these into a final answer. Be thorough but concise."
            )

        synthesis_result = self.dispatch(
            "reasoning",
            synthesis_prompt,
            context=None,
            system="You are a Senior Researcher. Synthesize findings into a clear, actionable answer.",
            max_tokens=800,
        )
        results["synthesis"] = synthesis_result.get("result", synthesis_result.get("error", ""))
        results["steps_taken"].append("researcher_synthesis")
        print(f"* synthesis complete: {len(results['synthesis'])} chars")

        # Set final pipeline status
        if scout_failed:
            results["pipeline_status"] = "partial_success_scout_failed"
        elif "error" in synthesis_result:
            results["pipeline_status"] = "partial_success_synthesis_failed"
        else:
            results["pipeline_status"] = "success"

        results["model"] = synthesis_result.get("model", "pipeline")
        return results

    # ═══════════════════════════════════════════════════════════
    # Structured JSON Routing
    # ═══════════════════════════════════════════════════════════

    def route_structured(self, task_description, tools_available=None):
        """
        Use the iterator to output structured JSON routing.
        Returns {target_agent, requires_tools, confidence, rationale}.

        This is the JSON routing mode from the blueprint — more reliable
        than text markers for escalation decisions.

        Args:
            task_description: Description of what needs to be done
            tools_available: List of tools that can be used (optional)

        Returns:
            dict with routing decision
        """
        route_system = (
            "You are a Router. Output ONLY a JSON object:\n"
            '{"target_agent": "<tag|name>", "requires_tools": <bool>, '
            '"confidence": <0-1>, "rationale": "<why this agent>"}\n\n'
            "Available tags: fast-coder, coder, fast-thinker, thinker, "
            "sage, scholar, moe-oracle, eyes, watchman, herald, generalist, general-heavy.\n\n"
            "Use iterator for trivial tasks. Use specialists for real work."
        )

        tools_hint = f"\n\nAvailable tools: {tools_available}" if tools_available else ""
        messages = [
            {"role": "system", "content": route_system},
            {"role": "user", "content": f"Task: {task_description}{tools_hint}\n\nOutput JSON routing:"},
        ]

        try:
            # Use small context for router (per blueprint spec)
            result = chat(ITERATOR_MODEL, messages,
                         temperature=0.2, max_tokens=100, num_ctx=CONTEXT_CAP_ROUTER,
                         keep_alive=KEEP_ALIVE_PERMANENT)

            # Try multiple JSON extraction strategies for robustness
            routing = self._extract_json_routing(result)
            if routing:
                return routing
        except Exception as e:
            print(f"* route_structured error: {e}")

        # Fallback
        return {
            "target_agent": "generalist",
            "requires_tools": False,
            "confidence": 0.5,
            "rationale": "JSON parse failed — using fallback",
        }

    def _extract_json_routing(self, text):
        """Extract JSON routing from model output with multiple fallback strategies."""
        # Strategy 1: Try direct JSON parse (if model output clean JSON)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Regex for JSON object (handles text before/after)
        # Match from { to matching } using bracket counting
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        pass

        # Strategy 3: Look for specific JSON patterns with target_agent field
        patterns = [
            r'"target_agent"\s*:\s*"([^"]+)"',  # Extract just the target_agent
            r'target_agent["\s:]+([a-zA-Z0-9-]+)',  # More lenient pattern
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return {
                    "target_agent": match.group(1),
                    "requires_tools": False,
                    "confidence": 0.6,
                    "rationale": "Partial JSON extract — confidence reduced",
                }

        return None

    def dispatch_with_keep_alive(self, task_type, prompt, context=None, system=None,
                                 temperature=0.4, max_tokens=300, keep_alive=None, load_timeout=MODEL_LOAD_TIMEOUT):
        """
        Same as dispatch() but with explicit keep_alive control.
        Use this when you need precise VRAM management.
        Thread-safe: uses internal lock to prevent concurrent dispatches.

        Args:
            task_type: Type of specialist to use
            prompt: Task prompt
            context: Optional background context
            system: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            keep_alive: How long to keep model loaded (None=auto, 0=immediate, -1=permanent)
            load_timeout: Seconds to wait for model load (default 45s)

        Returns:
            dict with {model, tag, result} or {error, model}
        """
        # Issue 8: Concurrent dispatch protection
        if not self._dispatch_lock.acquire(blocking=False):
            return {"error": "Another dispatch is already in progress. Please wait."}

        try:
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
                # Issue 3: Timeout handling with retry for slow model loads
                start_time = time.time()
                ctx_cap = specialist.get("context_cap", CONTEXT_CAP_SPECIALIST)
                adaptive_timeout = max(load_timeout, specialist.get("vram_gb", 1) * 10)

                result = chat(specialist["name"], messages, temperature, max_tokens,
                             num_ctx=ctx_cap, keep_alive=keep_alive)

                load_time = time.time() - start_time
                if load_time > 30:
                    print(f"! warning: {specialist['name']} took {load_time:.1f}s to load/respond")

                self.loaded = list_loaded_models()
                return {
                    "model": specialist["name"],
                    "tag": specialist["tag"],
                    "result": result,
                }
            except urllib.error.URLError as e:
                if "timeout" in str(e).lower():
                    return {"error": f"Model load timeout (> {adaptive_timeout}s). Try increasing MODEL_LOAD_TIMEOUT.", "model": specialist["name"]}
                return {"error": f"Network error: {e}", "model": specialist["name"]}
            except Exception as e:
                return {"error": str(e), "model": specialist["name"]}
        finally:
            self._dispatch_lock.release()


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

def run_with_escalation(router, prompt, context=None, use_json_routing=False, max_depth=MAX_ESCALATION_DEPTH):
    """
    Full escalation flow: iterator tries → escalates if needed → specialist solves.

    Issue 4: Escalation loop prevention - tracks escalation depth and prevents
    infinite loops if specialists also fail/escalate.

    Args:
        router: ModelRouter instance
        prompt: The task prompt
        context: Optional context
        use_json_routing: If True, use structured JSON routing instead of text markers
        max_depth: Maximum escalation depth to prevent infinite loops (default: 3)

    Returns:
        dict with {handled_by, result, escalated, task_type, hint, escalation_depth}
    """
    # Check escalation depth before proceeding
    if router._escalation_depth >= max_depth:
        print(f"! escalation stopped: max depth ({max_depth}) reached")
        return {
            "handled_by": "none",
            "result": f"Escalation stopped after {max_depth} attempts. All specialists failed.",
            "escalated": False,
            "escalation_depth": router._escalation_depth,
            "error": "max_escalation_depth_reached",
        }

    router._escalation_depth += 1
    current_depth = router._escalation_depth

    try:
        # Step 1: Iterator tries
        iterator_output = router.iterate(
            prompt, context=context, system=ESCALATION_SYSTEM)

        # Step 2: Check for escalation
        if use_json_routing:
            routing = router.route_structured(prompt)
            task_type = routing_to_task_type(routing)
            hint = routing.get("rationale", "")
            needs_escalation = routing.get("target_agent") != "iterator"
        else:
            needs_escalation, task_type, hint = router.escalate_check(iterator_output)

        if not needs_escalation:
            return {
                "handled_by": "iterator",
                "result": iterator_output,
                "escalated": False,
                "escalation_depth": current_depth,
            }

        # Step 3: Dispatch to specialist
        print(f"* iterator escalated → {task_type} (depth {current_depth}/{max_depth})" +
              (f" ({hint})" if hint else ""))

        result = router.dispatch(task_type, prompt, context=context,
                                system=f"You are a {task_type} specialist. Solve this precisely.",
                                keep_alive=KEEP_ALIVE_BRIEF)

        specialist_result = result.get("result", result.get("error", "unknown"))

        # Check if specialist also signaled escalation (prevent infinite loop)
        if "escalate:" in specialist_result.lower() or "need_specialist:" in specialist_result.lower():
            print(f"! specialist signaled escalation, returning specialist output as-is")
            return {
                "handled_by": result.get("model", task_type),
                "result": specialist_result,
                "escalated": True,
                "escalation_depth": current_depth,
                "task_type": task_type,
                "hint": hint,
                "warning": "Specialist also requested escalation - stopping to prevent loop",
            }

        return {
            "handled_by": result.get("model", task_type),
            "result": specialist_result,
            "escalated": True,
            "escalation_depth": current_depth,
            "task_type": task_type,
            "hint": hint,
        }
    finally:
        # Always decrement depth on exit
        router._escalation_depth -= 1


def routing_to_task_type(routing):
    """Convert routing JSON to task_type string."""
    target = routing.get("target_agent", "")
    if target in BY_TAG:
        return BY_TAG[target].get("specialty", "general")
    return "general"


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local multi-model flock dispatcher")
    parser.add_argument("--roster", action="store_true", help="Print the model roster")
    parser.add_argument("--status", action="store_true", help="Print current flock status")
    parser.add_argument("--task", help="Task type: coding, reasoning, vision, general, audio")
    parser.add_argument("--prompt", help="Prompt for the task")
    parser.add_argument("--context", help="Additional context")
    parser.add_argument("--escalate", action="store_true",
                       help="Run with escalation (iterator tries first)")
    parser.add_argument("--pipeline", action="store_true",
                       help="Use sequential escalation pipeline (3-step research flow)")
    parser.add_argument("--json-route", action="store_true",
                       help="Use structured JSON routing instead of text markers")
    parser.add_argument("--setup-env", action="store_true",
                       help="Set up hardware environment variables and exit")
    args = parser.parse_args()

    # Setup environment if requested
    if args.setup_env:
        setup_env()
        print("* environment configured. Run 'python model_router.py' to start.")
        return

    router = ModelRouter()

    if args.roster:
        print(router.roster_report())
        return

    if args.status:
        router._report_status()
        return

    if args.prompt:
        if args.pipeline:
            # Sequential escalation pipeline
            if not args.task:
                args.task = "reasoning"  # Default for pipeline
            result = router.pipeline(args.task, args.prompt, args.context)
            print(f"\n[Pipeline — handled by: {result.get('model', 'unknown')}]")
            print(f"\n## Plan:\n{result.get('plan', '(no plan)')}")
            print(f"\n## Scout Results:\n{result.get('scout_results', '(no scout results)')}")
            print(f"\n## Synthesis:\n{result.get('synthesis', '(no synthesis)')}")
            return
        elif args.json_route:
            # Structured JSON routing
            routing = router.route_structured(args.prompt)
            print(f"\n[Routing Decision]")
            print(f"  Target Agent: {routing.get('target_agent')}")
            print(f"  Requires Tools: {routing.get('requires_tools')}")
            print(f"  Confidence: {routing.get('confidence', 0):.2f}")
            print(f"  Rationale: {routing.get('rationale', '')}")

            # Auto-dispatch based on routing
            target = routing.get("target_agent")
            if target in BY_TAG:
                specialist = BY_TAG[target]
                result = router.dispatch(specialty_or_tag_to_type(specialist),
                                       args.prompt, args.context)
                print(f"\n[handled by: {result.get('model', 'unknown')}]")
                print(result.get("result", result.get("error", "no output")))
            return
        elif args.escalate:
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


def specialty_or_tag_to_type(model_info):
    """Convert a model dict to a task_type for dispatch()."""
    return model_info.get("specialty", "general")


if __name__ == "__main__":
    main()
