#!/usr/bin/env python3
"""
lora_registry.py — LoRA Adapter Registry for Model Flock

Instead of loading 14 separate models (~35GB), use 2 base models + 12 LoRA adapters.
Each LoRA adapter is ~50MB and swaps on top of a base model for specialty tasks.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  BASE MODELS (always or frequently loaded)           │
  │  ┌─────────────────┐  ┌──────────────────────┐     │
  │  │ qwen2.5:0.5b    │  │ qwen2.5-coder:3b     │     │
  │  │ (iterator base) │  │ (specialist base)    │     │
  │  └─────────────────┘  └──────────────────────┘     │
  │           │                     │                  │
  │           │     + LoRA adapter (~50MB each)         │
  │           ▼                     ▼                  │
  │  ┌─────────────────────────────────────────────┐   │
  │  │  ADAPTERS (swap on top of base)               │   │
  │  │  • general-write, general-summarize           │   │
  │  │  • thinker-plan, thinker-analyze, thinker-math│   │
  │  │  • coder-review, coder-refactor, coder-debug  │   │
  │  │  • coder-architect                             │   │
  │  └─────────────────────────────────────────────┘   │
  │                                                      │
  │  Other bases (loaded on demand):                    │
  │  • llava:7b → vision-describe, vision-ocr          │
  │  • minicpm-o:4.5 → audio-transcribe                 │
  └─────────────────────────────────────────────────────┘

Usage:
    from lora_registry import load_adapter, swap_adapter, estimate_vram

    # Load an adapter (creates Ollama model)
    load_adapter("coder-review")

    # Swap adapters atomically
    swap_adapter("coder-refactor", old_adapter="coder-review")

    # Estimate VRAM usage
    vram = estimate_vram("coder-review")  # Returns 2.6 GB

Zero dependencies. Stdlib only. Compatible with model_router.py.
"""

import json
import os
import sys
import threading
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SERVER_CREATE_URL = f"{OLLAMA_HOST}/api/create"
SERVER_DELETE_URL = f"{OLLAMA_HOST}/api/delete"
SERVER_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
SERVER_PS_URL = f"{OLLAMA_HOST}/api/ps"
REQUEST_TIMEOUT = 180

# LoRA adapter overhead in VRAM (estimated)
LORA_ADAPTER_VRAM_GB = 0.1  # ~100MB per adapter

# Base model VRAM requirements (from model_router.py)
BASE_MODEL_VRAM = {
    "qwen2.5:0.5b": 0.5,
    "qwen2.5:3b": 2.5,
    "qwen2.5-coder:3b": 2.5,
    "llava:7b": 5.0,
    "minicpm-o:4.5": 3.5,
}

# File path where LoRA adapters are stored
LORA_STORAGE_PATH = os.environ.get(
    "LORA_STORAGE_PATH",
    os.path.join(os.path.dirname(__file__), "lora_adapters")
)

# ═══════════════════════════════════════════════════════════════
# HTTP Helpers (stdlib, matches model_router.py pattern)
# ═══════════════════════════════════════════════════════════════

def _get_json(url: str, timeout: int = 5) -> dict:
    """Fetch JSON from URL."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"! error fetching {url}: {e}")
        return {}


def _post_json(url: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    """Post JSON to URL."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def _delete_json(url: str, payload: dict, timeout: int = 5) -> dict:
    """Send DELETE request with JSON body."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="DELETE"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Adapter Registry
# ═══════════════════════════════════════════════════════════════

ADAPTER_REGISTRY: List[dict] = [
    # ═══════════════════════════════════════════════════════════
    # Coder Adapters (base: qwen2.5-coder:3b)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "coder-review",
        "base_model": "qwen2.5-coder:3b",
        "specialty": "coding",
        "file_path": f"{LORA_STORAGE_PATH}/coder-review.safetensors",
        "description": "Code review specialist. Finds bugs, security issues, and anti-patterns.",
        "strengths": ["code_review", "security_audit", "bug_detection", "best_practices"],
        "vram_gb": BASE_MODEL_VRAM["qwen2.5-coder:3b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "qwen2.5-coder:3b-lora-review",
    },
    {
        "name": "coder-refactor",
        "base_model": "qwen2.5-coder:3b",
        "specialty": "coding",
        "file_path": f"{LORA_STORAGE_PATH}/coder-refactor.safetensors",
        "description": "Code refactor specialist. Improves structure, readability, and maintainability.",
        "strengths": ["refactoring", "code_cleanup", "modernization", "idiomatic_code"],
        "vram_gb": BASE_MODEL_VRAM["qwen2.5-coder:3b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "qwen2.5-coder:3b-lora-refactor",
    },
    {
        "name": "coder-debug",
        "base_model": "qwen2.5-coder:3b",
        "specialty": "coding",
        "file_path": f"{LORA_STORAGE_PATH}/coder-debug.safetensors",
        "description": "Debugging specialist. Analyzes errors, traces failures, suggests fixes.",
        "strengths": ["debugging", "error_analysis", "root_cause", "fix_suggestions"],
        "vram_gb": BASE_MODEL_VRAM["qwen2.5-coder:3b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "qwen2.5-coder:3b-lora-debug",
    },
    {
        "name": "coder-architect",
        "base_model": "qwen2.5-coder:3b",
        "specialty": "coding",
        "file_path": f"{LORA_STORAGE_PATH}/coder-architect.safetensors",
        "description": "Software architecture specialist. Design patterns, system design, scalability.",
        "strengths": ["architecture", "system_design", "design_patterns", "scalability"],
        "vram_gb": BASE_MODEL_VRAM["qwen2.5-coder:3b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "qwen2.5-coder:3b-lora-architect",
    },

    # ═══════════════════════════════════════════════════════════
    # Thinker Adapters (base: qwen2.5:3b)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "thinker-plan",
        "base_model": "qwen2.5:3b",
        "specialty": "reasoning",
        "file_path": f"{LORA_STORAGE_PATH}/thinker-plan.safetensors",
        "description": "Planning specialist. Breaks down complex goals into actionable steps.",
        "strengths": ["planning", "task_breakdown", "project_management", "roadmapping"],
        "vram_gb": BASE_MODEL_VRAM["qwen2.5:3b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "qwen2.5:3b-lora-plan",
    },
    {
        "name": "thinker-analyze",
        "base_model": "qwen2.5:3b",
        "specialty": "reasoning",
        "file_path": f"{LORA_STORAGE_PATH}/thinker-analyze.safetensors",
        "description": "Analysis specialist. Deep analysis of problems, data, and situations.",
        "strengths": ["analysis", "critical_thinking", "data_interpretation", "synthesis"],
        "vram_gb": BASE_MODEL_VRAM["qwen2.5:3b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "qwen2.5:3b-lora-analyze",
    },
    {
        "name": "thinker-math",
        "base_model": "qwen2.5:3b",
        "specialty": "reasoning",
        "file_path": f"{LORA_STORAGE_PATH}/thinker-math.safetensors",
        "description": "Mathematics specialist. Calculations, proofs, formulas, equations.",
        "strengths": ["mathematics", "calculations", "proofs", "statistical_analysis"],
        "vram_gb": BASE_MODEL_VRAM["qwen2.5:3b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "qwen2.5:3b-lora-math",
    },

    # ═══════════════════════════════════════════════════════════
    # General Adapters (base: qwen2.5:0.5b - iterator base)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "general-write",
        "base_model": "qwen2.5:0.5b",
        "specialty": "general",
        "file_path": f"{LORA_STORAGE_PATH}/general-write.safetensors",
        "description": "Writing specialist. Drafts, edits, and improves text content.",
        "strengths": ["writing", "drafting", "editing", "creative_writing"],
        "vram_gb": BASE_MODEL_VRAM["qwen2.5:0.5b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "qwen2.5:0.5b-lora-write",
    },
    {
        "name": "general-summarize",
        "base_model": "qwen2.5:0.5b",
        "specialty": "general",
        "file_path": f"{LORA_STORAGE_PATH}/general-summarize.safetensors",
        "description": "Summarization specialist. Condenses long content into concise summaries.",
        "strengths": ["summarization", "condensation", "abstraction", "key_points"],
        "vram_gb": BASE_MODEL_VRAM["qwen2.5:0.5b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "qwen2.5:0.5b-lora-summarize",
    },

    # ═══════════════════════════════════════════════════════════
    # Vision Adapters (base: llava:7b - different base)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "vision-describe",
        "base_model": "llava:7b",
        "specialty": "vision",
        "file_path": f"{LORA_STORAGE_PATH}/vision-describe.safetensors",
        "description": "Image description specialist. Detailed visual descriptions and explanations.",
        "strengths": ["image_description", "visual_explanation", "scene_understanding"],
        "vram_gb": BASE_MODEL_VRAM["llava:7b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "llava:7b-lora-describe",
    },
    {
        "name": "vision-ocr",
        "base_model": "llava:7b",
        "specialty": "vision",
        "file_path": f"{LORA_STORAGE_PATH}/vision-ocr.safetensors",
        "description": "OCR specialist. Text extraction from images, screenshots, and documents.",
        "strengths": ["ocr", "text_extraction", "document_digitization", "code_from_screenshots"],
        "vram_gb": BASE_MODEL_VRAM["llava:7b"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "llava:7b-lora-ocr",
    },

    # ═══════════════════════════════════════════════════════════
    # Audio Adapters (base: minicpm-o:4.5 - different base)
    # ═══════════════════════════════════════════════════════════
    {
        "name": "audio-transcribe",
        "base_model": "minicpm-o:4.5",
        "specialty": "audio",
        "file_path": f"{LORA_STORAGE_PATH}/audio-transcribe.safetensors",
        "description": "Speech-to-text specialist. Transcribes audio with high accuracy.",
        "strengths": ["transcription", "speech_recognition", "dictation", "audio_processing"],
        "vram_gb": BASE_MODEL_VRAM["minicpm-o:4.5"] + LORA_ADAPTER_VRAM_GB,
        "ollama_name": "minicpm-o:4.5-lora-transcribe",
    },
]

# Build lookup indexes
ADAPTER_BY_NAME: Dict[str, dict] = {a["name"]: a for a in ADAPTER_REGISTRY}
ADAPTER_BY_SPECIALTY: Dict[str, List[dict]] = {}
for adapter in ADAPTER_REGISTRY:
    ADAPTER_BY_SPECIALTY.setdefault(adapter["specialty"], []).append(adapter)

ADAPTER_BY_BASE: Dict[str, List[dict]] = {}
for adapter in ADAPTER_REGISTRY:
    ADAPTER_BY_BASE.setdefault(adapter["base_model"], []).append(adapter)

# Track loaded adapters (runtime state)
_loaded_adapters: Dict[str, str] = {}  # {adapter_name: ollama_model_name}
_load_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════
# Adapter Management Functions
# ═══════════════════════════════════════════════════════════════

def get_adapter(adapter_name: str) -> Optional[dict]:
    """
    Get adapter configuration by name.

    Args:
        adapter_name: Name of the adapter (e.g., "coder-review")

    Returns:
        Adapter dict or None if not found
    """
    return ADAPTER_BY_NAME.get(adapter_name)


def list_adapters(specialty: Optional[str] = None) -> List[dict]:
    """
    List all adapters, optionally filtered by specialty.

    Args:
        specialty: Filter by specialty (coding, reasoning, general, vision, audio)

    Returns:
        List of adapter dicts
    """
    if specialty:
        return ADAPTER_BY_SPECIALTY.get(specialty, []).copy()
    return ADAPTER_REGISTRY.copy()


def list_adapters_by_base(base_model: str) -> List[dict]:
    """
    List all adapters for a specific base model.

    Args:
        base_model: Base model name (e.g., "qwen2.5-coder:3b")

    Returns:
        List of adapter dicts
    """
    return ADAPTER_BY_BASE.get(base_model, []).copy()


def create_modelfile(adapter_name: str) -> Optional[str]:
    """
    Generate an Ollama Modelfile that combines base model + LoRA adapter.

    The Modelfile format:
        FROM <base_model>
        ADAPTER <file_path>
        PARAMETER temperature 0.4
        PARAMETER num_ctx 4096

    Args:
        adapter_name: Name of the adapter

    Returns:
        Modelfile content as string, or None if adapter not found
    """
    adapter = get_adapter(adapter_name)
    if not adapter:
        print(f"! adapter '{adapter_name}' not found")
        return None

    # Check if adapter file exists (warn if not)
    file_path = adapter["file_path"]
    if not os.path.exists(file_path):
        print(f"! warning: adapter file not found: {file_path}")
        print(f"  This is expected for theoretical adapters. "
              "Create the .safetensors file to use this adapter.")

    # Build Modelfile
    lines = [
        f"# Modelfile for {adapter['name']} adapter",
        f"# Base: {adapter['base_model']}",
        f"# Description: {adapter['description']}",
        f"",
        f"FROM {adapter['base_model']}",
        f"ADAPTER {file_path}",
        f"",
        f"# Parameters optimized for {adapter['specialty']} tasks",
        f"PARAMETER temperature 0.4",
        f"PARAMETER num_ctx 4096",
        f"PARAMETER num_predict 512",
        f"",
        f"# System prompt",
        f'SYSTEM """You are a {adapter["specialty"]} specialist.',
        f'{adapter["description"]}',
        f"Focus on: {', '.join(adapter['strengths'][:3])}.",
        f'"""',
    ]

    return "\n".join(lines)


def load_adapter(adapter_name: str, verify: bool = True) -> Tuple[bool, str]:
    """
    Load a LoRA adapter by creating the Ollama model from the Modelfile.

    This calls Ollama's /api/create endpoint with the Modelfile content.
    The resulting model is named: <base>-lora-<suffix>

    Args:
        adapter_name: Name of the adapter to load
        verify: If True, verify the model was created successfully

    Returns:
        Tuple of (success, message_or_model_name)
    """
    adapter = get_adapter(adapter_name)
    if not adapter:
        return False, f"Adapter '{adapter_name}' not found"

    # Generate Modelfile
    modelfile = create_modelfile(adapter_name)
    if not modelfile:
        return False, f"Failed to generate Modelfile for '{adapter_name}'"

    # Prepare payload for Ollama /api/create
    payload = {
        "name": adapter["ollama_name"],
        "modelfile": modelfile,
        "stream": False,
    }

    print(f"* creating model '{adapter['ollama_name']}' from adapter '{adapter_name}'...")

    # Call Ollama API
    response = _post_json(SERVER_CREATE_URL, payload, timeout=300)

    if "error" in response:
        return False, f"Failed to create model: {response['error']}"

    # Track loaded adapter
    with _load_lock:
        _loaded_adapters[adapter_name] = adapter["ollama_name"]

    # Verify if requested
    if verify:
        available = _get_json(SERVER_TAGS_URL)
        if adapter["ollama_name"] not in available.get("models", []):
            return False, f"Model created but not found in tags"

    return True, adapter["ollama_name"]


def unload_adapter(adapter_name: str) -> Tuple[bool, str]:
    """
    Unload a LoRA adapter by deleting the Ollama model.

    Args:
        adapter_name: Name of the adapter to unload

    Returns:
        Tuple of (success, message)
    """
    adapter = get_adapter(adapter_name)
    if not adapter:
        return False, f"Adapter '{adapter_name}' not found"

    if adapter_name not in _loaded_adapters:
        return False, f"Adapter '{adapter_name}' is not loaded"

    ollama_name = _loaded_adapters[adapter_name]
    payload = {"name": ollama_name}

    print(f"* deleting model '{ollama_name}'...")

    response = _delete_json(SERVER_DELETE_URL, payload, timeout=60)

    if "error" in response:
        return False, f"Failed to delete model: {response['error']}"

    # Remove from tracking
    with _load_lock:
        del _loaded_adapters[adapter_name]

    return True, f"Unloaded adapter '{adapter_name}' (deleted {ollama_name})"


def list_loaded_adapters() -> Dict[str, str]:
    """
    List which adapters are currently loaded.

    Returns:
        Dict mapping adapter_name → ollama_model_name
    """
    with _load_lock:
        return _loaded_adapters.copy()


def list_loaded_ollama_models() -> Dict[str, dict]:
    """
    List currently loaded Ollama models (from /api/ps).

    Returns:
        Dict mapping model_name → model_info
    """
    data = _get_json(SERVER_PS_URL)
    return {m["name"]: m for m in data.get("models", [])}


def swap_adapter(new_adapter: str, old_adapter: Optional[str] = None,
                keep_old: bool = False) -> Tuple[bool, str]:
    """
    Atomically swap adapters: unload old, load new.

    If old_adapter is None, this just loads the new adapter.
    If keep_old is True, the old adapter is kept loaded.

    Args:
        new_adapter: Name of the adapter to load
        old_adapter: Name of the adapter to unload (optional)
        keep_old: If True, don't unload the old adapter

    Returns:
        Tuple of (success, message)
    """
    # Step 1: Load new adapter
    success, result = load_adapter(new_adapter)
    if not success:
        return False, f"Failed to load '{new_adapter}': {result}"

    # Step 2: Unload old adapter if specified
    if old_adapter and not keep_old:
        unload_success, unload_msg = unload_adapter(old_adapter)
        if not unload_success:
            # Rollback: unload the new adapter
            unload_adapter(new_adapter)
            return False, f"Swap failed: {unload_msg}"

    return True, f"Swapped to '{new_adapter}' (model: {result})"


def estimate_vram(adapter_name: str, include_base: bool = True) -> Optional[float]:
    """
    Estimate VRAM usage for an adapter.

    Args:
        adapter_name: Name of the adapter
        include_base: If True, include base model VRAM (default)

    Returns:
        VRAM in GB, or None if adapter not found
    """
    adapter = get_adapter(adapter_name)
    if not adapter:
        return None

    if include_base:
        return adapter["vram_gb"]
    else:
        return LORA_ADAPTER_VRAM_GB


def estimate_total_vram(adapter_names: List[str]) -> float:
    """
    Estimate total VRAM usage for multiple adapters.

    This accounts for shared base models - if multiple adapters
    share the same base, the base VRAM is only counted once.

    Args:
        adapter_names: List of adapter names

    Returns:
        Total VRAM in GB
    """
    seen_bases = set()
    total = 0.0

    for name in adapter_names:
        adapter = get_adapter(name)
        if not adapter:
            continue

        base = adapter["base_model"]
        if base not in seen_bases:
            total += BASE_MODEL_VRAM.get(base, 0)
            seen_bases.add(base)

        total += LORA_ADAPTER_VRAM_GB

    return total


def find_best_adapter(task_type: str, hint: Optional[str] = None) -> Optional[dict]:
    """
    Find the best adapter for a given task type.

    Args:
        task_type: Type of task (coding, reasoning, general, vision, audio)
        hint: Optional hint for specific strength

    Returns:
        Best adapter dict or None
    """
    candidates = ADAPTER_BY_SPECIALTY.get(task_type, [])
    if not candidates:
        return None

    # If hint provided, try to find a match
    if hint:
        hint_lower = hint.lower()
        for adapter in candidates:
            if any(h in s for s in adapter["strengths"] for h in [hint_lower]):
                return adapter

    # Return first candidate
    return candidates[0]


def adapter_status_report() -> str:
    """
    Generate a status report of all adapters.

    Returns:
        Formatted report string
    """
    lines = [
        "# LoRA Adapter Registry Status",
        "",
        f"Total adapters: {len(ADAPTER_REGISTRY)}",
        f"Loaded adapters: {len(_loaded_adapters)}",
        "",
        "## By Specialty",
        "",
    ]

    for specialty in ["coding", "reasoning", "general", "vision", "audio"]:
        adapters = ADAPTER_BY_SPECIALTY.get(specialty, [])
        if not adapters:
            continue

        lines.append(f"### {specialty.capitalize()}")
        lines.append("")

        for adapter in adapters:
            loaded = "✓ loaded" if adapter["name"] in _loaded_adapters else "○ not loaded"
            lines.append(f"#### {adapter['name']} — {loaded}")
            lines.append(f"- Base: `{adapter['base_model']}`")
            lines.append(f"- VRAM: {adapter['vram_gb']:.1f} GB")
            lines.append(f"- {adapter['description']}")
            lines.append(f"- Strengths: {', '.join(adapter['strengths'])}")
            lines.append("")

    # Add storage path info
    lines.append("## Storage")
    lines.append(f"Adapter files path: `{LORA_STORAGE_PATH}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Adapters are theoretical until .safetensors files are created")
    lines.append("- Use `create_modelfile(name)` to generate Ollama Modelfiles")
    lines.append("- Use `load_adapter(name)` to create the Ollama model")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LoRA Adapter Registry for Model Flock"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all adapters"
    )
    parser.add_argument(
        "--list-by-base",
        metavar="BASE",
        help="List adapters for a specific base model"
    )
    parser.add_argument(
        "--list-specialty",
        metavar="SPECIALTY",
        help="List adapters by specialty (coding, reasoning, general, vision, audio)"
    )
    parser.add_argument(
        "--load",
        metavar="ADAPTER",
        help="Load an adapter (creates Ollama model)"
    )
    parser.add_argument(
        "--unload",
        metavar="ADAPTER",
        help="Unload an adapter (deletes Ollama model)"
    )
    parser.add_argument(
        "--swap",
        nargs=2,
        metavar=("NEW", "OLD"),
        help="Swap adapters: unload OLD, load NEW"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show adapter status report"
    )
    parser.add_argument(
        "--modelfile",
        metavar="ADAPTER",
        help="Generate and print Modelfile for an adapter"
    )
    parser.add_argument(
        "--vram",
        metavar="ADAPTER",
        help="Estimate VRAM usage for an adapter"
    )
    parser.add_argument(
        "--best",
        nargs=2,
        metavar=("TYPE", "HINT"),
        help="Find best adapter for task type and hint"
    )

    args = parser.parse_args()

    if args.list:
        print("## All Adapters")
        print("")
        for adapter in ADAPTER_REGISTRY:
            print(f"### {adapter['name']}")
            print(f"  Base: {adapter['base_model']}")
            print(f"  Specialty: {adapter['specialty']}")
            print(f"  VRAM: {adapter['vram_gb']:.1f} GB")
            print(f"  {adapter['description']}")
            print("")

    elif args.list_by_base:
        adapters = list_adapters_by_base(args.list_by_base)
        print(f"## Adapters for base '{args.list_by_base}'")
        print("")
        for adapter in adapters:
            print(f"- {adapter['name']}: {adapter['description']}")
        print("")

    elif args.list_specialty:
        adapters = list_adapters(specialty=args.list_specialty.lower())
        print(f"## {args.list_specialty.capitalize()} Adapters")
        print("")
        for adapter in adapters:
            print(f"- {adapter['name']}: {adapter['description']}")
        print("")

    elif args.load:
        success, msg = load_adapter(args.load)
        if success:
            print(f"✓ {msg}")
        else:
            print(f"✗ {msg}")
            sys.exit(1)

    elif args.unload:
        success, msg = unload_adapter(args.unload)
        if success:
            print(f"✓ {msg}")
        else:
            print(f"✗ {msg}")
            sys.exit(1)

    elif args.swap:
        new_adapter, old_adapter = args.swap
        success, msg = swap_adapter(new_adapter, old_adapter)
        if success:
            print(f"✓ {msg}")
        else:
            print(f"✗ {msg}")
            sys.exit(1)

    elif args.status:
        print(adapter_status_report())

    elif args.modelfile:
        modelfile = create_modelfile(args.modelfile)
        if modelfile:
            print(modelfile)
        else:
            print(f"✗ Failed to generate Modelfile for '{args.modelfile}'")
            sys.exit(1)

    elif args.vram:
        vram = estimate_vram(args.vram)
        if vram is not None:
            print(f"{args.vram}: {vram:.1f} GB")
        else:
            print(f"✗ Adapter '{args.vram}' not found")
            sys.exit(1)

    elif args.best:
        task_type, hint = args.best
        adapter = find_best_adapter(task_type.lower(), hint)
        if adapter:
            print(f"Best adapter for {task_type} (hint: {hint}):")
            print(f"  Name: {adapter['name']}")
            print(f"  Base: {adapter['base_model']}")
            print(f"  VRAM: {adapter['vram_gb']:.1f} GB")
            print(f"  {adapter['description']}")
        else:
            print(f"✗ No adapter found for task type '{task_type}'")
            sys.exit(1)

    else:
        # Default: show status
        parser.print_help()
        print("")
        print(adapter_status_report())


if __name__ == "__main__":
    main()
