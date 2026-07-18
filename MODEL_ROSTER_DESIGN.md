# MODEL_ROSTER_DESIGN.md

Multi-model routing layer for the autonomous agent loop.

## Overview

A small fast model (the *iterator*) runs continuously as router/dispatcher. When stumped, it escalates to specialist models loaded on-demand. A VRAM budget manager tracks what's loaded and unloads to make room.

Target hardware: laptop with 16-32GB RAM, RTX 4050/4060 (6-8GB VRAM). All models served via Ollama on localhost:11434.

---

## 1. Model Registry Format

```yaml
# models.yaml — the roster
models:
  - id: iterator
    name: "qwen2.5:0.5b"
    ollama_tag: "qwen2.5:0.5b"
    size_gb: 0.4
    vram_gb: 1.2
    specialty: "routing, fast iteration"
    capabilities:
      - route_request
      - parse_action
      - detect_stuck
    always_loaded: true
    context_window: 32768
    quantization: "q4_0"

  - id: coder
    name: "qwen2.5-coder:3b"
    ollama_tag: "qwen2.5-coder:3b"
    size_gb: 1.9
    vram_gb: 4.5
    specialty: "code generation, debugging, refactoring"
    capabilities:
      - generate_code
      - refactor
      - debug_stacktrace
      - write_tests
    always_loaded: false
    context_window: 32768
    quantization: "q4_0"

  # ... more models
```

**Field meanings:**
- `id`: internal identifier, used in escalation protocol
- `name`: human-readable name
- `ollama_tag`: exact tag for Ollama API (`model` field)
- `size_gb`: disk footprint (for load time estimates)
- `vram_gb`: GPU memory when loaded (quantized, inference-ready)
- `specialty`: one-liner description
- `capabilities`: list of action types this model can handle
- `always_loaded`: if true, never unload (iterator only)
- `context_window`: max tokens the model supports
- `quantization`: Ollama quant format (q4_0, q5_0, q8_0, f16)

---

## 2. Escalation Protocol

### 2.1 Iterator Signal Format

The iterator model outputs a special action when it needs help:

```json
{
  "action": "escalate",
  "reason": "<why I'm stuck>",
  "task_type": "<specialty needed>",
  "context_summary": "<brief context for specialist>",
  "urgency": "normal|low|high"
}
```

**Example escalations:**

```json
// Need deep reasoning
{
  "action": "escalate",
  "reason": "Task requires multi-step logical deduction beyond my capacity",
  "task_type": "reasoning",
  "context_summary": "User asks to prove algorithm correctness invariant",
  "urgency": "normal"
}

// Need vision
{
  "action": "escalate",
  "reason": "Request involves image analysis",
  "task_type": "vision",
  "context_summary": "User uploaded screenshot of error UI",
  "urgency": "high"
}

// Need creative writing
{
  "action": "escalate",
  "reason": "User requests creative content generation",
  "task_type": "creative",
  "context_summary": "Blog post about agent architecture",
  "urgency": "low"
}
```

### 2.2 Detection Loop

The iterator also has internal stuck detection:

```python
# In autonomous.py, after each iterator turn
stuck_signals = [
    "I don't understand",
    "I'm unable",
    "This requires",
    "I cannot",
    "Not enough information",
]

if any(sig in text.lower() for sig in stuck_signals):
    # Auto-escalate without explicit action
    escalate(task_type="general", reason="iterator stuck")
```

### 2.3 Escalation Flow

```
┌─────────────────┐
│   Iterator      │
│  (qwen2.5:0.5b) │
└────────┬────────┘
         │
         │ escalate?
         ▼
┌─────────────────────────────────┐
│   Task Type Classifier           │
│   (regex + iterator summary)     │
└────────┬─────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Specialist Selector           │
│   (match task_type → model)     │
└────────┬─────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   VRAM Budget Manager           │
│   (load/unload models)           │
└────────┬─────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Specialist Model               │
│   (execute with handoff)         │
└────────┬─────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Response Integration           │
│   (merge back into stream.md)    │
└─────────────────────────────────┘
```

---

## 3. VRAM Budget Manager

### 3.1 Budget Configuration

```yaml
# config.yaml extension
vram:
  total_budget_gb: 7.5        # Leave headroom on 8GB card
  reserve_gb: 0.5             # Safety margin
  iterator_reserve_gb: 1.5    # Always keep iterator loaded
  load_threshold_gb: 6.0      # Start unloading above this
  unload_policy: "lru"        # or "fifo", "priority"

# Unload priority (lower = unload first)
unload_priority:
  - creative: 1
  - search: 2
  - verification: 3
  - reasoning: 4
  - coder: 5
  - vision: 6
```

### 3.2 Load State Tracking

```python
# Runtime state
class VRAMManager:
    def __init__(self, budget_gb):
        self.budget = budget_gb
        self.loaded = {}  # {model_id: loaded_since_timestamp}
        self.vram_used = 0.0

    def can_load(self, model_id, vram_gb):
        """Check if we have room, unload if needed."""
        available = self.budget - self.vram_used
        if available >= vram_gb:
            return True

        # Need to unload something
        return self._make_room(vram_gb)

    def load(self, model_id, vram_gb):
        """Call Ollama to preload model."""
        if model_id in self.loaded:
            return  # Already loaded

        if not self.can_load(model_id, vram_gb):
            raise VRAMFullError(f"Cannot load {model_id}: no room")

        # Warm load via Ollama API
        resp = _post_json(SERVER_CHAT_URL, {
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "keep_alive": -1,  # Keep loaded
        })

        self.loaded[model_id] = time.time()
        self.vram_used += vram_gb
        self._log_load(model_id)

    def unload(self, model_id, vram_gb):
        """Tell Ollama to unload."""
        # Ollama unloads via /api/generate with keep_alive=0
        _post_json(f"{SERVER_BASE}/api/generate", {
            "model": model_id,
            "prompt": "",
            "keep_alive": 0,
        })

        if model_id in self.loaded:
            del self.loaded[model_id]
            self.vram_used -= vram_gb
            self._log_unload(model_id)

    def _make_room(self, needed_gb):
        """Unload models by LRU + priority until we have room."""
        candidates = sorted(
            self.loaded.items(),
            key=lambda x: (x[1], UNLOAD_PRIORITY.get(x[0], 99))
        )

        freed = 0.0
        for model_id, loaded_at in candidates:
            if model_id == "iterator":
                continue  # Never unload iterator

            vram_gb = MODEL_VRAM.get(model_id, 0)
            self.unload(model_id, vram_gb)
            freed += vram_gb
            if self.budget - (self.vram_used - freed) >= needed_gb:
                return True

        return False
```

### 3.3 API Endpoints

Ollama provides model management:

```python
# List loaded models (inference-time proxy)
GET /api/tags  # returns all, but doesn't show "loaded" state

# Force unload via zero-duration keep_alive
POST /api/generate
{
  "model": "qwen2.5-coder:3b",
  "prompt": "",
  "keep_alive": 0  # Unload after response
}

# Keep loaded (warm)
POST /api/chat
{
  "model": "...",
  "keep_alive": -1  # Never unload
}
```

---

## 4. Specialist Selection Algorithm

### 4.1 Task Type Taxonomy

```yaml
task_types:
  coding:
    keywords: ["write", "implement", "refactor", "debug", "fix", "function", "class"]
    default_model: "coder"

  reasoning:
    keywords: ["prove", "analyze", "deduce", "why", "how does", "explain logic"]
    default_model: "reasoner"

  vision:
    keywords: ["image", "screenshot", "photo", "diagram", "visual"]
    default_model: "vision"

  search:
    keywords: ["search", "find", "lookup", "what is", "who is"]
    default_model: "search"

  creative:
    keywords: ["write", "draft", "blog", "story", "creative", "narrative"]
    default_model: "creative"

  verification:
    keywords: ["verify", "check", "validate", "test", "review"]
    default_model: "verifier"

  fast:
    keywords: ["quick", "simple", "short", "summary"]
    default_model: "iterator"  # Handle ourselves

  math:
    keywords: ["calculate", "compute", "solve", "equation", "formula"]
    default_model: "math"

  embedding:
    keywords: ["similar", "cluster", "vector", "semantic"]
    default_model: "embedding"

  translation:
    keywords: ["translate", "in spanish", "in french", "convert to"]
    default_model: "translator"
```

### 4.2 Selection Logic

```python
def select_specialist(escalation_request, models_registry):
    """
    Select the best model for an escalation request.

    Args:
        escalation_request: dict with task_type, reason, context_summary, urgency
        models_registry: ModelRegistry instance

    Returns:
        model_id to load and call
    """
    task_type = escalation_request.get("task_type", "general")
    urgency = escalation_request.get("urgency", "normal")

    # Direct task_type → model mapping
    task_to_model = {
        "coding": "coder",
        "reasoning": "reasoner",
        "vision": "vision",
        "search": "search",
        "creative": "creative",
        "verification": "verifier",
        "math": "math",
        "embedding": "embedding",
        "translation": "translator",
    }

    model_id = task_to_model.get(task_type, "general")

    # Fallback to keyword match if unknown type
    if model_id == "general":
        context = escalation_request.get("context_summary", "")
        model_id = _classify_by_keywords(context, models_registry)

    # Respect VRAM budget — if specialist won't fit, use general
    model = models_registry.get(model_id)
    if not vram_manager.can_load(model_id, model.vram_gb):
        logger.warning(f"VRAM full, falling back to general model for {task_type}")
        model_id = "general"

    return model_id

def _classify_by_keywords(text, registry):
    """Fallback keyword classifier."""
    scores = collections.Counter()
    for model in registry.all():
        for keyword in model.keywords:
            if keyword.lower() in text.lower():
                scores[model.id] += 1
    return scores.most_common(1)[0][0] if scores else "general"
```

---

## 5. Context Handoff

### 5.1 What Context Goes to Specialist?

**Always include:**
1. Current goal (from goals.md) — active goal title, context, status
2. Working memory tail (last 10 entries) — recent context/observations
3. The original request that triggered escalation
4. Iterator's summary (why it escalated, what it tried)

**Conditionally include:**
- Relevant file excerpts (if task is file-related)
- Tool results (if escalation follows a tool call)
- Stream.md tail (to avoid repeating completed steps)

**Never include:**
- Full workspace.md (too big) — send tail + goal summary
- Full history (token budget) — send last N lines
- Binary data

### 5.2 Handoff Prompt Format

```python
def build_specialist_prompt(escalation_request, active_goal, working_memory, iterator_attempt):
    """
    Build prompt for specialist model.

    The specialist receives:
    1. Why it was escalated (iterator's confession)
    2. What it needs to do (clear task)
    3. Relevant context (bounded)
    4. Output format (JSON action)
    """
    task_type = escalation_request.get("task_type", "general")
    reason = escalation_request.get("reason", "")
    iterator_summary = escalation_request.get("context_summary", "")

    specialist_prompts = {
        "coding": "You are a Code Specialist. Write clean, working code.",
        "reasoning": "You are a Reasoning Specialist. Think step-by-step.",
        "vision": "You are a Vision Specialist. Describe and analyze images.",
        "creative": "You are a Creative Specialist. Write engaging content.",
        # ...
    }

    system = (
        f"{specialist_prompts.get(task_type, 'You are a Generalist.')}\n\n"
        f"The iterator model escalated this request because: {reason}\n\n"
        f"Iterator's attempt: {iterator_attempt}\n\n"
        "Provide the best solution. Output as JSON action:"
    )

    user = f"=== Task ===\n{escalation_request.get('context_summary', '')}\n\n"

    if active_goal:
        user += f"=== Active Goal ===\n{active_goal['title']}\n{active_goal['context'][:500]}\n\n"

    mem_tail = load_working_memory(limit=5)
    if mem_tail:
        user += "=== Recent Working Memory ===\n"
        for m in mem_tail:
            user += f"- [{m['type']}] {m['content'][:100]}\n"
        user += "\n"

    user += 'Output: {"action": "...", ...}'

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
```

### 5.3 Response Integration

```python
def integrate_specialist_response(specialist_action, original_goal):
    """
    Merge specialist's output back into the main loop.

    The specialist's JSON action is treated as if the iterator produced it:
    -写入 stream.md if it's a suggestion
    - Execute if it's a tool call
    - Update working memory with the collaboration
    """
    action_type = specialist_action.get("action", "suggest")

    if action_type == "suggest":
        text = specialist_action.get("text", "")
        append_text(STREAM_FILE, text)
        append_working_memory("collaboration",
            f"Escalated to specialist; result: {text[:100]}")
        return f"[specialist] → {text}"

    elif action_type == "tool":
        # Execute tool via sandbox
        result = sandbox.execute(specialist_action["command"])
        append_working_memory("specialist_tool",
            f"Tool via specialist: {result['status']}")
        return f"[specialist] tool: {result['status']}"

    elif action_type == "complete_goal":
        gid = specialist_action.get("goal_id", original_goal["id"])
        update_goal_status(gid, "complete", specialist_action.get("summary"))
        return f"[specialist] completed goal {gid}"

    return f"[specialist] {action_type}"
```

---

## 6. Concrete Model Roster

### 6.1 The Twelve Models

All are real Ollama models as of 2026-07. Sizes and VRAM are approximate for q4_0 quantization.

| ID | Model Name | Ollama Tag | Size GB | VRAM GB | Specialty | Keywords |
|----|------------|------------|---------|---------|-----------|----------|
| iterator | Qwen2.5 0.5B | `qwen2.5:0.5b` | 0.4 | 1.2 | routing, fast iteration | route, parse, detect |
| coder | Qwen2.5-Coder 3B | `qwen2.5-coder:3b` | 1.9 | 4.5 | code generation, debugging | code, function, bug |
| coder-light | Qwen2.5-Coder 1.5B | `qwen2.5-coder:1.5b` | 1.0 | 2.8 | quick code fixes | quick fix, simple |
| reasoner | Qwen2.5 7B | `qwen2.5:7b` | 4.3 | 7.5 | logical reasoning | prove, analyze, logic |
| general | Phi-3 3.8B | `phi3:3.8b` | 2.3 | 5.0 | general purpose fallback | help, explain, general |
| vision | LLaVA 7B | `llava:7b` | 4.0 | 7.0 | image understanding | image, screenshot, photo |
| creative | Gemma 2 9B | `gemma2:9b` | 5.5 | 8.0 | creative writing | write, blog, story |
| verifier | Qwen2.5 14B | `qwen2.5:14b` | 8.5 | 12.0 | verification, review | verify, check, review |
| search | DeepSeek-R1 1.5B | `deepseek-r1:1.5b` | 1.2 | 3.0 | search, retrieval | search, find, lookup |
| math | Qwen2.5-Math 7B | `qwen2.5-math:7b` | 4.5 | 7.8 | mathematical reasoning | calculate, solve, prove |
| embedding | Nomic Embed 1B | `nomic-embed-text:1b` | 0.7 | 1.5 | vector embeddings | similar, cluster, embed |
| translator | NLLB 3.3B | `nllb-3.3b:distilled` | 2.1 | 4.8 | translation | translate, in spanish |

### 6.2 Model Details

#### iterator: qwen2.5:0.5b
- **Role:** Always-loaded router
- **Strengths:** Fast, low VRAM, sufficient for pattern matching
- **Weaknesses:** Limited reasoning, no world knowledge
- **Context:** 32K tokens
- **Quant:** q4_0

#### coder: qwen2.5-coder:3b
- **Role:** Primary coding specialist
- **Strengths:** Code generation, debugging, refactoring
- **Weaknesses:** Weak on non-code tasks
- **Context:** 32K tokens
- **Quant:** q4_0

#### coder-light: qwen2.5-coder:1.5b
- **Role:** Quick code fixes when VRAM tight
- **Strengths:** Fast, smaller VRAM footprint
- **Weaknesses:** Less capable than 3B
- **Context:** 32K tokens
- **Quant:** q4_0

#### reasoner: qwen2.5:7b
- **Role:** Logical reasoning, multi-step deduction
- **Strengths:** Strong reasoning, good at explanations
- **Weaknesses:** Heavy VRAM use
- **Context:** 32K tokens
- **Quant:** q4_0

#### general: phi3:3.8b
- **Role:** Fallback for unspecified tasks
- **Strengths:** Balanced generalist
- **Weaknesses:** Not exceptional at anything
- **Context:** 12K tokens
- **Quant:** q4_0

#### vision: llava:7b
- **Role:** Image understanding
- **Strengths:** Multi-modal, can describe images
- **Weaknesses:** Heavy, slower
- **Context:** 4K image + 2K text
- **Quant:** q4_0

#### creative: gemma2:9b
- **Role:** Creative writing
- **Strengths:** Natural language generation
- **Weaknesses:** Very heavy, may not fit
- **Context:** 8K tokens
- **Quant:** q4_0 (tight fit)

#### verifier: qwen2.5:14b
- **Role:** Verification, code review (swap-in only)
- **Strengths:** High quality analysis
- **Weaknesses:** May not fit on 8GB VRAM at all (CPU fallback)
- **Context:** 32K tokens
- **Quant:** q4_0 (needs 12GB VRAM, uses CPU fallback)

#### search: deepseek-r1:1.5b
- **Role:** Search-like tasks, information retrieval
- **Strengths:** Fast, trained for search-like tasks
- **Weaknesses:** Not a real search engine (no internet)
- **Context:** 32K tokens
- **Quant:** q4_0

#### math: qwen2.5-math:7b
- **Role:** Mathematical reasoning
- **Strengths:** Math problems, proofs, calculations
- **Weaknesses:** Heavy VRAM
- **Context:** 32K tokens
- **Quant:** q4_0

#### embedding: nomic-embed-text:1b
- **Role:** Vector embeddings for similarity
- **Strengths:** Fast, small, good embeddings
- **Weaknesses:** Not generative
- **Context:** 8K tokens
- **Quant:** f16

#### translator: nllb-3.3b:distilled
- **Role:** Translation between languages
- **Strengths:** Multilingual (200+ languages)
- **Weaknesses:** Niche use case
- **Context:** 512 tokens (sentence-level)
- **Quant:** q4_0

### 6.3 VRAM Budget Scenarios

**8GB VRAM (RTX 4050/4060):**
- Always loaded: iterator (1.2 GB)
- Available: ~6 GB for specialists
- Typical concurrent load: 1 specialist at a time
- Can fit: coder (4.5) OR reasoner (7.5 tight) OR vision (7.0 tight)
- Strategy: Keep coder loaded by default, unload for vision/reasoner

**16GB VRAM (RTX 4070+):**
- Always loaded: iterator (1.2 GB)
- Available: ~14 GB
- Can fit: coder + reasoner + search simultaneously
- Strategy: Pre-load coder and reasoner, load others on-demand

**6GB VRAM (older laptops):**
- Always loaded: iterator (1.2 GB)
- Available: ~4.5 GB
- Can fit: coder-light (2.8) only
- Strategy: Swap aggressively, use CPU fallback for heavier models

---

## 7. Integration with autonomous.py

### 7.1 New Constants

```python
# Add to autonomous.py
ITERATOR_MODEL = "qwen2.5:0.5b"           # The router
DEFAULT_SPECIALIST = "phi3:3.8b"         # Fallback
MODELS_REGISTRY_FILE = "models.yaml"
VRAM_BUDGET_GB = 7.5                    # Configurable
```

### 7.2 New Functions

```python
class ModelRoster:
    """Multi-model routing layer."""

    def __init__(self, models_file, vram_budget):
        self.models = self._load_registry(models_file)
        self.vram = VRAMManager(vram_budget)
        self.vram.load("iterator", self.models["iterator"].vram_gb)

    def route(self, user_request, active_goal, working_memory):
        """Main entry: route request through iterator or escalate."""
        # Try iterator first
        iterator_prompt = build_iterator_prompt(user_request, active_goal)
        iterator_response = ask_model(ITERATOR_MODEL, iterator_prompt)

        action = parse_action(iterator_response)
        if action.get("action") == "escalate":
            return self._escalate(action, active_goal, working_memory)

        return execute_action(action)

    def _escalate(self, escalation, active_goal, working_memory):
        """Escalate to specialist."""
        model_id = select_specialist(escalation, self.models)

        # Load specialist if needed
        model = self.models[model_id]
        if model_id not in self.vram.loaded:
            self.vram.load(model_id, model.vram_gb)

        # Build specialist prompt and call
        specialist_prompt = build_specialist_prompt(
            escalation, active_goal, working_memory)
        specialist_response = ask_model(model.ollama_tag, specialist_prompt)

        return integrate_specialist_response(specialist_response, active_goal)
```

### 7.3 Main Loop Integration

```python
# In main(), replace the existing model call:
roster = ModelRoster(MODELS_REGISTRY_FILE, VRAM_BUDGET_GB)

# Inside the tick loop:
# OLD:
# messages = build_autonomous_prompt(rules, active_goal, config, notes)
# text = ask_model(messages, temp=0.5)
# action = parse_action(text)

# NEW:
action = roster.route(
    {"goal": active_goal, "workspace": workspace_tail},
    active_goal,
    load_working_memory(active_goal["id"])
)
```

---

## 8. Failure Modes and Fallbacks

### 8.1 VRAM Full
- **Symptom:** vram_manager.can_load() returns False
- **Fallback:** Use iterator (degraded quality) or CPU fallback (slow)
- **User notification:** tick.md shows "! VRAM full, using degraded mode"

### 8.2 Model Not Pulled
- **Symptom:** Ollama returns "model not found"
- **Fallback:** Auto-suggest pull command in tick.md, skip specialist turn
- **Recovery:** `!pull <model>` command to pull on-demand

### 8.3 Specialist Timeout
- **Symptom:** Request exceeds REQUEST_TIMEOUT
- **Fallback:** Return to iterator, mark task as "blocked"
- **Recovery:** Retry on next tick with larger timeout

### 8.4 All Specialists Unavailable
- **Symptom:** No models can be loaded
- **Fallback:** Iterator alone with degraded quality
- **User notification:** tick.md warning

---

## 9. Deployment Checklist

- [ ] Pull all 12 models via `ollama pull`
- [ ] Create `models.yaml` with registry
- [ ] Update `config.yaml` with VRAM budget
- [ ] Test escalation with each task type
- [ ] Verify VRAM manager unload/load
- [ ] Measure tick latency with each specialist
- [ ] Document CPU fallback behavior

---

## 10. Open Questions

1. **Cold start:** Should we pull models on first use or require pre-pull?
   - *Decision:* Pre-pull via script, lazy-load at runtime

2. **Priority preloading:** Should we preload coder on startup?
   - *Decision:* Yes, if VRAM allows

3. **Parallel specialists:** Could we run two specialists for complex tasks?
   - *Decision:* No, keep simple for v1. Single specialist per escalation.

4. **Learning from escalations:** Should we track which task types trigger which specialists?
   - *Decision:* Yes, log to audit.jsonl for pattern analysis.

5. **User override:** Can user force a specific specialist?
   - *Decision:* Add `!use <model>` command for v2.
