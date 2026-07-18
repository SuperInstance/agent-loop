# Emergence by Arrangement

### How constrained agent loops produce collaborative capability beyond their model weights

**The agent-loop project — White Paper v1.0 — July 2026**

*Status: working paper. Sections 1–5 report mechanisms and observed results from a running reference implementation. Section 7 is a proposed study program, clearly marked as such; no claim in this paper rests on data that does not exist.*

---

## Abstract

The prevailing bet in applied AI is that capability comes from scale: larger models, longer contexts, more tools. This paper documents a complementary bet, tested in a small working system: that for a well-defined class of human–machine collaboration, **capability emerges from arrangement** — from the protocol that constrains a model, not merely from the model itself. We present six mechanisms — write-domain separation, human-move clocking, dual-rate memory consolidation, deterministic guardrails, progressive capability exposure, and medium-as-protocol interoperability — each of which produces a system-level behavior that was never programmed into any component. We report a live experimental review in which a 1.5-billion-parameter open model, arranged inside these mechanisms, exhibited stable turn-taking, acquired persistent personal style rules from single corrections, and recovered from its own failure modes without human repair. We describe what the system cannot do, and we close with a fully specified, not-yet-run study program for testing generalizability across languages and cultures — offered as an open invitation to replicate. The reference implementation is 279 lines of dependency-free Python; every claim herein is accompanied by the means to check it.

---

## 1. Introduction: two kinds of bet

The last fifteen years of machine learning rewarded one bet above all: make the model bigger and it will surprise you. The bet has paid so reliably that its inverse is rarely stated plainly: *a small model, well arranged, can also surprise you* — and the surprise will be cheaper, more legible, and easier to keep.

This paper is about arrangement. The word is chosen carefully. An arrangement is not an algorithm; it is the set of constraints within which components — human, model, files, tools — act on one another. Change the arrangement and the same components exhibit different behaviors, sometimes qualitatively different ones. When a behavior appears at the level of the system and in none of its parts, we call it emergent. The thesis of this paper is modest and testable:

> *For sustained, personal, mixed-initiative collaboration between a human and a language model, a substantial fraction of the observable capability is a property of the arrangement, and can be obtained with deliberately small models.*

The claim is deliberately narrow. It does not concern benchmark reasoning, agentic tool chains, or autonomous operation. It concerns the everyday loop of working *with* a model on shared text over time — the loop most developers actually live in — and it is evaluated against three requirements that loop implies: the system must not fight the human for control of the shared artifact; it must learn the human's corrections durably, not sessionally; and it must fail in ways that are visible and bounded.

We built the smallest system we could that satisfies these requirements, and then we tried to break it. The mechanisms below are what survived; the failures are in §5, with the evidence.

## 2. Context: the medium is already a protocol

The idea that shared computational state needs ownership semantics is old. Blackboard architectures partitioned a common workspace among cooperating knowledge sources, with a control shell deciding who spoke next [1]. Linda's tuple spaces gave concurrent processes a common medium with strict, minimal operations [2]. What is new in 2026 is the cast: one of the principals is a language model, and the shared medium is not a data structure but **ordinary text files** — the same files the human already edits.

Three concurrent developments frame the moment. First, the local-first movement articulated a discipline for software that keeps data on the user's machine, in open formats, under the user's control [3]; our constraints descend directly from theirs. Second, the industry's agentic turn produced tool-calling loops (e.g., ReAct [7]) whose failures are dominantly *procedural* — loops that ramble, repeat, or act without consent — rather than intellectual. Third, and most quietly, developers began writing instructions *for* agents into their repositories: the `AGENTS.md` convention, introduced in 2025 and now present in tens of thousands of repositories [10], and the `llms.txt` proposal for agent-legible documentation [9]. Early empirical work treats these files as a new class of software artifact — persistent, versioned context that measurably changes agent behavior [11]. The file, in other words, has already become the interface between humans and agents. This paper takes that observation to its conclusion: **if files are the interface, then a small set of ownership and pacing rules over files is a complete collaboration protocol** — and protocols, unlike models, can be verified by reading them.

## 3. The mechanisms

Each mechanism is presented the same way: the principle, the mechanism itself, the emergent effect (what the system does that no component was told to do), and where the evidence sits in §5.

### M1 — Write-Domain Separation

> *Give every principal a file it owns, and forbid it to write anywhere else.*

The human owns `workspace.md`. The agent owns `stream.md`. Optional tools own their own files. Reads are unrestricted; writes are owned absolutely. There is no locking, no scheduler, no merge logic — the partition itself is the concurrency control, a direct descendant of blackboard partitioning [1] with the control shell replaced by social convention enforced in code.

**Emergent effect: calm turn-taking.** The agent cannot talk over the human, cannot fight an editor over a buffer, and cannot mistake its own output for the human's intent — the three failure modes that dominate naive shared-document agent loops. None of this was programmed as behavior; it follows from the ownership rule. (Evidence: §5, E4.)

### M2 — Human-Move Clocking

> *The human's edit is the system clock.*

The agent takes exactly one turn per human move: a saved edit to `workspace.md`, plus a quiet interval, yields at most one suggestion. A `!step` command exists for autonomous nudges, but the default clock is the human. This inverts the usual agent loop, in which the model decides when to act and the human supervises a runaway process from behind.

**Emergent effect: pacing as a property.** The system's tempo tracks the human's attention without any attention modeling. When the human thinks, the system waits; when the human moves, the system responds once. In Weiser and Brown's terms, the technology becomes calm not through design of any interface element but through the structure of its clock [4]. (Evidence: §5, E4.)

### M3 — Dual-Rate Memory Consolidation

> *Every correction is stored twice: once for the next prompt, once for the next model.*

When the human types `!fix rejected => preferred`, two things happen. The rule is injected into the system prompt immediately (fast store: in-context learning, effective on the very next turn), and an Alpaca-shaped row is appended to `preferences.jsonl` (slow store: a training dataset, accumulating for later fine-tuning with parameter-efficient methods such as LoRA [5]). The design mirrors the complementary learning systems of biological memory: a fast hippocampal trace for immediate use, a slow cortical trace for permanence. The fast store has a deliberate capacity limit (five rules); the slow store is append-only and unlimited.

**Emergent effect: personalization that compounds.** Single corrections visibly change behavior within one turn and survive process restarts, because the slow store re-seeds the fast store at boot. Over weeks, the slow store becomes a personal dataset whose existence changes what the *same base model* means to its user — a property no prompt engineering alone produces. (Evidence: §5, E3.)

### M4 — Deterministic Guardrails

> *Never ask a stochastic component to enforce an invariant that a deterministic one can.*

Small models ignore instructions reliably and charmingly. In testing, a system-prompt injunction "never repeat yourself" failed one hundred percent of the time. The guardrail therefore lives in code: suggestions are normalized and compared against history; a repeat triggers one retry at higher temperature; a second repeat voids the turn silently. The same principle governs output parsing (fences are recognized before any stripping — an ordering bug here was found by the model, not by us), output length (capped by token count, not by plea), and turn frequency (M2, enforced by state machine, not by asking the model to wait).

**Emergent effect: reliability beyond obedience.** The system's worst-case behavior is bounded by code that never sees the model's output until after it has been constrained. The observable difference is between a collaborator who occasionally goes quiet and one who occasionally floods — only the former is tolerable in a shared workspace. (Evidence: §5, E1, E2, E6.)

### M5 — Progressive Capability Exposure

> *Capabilities are a ladder of trust, not a manifest of features.*

The core loop executes nothing the model produces. Its only powers are to read three files and write one. Optional capabilities — a local completion engine, a test harness, a future tool runner — are separate processes, installed and run only when wanted, each bound by the same file-ownership discipline (M1). A tool runner, if ever built, inherits hard requirements as a condition of acceptance: allowlisted verbs, argv execution, an approval gate, no secrets in its environment.

**Emergent effect: security as architecture.** There is no policy document that could be misread, because the dangerous possibilities do not exist in the running process. The audit surface for the entire system is a short file a developer can read in five minutes — and, in an agent-mediated world, a file a *scout* can verify in seconds. (Evidence: §5, E5 discussion; verification commands in the repository.)

### M6 — Medium-as-Protocol Interoperability

> *When the medium is the API, anything that can write text can join the collaboration.*

The system's module interface is the file set itself. Any process, in any language, from any vendor, that respects ownership and turn discipline is a compatible module — no SDK, no schema negotiation, no permissioning layer. The Director pattern demonstrates this: a frontier agent (Claude Code, Codex, or a script) writes a `quest.md`; the small local model reads it on its next turn and steers accordingly. Hierarchy emerges from file presence, not from wiring.

**Emergent effect: vendor-neutral agent-to-agent collaboration.** The much-discussed A2A future arrives without new infrastructure, in the same idiom the ecosystem is already converging on for agent-facing documentation [9][10]. The protocol's entire specification fits on one page (see the repository's `docs/PROTOCOL.md`), because the medium carries almost all of it. (Evidence: §5, E4 note; §6.)

## 4. The reference system

The mechanisms are instantiated in `agent-loop`, a reference implementation whose constraints were chosen to make the arrangement auditable: a single 279-line Python file using only the standard library; a local model server (Ollama) as the sole network endpoint; four files as the entire state. A second process (`extras/reflex.py`, 87 lines) plays the role of a completion engine for fully local setups; a third (`extras/conductor_driver.py`) is a scripted human used for testing. Nothing else is required, and nothing else runs.

The numbers matter to the thesis. The system occupies roughly 15 MB of memory and performs one file-stat per second when idle. The model it was tested against is 1.5 billion parameters — small enough to run on a laptop CPU in seconds per turn. The claim is not that this system competes with frontier coding agents; it is that a system *this small* can exhibit the collaborative behaviors in §3 at all.

## 5. Experimental review

All results below come from recorded sessions; transcripts are preserved in the repository (`session-logs/`) with a full report (`SESSION_REPORT.md`). The model was Qwen2.5-Coder-1.5B-Instruct (Q4_K_M) [8] served by llama.cpp on a 2-core CPU. The human was played by the scripted conductor, executing a fixed sequence of moves: seed a goal, steer with `!note`, accept a completion, teach a `!fix`, force `!step`, and steer again. The goal given to the system was reflexive: *design the v2 sandboxed tool layer for this agent loop itself.*

**E1 — Baseline repetition collapse.** With the guardrail (M4) absent, the loop produced the identical micro-step — *"Add a function to check if a command is allowed."* — six times in succession, despite a system-prompt instruction forbidding repetition. This is the expected behavior of a small model in a near-static context, and it is the motivating failure for M4: prompt-level injunctions are not enforceable.

**E2 — Guardrail effectiveness.** With M4 active, repeat detection fired three times in the same-length session; each firing triggered the hot retry, and each retry produced a distinct, on-task advance (from allowlist definition to approval-gate sketch to gate implementation). Zero duplicate lines were written to the shared stream. Worst case became a skipped turn — invisible to the human — rather than a flood.

**E3 — Style acquisition and persistence.** Immediately after the single correction `!fix ALLOWED = ['ls', 'pytest', 'git status'] => class Tool(Enum): LS='ls'; PYTEST='pytest'`, the agent's suggestions switched to enum-style declarations consistent with the taught rule. The rule was logged to `preferences.jsonl` and, after a full process restart, was re-loaded and continued to shape output — the fast/slow store behavior of M3 operating end-to-end.

**E4 — Turn-taking stability.** Across a six-move session, the one-suggestion-per-move discipline held with no stalls, no self-triggering, and no editor conflicts (M1, M2). The acceptance gesture — the human copying a completion into their own file — correctly registered as a human move and advanced the collaboration, a behavior that was never explicitly programmed.

**E5 — Reflex quality.** The completion engine's expansion of the "human approval gate" intent produced structurally sound boilerplate (an `ApprovalStatus` enum and an `approve_tool` function) — adequate for a 1.5B model and, notably, the beginning of the very tool-layer design the session was tasked to ideate.

**E6 — Parser robustness.** A fenced model reply was mis-parsed into the one-word suggestion `python` (backticks stripped before fence detection), demonstrating that M4 applies to *our own* deterministic code as much as to the model: the fix was to check fences before stripping, and a regression test now covers it.

**Validation of the stripped core.** After the system was reduced to its zero-dependency form, the rewritten transport and logic paths were re-verified against a deterministic fake model server: 8/8 checks passed, including two assertions observed server-side (quest intake and rule injection).

### Threats to validity

We state these before the reader has to. The evidence comes from **one model family at one size**, on English-language content, with a **scripted** human, over sessions of minutes. No comparison against a larger model in the same arrangement is included. The results demonstrate that the emergent behaviors *occur* under the arrangement; they do not establish how strongly model size, language, task domain, or human variability moderate them. Section 7 exists because of this paragraph.

## 6. Discussion

**When does arrangement beat scale?** The honest answer from this work: when the bottleneck is procedural rather than intellectual. The failures we observed — repetition, timing violations, ownership conflicts, forgotten corrections — are not reasoning failures and are not cured by reasoning ability. They are arrangement failures, and they are cured by arrangement, cheaply and verifiably. Conversely, where the bottleneck *is* intellectual — novel algorithm design, deep debugging, long-horizon planning — the 1.5B model's ceiling was plainly visible, and no protocol lifts it. The practical reading is a division of labor, not a rivalry: let large models do what requires scale, and let arrangement make small models trustworthy company for everything else.

**The A2A implication.** The ecosystem is standardizing on files as the human-to-agent interface [9][10][11]. M6 suggests the same idiom extends to agent-to-agent: agents of different vendors and sizes, collaborating through owned files with no shared runtime. The properties that make this attractive — auditability, vendor neutrality, zero deployment — are precisely the properties the local-first literature argued users deserve from their tools [3], now applied to the agents themselves.

**On honesty as a mechanism.** We include, deliberately, one more design choice that is not in the code: the system's documents state their limitations next to their features, and every claim ships with a verification command. In an environment where the first reader of a repository is increasingly an agent evaluating it for a human, verifiability is not a virtue signal; it is an interoperability feature [9][10].
