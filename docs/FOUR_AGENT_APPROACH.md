# ConvFinQA Four-Agent Framework (Archived)

*(Implementation in `src/predictors/multi_agent/archive/four_agent_approach/` – 2024)*

---

## 1  Overview
Prior to adopting the three-agent reflection framework we experimented with an intuitive, modular design that splits the financial QA workflow across **four** independent CrewAI agents:

```text
Supervisor → Extractor → Calculator → Validator → Final answer
```

No academic paper informed this layout; it was motivated by traditional software-engineering separation-of-concerns: identify → compute → verify.  Although conceptually clear, the system proved brittle in practice and has since been archived.

---

## 2  Agent Roles

| Agent | Role & Goal | Model / Temperature | Delegation / Tools |
|-------|-------------|---------------------|--------------------|
| **Supervisor** | Breaks a conversational query into subtasks, maintains dialogue context, and orchestrates other agents. | Configurable (`crewai.supervisor_*`) | None – pure reasoning |
| **Extractor** | Locates the exact numerical values in pre-text and tables, resolves pronouns such as "that year". Emits JSON `{candidates: [...], notes: ...}` | Configurable | None (tools were planned but disabled) |
| **Calculator** | Builds a minimal ConvFinQA DSL program (`add`, `subtract`, `divide`, `multiply`) using extractor output and conversation memory. | Configurable | None |
| **Validator** | Recalculates the DSL, checks logical consistency, assigns a confidence score, and decides if the answer is deliverable. | Configurable | None |

All parameters reside in `config/base.json` under the `crewai` section.

---

## 3  Task Construction (`tasks.py`)
* **Extraction task**
  * Provides intelligent KPI-matching hints using `financial_matcher`.
  * Requires exact JSON with numeric candidate list.
* **Calculation task**
  * Consumes extractor JSON; must output a valid DSL expression covering percentage-change, ratios, conversational references (`that`, `it`, etc.).
* **Validation task**
  * Re-runs the DSL (via helper functions) and flags mismatches or implausible magnitudes; can request rework by returning `is_valid: false`.
* Each task template is large and highly structured; braces are escaped to avoid f-string errors.

---

## 4  Orchestration (`orchestrator.py`)
1. Supervisor decomposes and dispatches work.
2. Extractor runs first ‑› Calculator ‑› Validator, **sequentially** (`Process.sequential`).
3. If the validator returns `false`, the orchestrator currently *logs* the issue but does **not** iterate, so the pipeline ends with an error.
4. Conversation history is maintained but only the **supervisor** has global visibility; downstream agents do not see previous turns unless the supervisor includes them in the task description.

---

## 5  Design Rationale (Intuition)
* *Modularity* – mirrors classic ETL pipelines: extract → transform → load.
* *Explicit DSL* – calculator produces deterministic, auditable code that can be unit-tested.
* *Single-responsibility* agents are easier to prompt and monitor individually.

---

## 6  Observed Issues & Performance
| Symptom | Root Cause |
|---------|------------|
| Extraction misses rows / wrong metric | KPI hints unreliable; no second-pass critic to catch error. |
| Calculator builds invalid / non-parsable DSL | Lack of immediate feedback loop; validator only runs last. |
| Validator flags error but orchestrator ends run | No iterative refinement implemented; one-shot pipeline. |
| JSON formatting errors crash tasks | Vast prompt templates + missing brace escapes (now patched in three-agent). |

In internal benchmarks this led to **≤ 45 % accuracy** on a small sample and frequent runtime exceptions, motivating the migration to the three-agent reflection design.

---

## 7  Lessons Learnt
1. Early decomposition without feedback loops allows errors to propagate unchecked.
2. Tool-less extractor struggles with complex tables; critics or retrieval helpers are essential.
3. A validator at the *end* cannot rescue earlier stages without the ability to trigger revisions.
4. Strict DSL is powerful but must be paired with automated execution during validation to provide concrete feedback.

---

## 8  Status
The four-agent framework is **deprecated**.  Code is retained for historical reference under `src/predictors/multi_agent/archive/four_agent_approach/` but is not executed by current benchmarks. 