# ConvFinQA Three-Agent Framework

*(Implementation in `src/predictors/multi_agent/` – June 2025)*

---

## 1  Overview
The predictor adopts the reflection framework proposed in *Enhancing Financial Question Answering with a Multi-Agent Reflection Framework* (arXiv 2410.21741). It replaces the legacy four-stage pipeline with three specialised CrewAI agents that interact iteratively until the answer is ratified.

```text
Expert → [draft answer]
        ↓                       ↘
Extraction Critic ──┐             ⇢  Aggregated verdict  →  accept / revise
Calculation Critic ─┘             ↗
        ↑ (if revise)  [critic feedback]
```

---

## 2  Agent Roles

| Agent | Role & Goal | Model / Temperature | Delegation / Tools |
|-------|-------------|---------------------|--------------------|
| **Financial Analysis Expert** | Performs **both** data extraction **and** calculation in one chain-of-thought. Outputs strict JSON:<br>`{"steps": [...], "answer": "…"}` | Configurable (`three_agent_config.expert_*`) | No delegation, no external tools (matches the paper). |
| **Data-Extraction Critic** | Verifies relevance, completeness and provenance of extracted numbers. Returns JSON with `"is_correct"`. | Configurable | None |
| **Calculation-Logic Critic** | Audits mathematical reasoning, units, sign handling, etc. Returns JSON with `"is_correct"`. | Configurable | None |

All settings live in `config/base.json` under `three_agent_config`, enabling model swaps without code changes.

---

## 3  Task Construction (`tasks.py`)
* **Expert prompt** combines:
  * Question, conversation history, formatted document (pre-text + table)
  * Detailed extraction **&** calculation guidelines
  * Iteration context – previous draft + critic feedback
* **Critic prompts** include original question, document context, expert JSON and a rubric; critics must respond in **exact JSON**.
* Literal braces are escaped (`{{ … }}`) to avoid Python f-string errors.

---

## 4  Orchestration (`orchestrator.py`)
1. **Iteration loop** (default `max_iterations = 2`):
   1. Crew with Expert → draft
   2. Crew with both Critics → feedback
   3. **Unanimous approval** needed (`AND` over all `is_correct` flags)
   4. If rejected, feedback is injected and the loop repeats
2. **Conversation memory** – each Q/A pair is appended so pronouns like "that value" resolve across turns.
3. **Answer extraction** – robust regex + JSON parser extracts `"answer"`.
4. **Post-processing** – optional scale normalisation, comma removal, % → decimal.
5. **Tracing** – when a `trace_dir` is supplied, every prompt/response is logged to JSONL for audit.

---

## 5  Design Decisions & Rationale

* **Single expert agent** – unified CoT has higher factual consistency than split roles.
* **Two distinct critics** – orthogonal checks; unanimous approval stops silent failure.
* **Tool-less reasoning** – mirrors the reference study and keeps dependencies light.
* **Strict JSON contracts** – allows automatic parsing and quick validity checks.
* **Config-driven models** – rapid A/B testing without touching code.
* **Fail-soft heuristics** – keyword fallback ensures resilience to malformed JSON during experimentation.

---

## 6  Known Limitations

1. Critics currently rely on language cues; they do **not** recompute the answer.
2. Only two iterations run by default; stubborn errors may persist.
3. Complex table layouts without explicit headers still challenge extraction.
4. CrewAI memory is disabled; expert depends on injected history blocks.

---

## 7  Suggested Enhancements

* Add automatic numeric replay inside critics to detect operand swaps or scale errors.
* Increase `max_iterations` adaptively until both critics approve or a hard limit is reached.
* Employ `extract_target_kpis()` to highlight candidate rows/columns, improving recall.
* Enable CrewAI memory or pass the previous `"steps"` array to shorten prompts and support progressive refinement.

---

## 8  Reference
* *Enhancing Financial Question Answering with a Multi-Agent Reflection Framework* – [arXiv 2410.21741](https://arxiv.org/pdf/2410.21741) 