# ConvFinQA Six-Agent Framework

*(Draft implementation plan – July 2025)*

---

## 1  Motivation

Benchmarking shows our current three-agent reflection system hovers around **40-50 % EM** on a representative slice of ConvFinQA. Error tracing indicates two root causes:

1. **Extraction drift** – the Expert sometimes selects wrong rows, wrong fiscal years, or mis-scales values.
2. **Cognitive overload** – combining extraction *and* calculation reasoning in one prompt amplifies hallucinations.

To isolate failure modes we decompose the pipeline into **specialised, lightweight agents** while preserving the iterative reflection loop that drove gains in prior research.

---

## 2  High-Level Architecture

```mermaid
flowchart TD
    subgraph Tier 0
        Mngr["Conversation<br/>Manager"]
    end
    subgraph Tier 1
        Ext["Data-Extraction<br/>Specialist"]
        Calc["Fin-Calc<br/>Reasoner"]
    end
    subgraph Tier 2
        EC["Extraction<br/>Critic"]
        CC["Calculation<br/>Critic"]
    end
    subgraph Tier 3
        Synth["Answer<br/>Synthesiser"]
    end

    Mngr -->|run_pipeline| Ext --> Calc --> Synth
    Calc --> EC
    Calc --> CC
    EC --> Synth
    CC --> Synth
    Synth --|revise| Calc
```

• **Conversation Manager** short-circuits trivial follow-ups via cache and kicks off the heavy pipeline only when needed.
• **Specialist + Reasoner** cleanly separate *finding* numbers from *using* them.
• **Twin Critics** replicate the reflection gains from the 2024 paper, each focusing on a single dimension.
• **Synthesiser** merges outputs, applies deterministic post-processing, and returns a conversational reply.

---

## 3  Agent Catalogue

| Tier | Agent ID | Primary Goal | Model (default) | Temp. | Delegation | Tools |
|------|----------|--------------|-----------------|-------|------------|-------|
| 0 | `manager` | Router + cache-hit detection | gpt-3.5-turbo-0125 | 0.2 | no | – |
| 1 | `extractor` | Locate exact cells, normalise units/sign | gpt-4o-mini | 0.35 | no | – |
| 1 | `reasoner` | Generate CoT + executable DSL, call calculator | gpt-4o | 0.50 | no | `CalculatorTool` |
| 2 | `extraction_critic` | Audit `extractor` JSON | gpt-3.5-turbo-0125 | 0.2 | no | – |
| 2 | `calculation_critic` | Audit `reasoner` JSON | gpt-3.5-turbo-0125 | 0.2 | no | – |
| 3 | `synthesiser` | Decide *final* vs *revise*, craft user-facing answer | gpt-3.5-turbo-0125 | 0.3 | no | – |

All settings are surfaced in `config/base.json → six_agent_config` so we can A/B models without code edits.

---

## 4  Prompt & Schema Contracts

### 4.1  Extractor → Specialist JSON
```jsonc
{
  "extractions": [
    {"row": "Total Revenue", "col": "2022", "raw": "60.94", "unit": "million", "scale": 1e6},
    …
  ]
}
```

### 4.2  Reasoner → Calculation JSON
```jsonc
{
  "steps": ["Extract …", "Calculate …"],
  "dsl": "divide(84.88, 100)",          // optional when calculation is trivial
  "answer": "0.8488"
}
```

### 4.3  Critics → Verdict JSON
```jsonc
{ "is_correct": true, "issues": ["…"], "suggested_fix": "…" }
```

### 4.4  Synthesiser → Final Contract
```jsonc
{
  "status": "final" | "revise",
  "answer": "0.8488",                 // always present when status==final
  "critique_summary": "…"            // free-text; hidden from end user
}
```

**Strict JSON** with no stray tokens enables automatic parsing in the orchestrator.

**Additional Implementation Notes:**
- All agent outputs are validated as strict JSON. If parsing fails, the orchestrator generates a fallback output to keep the pipeline running.
- The system guarantees a numeric answer is always returned, even if it must fall back to a dummy value (e.g., "0").
- Post-processing of answers (e.g., removing commas, preferring largest number, excluding years, converting percentages) is fully configurable via `six_agent_config`.

---

## 5  Orchestration Logic

1. **Manager Task**
   • If cache hit ⇒ return cached answer.
   • Else ⇒ proceed.
2. **Extractor Task**
3. **Reasoner Task**
   • If JSON contains a `dsl` program ⇒ orchestrator executes it via `execute_dsl_program()` and injects result back into JSON before critics run.
4. **Critic Phase**
   • `extraction_critic` runs on extractor JSON.
   • `calculation_critic` runs on reasoner JSON.
5. **Synthesiser Task**
   • Reads critics' verdicts.
   • If both `is_correct==true` ⇒ status=`final`. Cache answer, post-process (scale normaliser, comma removal) and return.
   • Else ⇒ builds condensed feedback and returns status=`revise`.
6. **Iteration Loop** (max `six_agent_config.max_iterations`)
   • Feedback injected into Reasoner prompt, goto step 3.

Fallback: after max iterations we return the latest Reasoner answer flagged with low confidence.

**Additional Implementation Details:**
- **Error Recovery:** Each stage (manager, extraction, reasoning, critics, synthesis) is wrapped in error recovery logic. If a stage fails (e.g., LLM error, invalid output), a fallback output is generated and the pipeline continues.
- **Circuit Breaker:** A circuit breaker monitors repeated failures per stage and can skip to fallback logic if a stage is unstable.
- **Budget-Aware Early Exit:** If the estimated token/cost budget is nearly exhausted, the pipeline will exit early and return the best available answer.
- **Token & Cost Tracking:** All stages log token usage and cost for later analysis. Token and cost budgets are configurable in `six_agent_config`.
- **Cache:** The manager agent can short-circuit the pipeline and return cached answers for repeated or trivial follow-up questions. This is controlled by `cache_enabled` in config.
- **Verbose Logging & Tracing:** Verbose mode and optional tracing log detailed progress and decisions at each stage for debugging and experiment tracking.

---

## 6  Implementation Roadmap

1. **Agents** – add `build_agents_six()` in `agents.py` (reuse helper functions).  
2. **Tasks** – new builders in `tasks.py` with schemas above.  
3. **Tools** – `CalculatorTool` wraps `execute_dsl_program` for deterministic maths.  
4. **Orchestrator** – switch on `mode` (`three` vs `six`). Sequential execution keeps memory footprint low; parallel critic runs are optional optimisation.  
5. **Predictor** – expose `ConvFinQAMultiAgentPredictorV3` that selects orchestrator mode based on config.  
6. **Tests** – add unit tests for each prompt builder & contract parsing; integration smoke test on 3 records.  
7. **Benchmark harness** – extend `scripts/benchmark_multi_agent.py` to toggle modes for A/B runs.

**Additional Implementation Steps:**
- Implement `StageRecoveryManager` and `TokenUsageTracker` for robust error handling and cost control.
- Add post-processing hooks to clean and normalize numeric answers according to config.
- Integrate verbose logging and optional tracing for debugging and experiment tracking.

---

## 7  Design Rationale

* **Separation of concerns** – empirical gains in FinQA tasks when extraction & calculation are decoupled.
* **Early fault isolation** – critics judge *before* the answer reaches the user, preventing compounded errors.
* **Cheap routing** – Manager avoids heavy model calls on trivial follow-ups (~12 % of turns in training set).
* **Strict contracts** – JSON everywhere enables fast, deterministic post-processing and easier eval.
* **Config-driven** – researchers can swap model classes (Gemini, Claude, etc.) with a single JSON edit.

**Additional Rationale:**
- Robustness to LLM or infrastructure failures is achieved via fallback logic and circuit breakers.
- Cost and token usage are tightly controlled to enable large-scale benchmarking and avoid overruns.
- All agent roles, models, and temperatures are fully config-driven for easy A/B testing and extension.

---

## 8  Known Limitations & Future Work

1. Manager cache currently string-matches questions; semantic cache todo.
2. Critics rely on language plausibility, not re-computing maths – future: integrate deterministic replay.
3. Parallel execution requires CrewAI v0.29+; fallback sequential mode is slower.
4. Tooling limited to single calculator; adding **Table Viewer** tool could tighten extractor grounding.

**Additional Limitations:**
- Current fallback logic is conservative (returns "0" or minimal output); future work could use more sophisticated heuristics for fallback answers.
- Token/cost tracking is based on estimates; actual API costs may vary slightly.

**Future Work: Model Efficiency and Tooling**
- Further empirical testing is required to systematically reduce reliance on sophisticated LLMs such as GPT-4o and GPT-4o-mini. A key research direction is to evaluate whether lighter-weight models (e.g., GPT-3.5) can achieve comparable performance when paired with improved intermediate tooling, deterministic processing, or enhanced prompt engineering between agent interactions. This could significantly reduce inference costs and improve scalability for large-scale deployments.

---

## 9  References

* Fatemi, N., & Hu, X. (2024). *Enhancing Financial Question Answering with a Multi-Agent Reflection Framework*. arXiv:2410.21741.  
* Ho, J. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*.  
* ConvFinQA dataset & evaluation protocol – original repository. 