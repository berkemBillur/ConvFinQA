# ConvFinQA Multi-Agent Solution – Project Report

## 1. Problem Statement
The goal of this project is to build an automated system that can answer **Conversational Financial Question Answering (ConvFinQA)** problems – multi-turn dialogues where each user question may refer to both the **financial tables / narrative** of an SEC filing *and* to **previous turns in the dialogue**.  The system must:

1. Retrieve the correct numeric facts from a semi-structured table embedded in the filing,
2. Perform the necessary arithmetic (differences, ratios, percentage change, etc.), and
3. Return a single numeric answer in machine-readable form.

ConvFinQA is challenging because it combines **information extraction, conversational coreference, and numerical reasoning**.  Recent work (e.g. *Enhancing Financial QA with a Multi-Agent Reflection Framework*, arXiv:2410.21741) shows that a **team of specialised LLM agents** can outperform single-prompt baselines.  Our task was therefore to design, implement, and benchmark such a multi-agent approach under tight time constraints. Over the course of the project, we iteratively explored and benchmarked several agent-based architectures, culminating in a robust six-agent system.

## 2. Dataset & Evaluation
We used the cleaned dataset provided in `data/convfinqa_dataset.json` (see `dataset.md`).  Our scripts sample a subset of conversations for quick iteration and evaluate with the official ConvFinQA metrics implemented in `src/evaluation/`.

Each benchmark run produces:
* **`all_results.txt / passed_results.txt / failed_results.txt`** – granular pass/fail logs.
* **`run_metadata.txt`** – configuration hash, dataset slice, and exec-time statistics.
* **`run_time_logs.txt`** – per-turn latency & cost estimates.

Results are stored under `experiment_tracking/results/<timestamp>/`, enabling side-by-side comparison and easy rollback.

> **Note:** The evaluation protocol has remained consistent throughout all iterations. For cost reasons, we typically benchmarked on small subsets of the dataset and iteratively improved the implementation based on these results.

## 3. Approach: Six-Agent System (Final)
Our final implementation adopts a **six-agent, tiered architecture** (see `src/predictors/multi_agent/agents.py` for technical details):

| Tier | Agent                | Role & Goal |
|------|----------------------|-------------|
| 0    | Conversation Manager | Routes questions, checks cache, and decides whether to run the full pipeline |
| 1    | Data Extraction Specialist | Extracts exact table cells, normalizes units, scale, and polarity |
| 1    | Financial Calculation Reasoner | Performs chain-of-thought arithmetic and generates executable DSL programs |
| 2    | Extraction Critic    | Verifies extraction accuracy: correct rows, right fiscal year, proper scaling |
| 2    | Calculation Critic   | Verifies order of operations, unit consistency, and magnitude sanity |
| 3    | Answer Synthesiser   | Merges reasoner output with critic feedback and formats the final answer |

The high-level workflow is:

```text
Manager → Extractor → Reasoner → [Critics in parallel] → Synthesiser → (If revision needed, loop)
```

- **Manager**: Checks for cache hits and routes trivial questions directly; otherwise, initiates the full pipeline.
- **Extractor**: Isolates relevant numbers from the document, resolving references and normalizing values.
- **Reasoner**: Applies financial logic, performs calculations, and outputs both step-by-step reasoning and an optional DSL program.
- **Critics**: Independently audit extraction and calculation for correctness, providing actionable feedback.
- **Synthesiser**: Aggregates outputs, determines if the answer is final or needs revision, and formats the user-facing response.

All agent roles, prompts, and configuration are defined in `src/predictors/multi_agent/agents.py` and are fully config-driven for easy experimentation.

### Why Move Beyond the Three-Agent Approach?
Despite careful prompt engineering and iterative improvements, the three-agent system (see below) plateaued in performance, particularly on complex multi-turn reasoning and error isolation. The six-agent system was adopted to:
- Decouple extraction and calculation for better fault isolation
- Add a manager for efficient routing and caching
- Introduce a synthesiser for robust answer aggregation and revision handling
- Enable more granular error analysis and targeted improvements

## 4. Historical Architectures & Limitations

### Three-Agent Reflection Framework (Deprecated)
- **Expert**: Unified extraction + calculation reasoning
- **Extraction Critic**: Reviews data extraction quality
- **Calculation Critic**: Reviews mathematical reasoning

**Workflow:**
```text
Expert → Critics → (If critique) Expert-Revision → … (max 2 iterations)
```
**Limitations:**
- Combined extraction and calculation led to cognitive overload and error propagation
- Critics could only flag issues after-the-fact, with limited ability to isolate root causes
- No manager or synthesiser, so trivial questions and answer formatting were less robust

### Four-Agent Hierarchical Crew (Archived)
- **Supervisor**: Decomposes the question and orchestrates agents
- **Extractor**: Locates exact numerical values and resolves references
- **Calculator**: Builds DSL programs and performs calculations
- **Validator**: Recalculates, checks logic, and assigns confidence

**Workflow:**
```text
Supervisor → Extractor → Calculator → Validator → Final answer
```
**Limitations:**
- Rigid, sequential pipeline with no iterative feedback loop
- Validator could not trigger revisions, so errors often propagated to the final answer
- No caching, answer synthesis, or parallel critic review

> **See**: `src/predictors/multi_agent/archive/` for code and further details on these historical implementations.

## 5. Iterative Development Cycle
1. **Prototype** – Implemented minimal CrewAI scaffold and verified end-to-end execution on 10 conversations.
2. **Benchmark** – Ran `scripts/benchmark_multi_agent.py` which logs results to `experiment_tracking/results/`.
3. **Analyse** – Used quick pandas notebooks + the tracker dashboard (`scripts/dashboard.py`) to inspect failure modes (e.g., year vs. value confusion, percentage scaling).
4. **Refine** – Tweaked prompts ⟶ re-benchmark ⟶ compare snapshot hashes.  Roughly ten such cycles are recorded in the repo.
5. **Architectural Evolution** – Progressed from four-agent to three-agent to the final six-agent system, each time addressing the limitations of the previous design.

## 6. Key Findings & Error Analysis
| Category | Typical Failure | Mitigation in Six-Agent System |
|----------|-----------------|-------------------------------|
| **Extraction** | Metric synonym not matched ("repairs & maintenance" vs "maintenance") | Dedicated extraction agent and critic, plus improved prompt rules |
| **Conversation Reference** | Incorrect resolution of "that value less 1" | Explicit reference resolution in extractor, critic feedback loop |
| **Calculation** | Sign flips in % change when base value is negative | Reasoner agent with step-by-step logic, critic validation |
| **Output Format** | JSON malformed under iteration pressure | Synthesiser agent for robust formatting and revision handling |

### Performance Snapshot (BestPerformanceLog.csv)
The file `BestPerformanceLog.csv` captures the **current best run** of the 3-agent reflection system:

* Questions evaluated: **31**  
* Correct answers: **16**  
* Exact-match accuracy: **≈ 51.6 %**  
* Total cost: **≈ $0.36**  
* Avg. cost per question: **≈ $0.011**  
* Avg. execution time per question: **≈ 9 s** (31 questions, 279 s total)

While this single run does not convey the full learning curve, it provides a clear reference point for the system's best observed performance to date.  A time-series breakdown of accuracy, cost and latency for **all** experiments can be explored interactively via the Streamlit dashboard (`scripts/dashboard.py`).

## 7. Future Work
- **Automated Error Analysis & Targeted Retraining:**
  Implement a simple pipeline that automatically clusters errors by type (e.g., extraction, calculation, reference) and triggers targeted prompt or model retraining for the weakest agent. This would allow for rapid, data-driven improvement without manual triage.
- **Model Matrix & Cost/Accuracy Trade-off:**
  Benchmark a range of models (large, mid-tier, lightweight) for each agent role to optimize cost and accuracy.
- **Tool Augmentation:**
  Integrate deterministic tools (e.g., table retrievers, function calling) for critical steps, especially in extraction and calculation.
- **Semantic Caching:**
  Upgrade the manager to use semantic similarity for cache hits, not just string matching.

## 8. Conclusion
We delivered a fully-functional **CrewAI-based six-agent system** with robust experiment tracking and a reproducible benchmarking pipeline. The current architecture addresses the limitations of previous designs and is ready for further enhancement:
* Plug-and-play prompts / models, thanks to config-driven factory functions.
* Rich historical snapshots to guide data-driven prompt engineering.
* Clear roadmap towards even more modular, tool-augmented, and cost-efficient reasoning.

With additional time for prompt tuning and more powerful models, we are confident the system can reach or surpass state-of-the-art results on ConvFinQA.

## 9. Note on AI-Assisted Development
Throughout this project, we used **ChatGPT** to assist with documentation write-ups for their ease, debugging and generation of code. All planning, architectural decisions, and critical logic were performed by the author. For research and literature review, tools like **Perplexity** were also utilised.
