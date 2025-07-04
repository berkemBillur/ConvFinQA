# Project Setup & Quick-Start Guide

> This document pulls together the essential information scattered across the repository so you can get up and running in minutes.

---

## 1. Prerequisites

• **Python 3.12 or newer**  
• **[UV](https://docs.astral.sh/uv/)** – lightning-fast dependency manager and virtual-env wrapper  
• An **OpenAI account** with an API key

---

## 2. Clone & create an environment

```bash
# 1. Create a fresh virtual env & install all locked dependencies
uv sync            # reads pyproject.toml & poetry-style lock file

```

> **Tip:** `uv sync` is idempotent: rerun it at any time to ensure your env matches the lock file.

---

## 3. Configure your OpenAI key

The multi-agent components talk to the OpenAI API. There are two equivalent ways to provide the key.

### A. Local JSON file

Create `config/api_keys.json` (already git-ignored):

```json
{
    "openai": {
        "api_key": "{YOUR OPENAI API KEY}",
        "organization_id": null
    },
    "description": "API keys for ConvFinQA project - this file is gitignored and local only"
} 
```

The code will attempt to read the environment variable first and fall back to the JSON file.

> **❗ Important:** Without a valid key the multi-agent predictor and dashboard will not work.

---

## 4. Running the benchmark

The main entry point is the **multi-agent benchmark**:

```bash
python scripts/benchmark_multi_agent.py           # smoke-test on 5 conversations
python scripts/benchmark_multi_agent.py -n 1 --save --random --seed 42
```

Each run creates a timestamped folder under `experiment_tracking/results/` containing:

```
all_results.txt      # detailed Q&A log
failed_results.txt   # only the misses
passed_results.txt   # only the hits
run_metadata.txt     # config & summary statistics
run_time_logs.txt    # full console output
```

Configurations (agent models, temperatures, workflow type, etc.) are captured automatically so you can reproduce any run later.

---

## 5. Visualising results

Launch the interactive **Streamlit dashboard**:

```bash
streamlit run scripts/dashboard.py
```

The dashboard reads the `experiment_tracking/results/` directory and lets you:

• filter experiments by date, sample size, or config hash  
• inspect per-agent costs & execution times  
• plot accuracy trends over time  
• drill into any question to view the full reasoning chain

---

## 6. Exploring the multi-agent framework

The implementation lives in `src/predictors/multi_agent/`.

Key files:

• `agents.py`        – role definitions and LLM wrappers  
• `orchestrator.py`  – task decomposition & agent coordination  
• `tasks.py`         – shared task objects and tools  
• `predictor.py`     – high-level API used by the benchmark  
• `tracer.py`        – step-by-step execution logging

> The design follows recent research showing that **critic-style agents** improve numeric reasoning by catching gaps in the calculation chain.

For a quick read-through of the code paths, start with `predictor.py` and follow the calls into `orchestrator.py`.

---

## 7. Further utilities

• `scripts/experiment_manager.py` – CLI to list experiments and show DB stats  
• `scripts/test_enhanced_tracking.py` – run a fast regression check on the tracking layer  
• `scripts/archived/` – older prototypes kept for reference

---

## 8. Troubleshooting

Problem | Fix
--- | ---
`KeyError: OPENAI_API_KEY` | Ensure the env variable is exported **or** `config/api_keys.json` exists.
Streamlit can't find data | Launch from repo root so relative paths resolve.
Module not found (crewai) | `uv add crewai crewai-tools langchain-openai`