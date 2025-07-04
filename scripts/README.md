# Scripts Directory

This folder contains the command-line utilities and developer tools that live outside the main Python package (`src`). They are intended for:

• benchmarking the predictors  
• managing and analysing experiment runs  
• visualising results in a browser  
• ad-hoc validation during development.

## Directory snapshot

```
scripts/
├── benchmark_multi_agent.py   # Run an end-to-end evaluation of the CrewAI predictor
├── dashboard.py               # Streamlit dashboard that visualises experiment logs
├── experiment_manager.py      # Lightweight CLI to query the experiment database
├── test_enhanced_tracking.py  # Unit-style tests for the enhanced tracking layer
├── archived/                  # Legacy scripts kept for reference
│   ├── benchmark_hybrid.py
│   ├── benchmark_multi_agent.py
│   ├── analyse_experiments.py
│   ├── experiment_runner.py
│   ├── validate_pipeline.py
│   └── test_suite.py
└── README.md                  # you are here
```

### Why are there two `benchmark_multi_agent.py` files?

Older versions of the benchmark were moved to `scripts/archived/`. The root-level `benchmark_multi_agent.py` is the maintained and feature-rich rewrite. When in doubt, use the root-level version.

## Key scripts

### benchmark_multi_agent.py

**Purpose:**

This script is the main entry point for evaluating the CrewAI multi-agent predictor on the ConvFinQA dataset. It is instrumental in recording and tracking every experiment conducted, especially as the system evolved through different implementations and configurations. Each time you run this script, it:

- Benchmarks the multi-agent system on a chosen dataset sample
- Records the full configuration of the agents, models, and workflow used for that run
- Logs detailed results, including accuracy, cost, execution time, and per-agent breakdowns
- Saves all outputs in a timestamped folder under `experiment_tracking/results/`

This makes it easy to compare different approaches, track improvements, and audit the effect of any change to the multi-agent setup.

**How to view experiment results:**

- **Directly:** Browse the `experiment_tracking/results/` directory. Each subfolder (named by timestamp) contains:
  - `all_results.txt` (full question-by-question breakdown)
  - `failed_results.txt` / `passed_results.txt`
  - `run_metadata.txt` (configuration and summary)
  - `run_time_logs.txt` (full console output)
- **Interactively:** Run the Streamlit dashboard app:
  ```
  streamlit run scripts/dashboard.py
  ```
  This provides an interactive breakdown of all experiments, allowing you to:
  - Filter and compare runs by date, configuration, or dataset
  - Visualise accuracy, cost, and agent performance over time
  - Inspect the exact configuration (models, parameters, etc.) used for any run

**Example usage:**
```
python scripts/benchmark_multi_agent.py                # quick 5-conversation smoke test
python scripts/benchmark_multi_agent.py -n 20 --save   # full run and persistent log
python scripts/benchmark_multi_agent.py -n 10 --random-sample --seed 42
```

**Output:**
```
experiment_tracking/results/YYYYMMDD_HHMMSS/
├── all_results.txt
├── failed_results.txt
├── passed_results.txt
├── run_metadata.txt
└── run_time_logs.txt
```

### dashboard.py

A Streamlit application that turns the contents of `experiment_tracking/results/` into interactive charts and tables.

Run with:
```
streamlit run scripts/dashboard.py
```

### experiment_manager.py

A thin CLI wrapper around the enhanced tracker database.

```
python scripts/experiment_manager.py list -n 10   # last 10 experiments
python scripts/experiment_manager.py stats        # database size and path
```

### test_enhanced_tracking.py

Developer-focused regression suite that checks the tracker can:

1. create a snapshot  
2. store performance results  
3. rollback and delete data.

Run:
```
python scripts/test_enhanced_tracking.py          # runs the full test battery
```

## Setting up the environment

1. Install the dependencies defined in `pyproject.toml`. For a minimal setup:

```
uv pip install -r requirements.txt   # or poetry install
```

2. The multi-agent benchmark requires an OpenAI key:

```
export OPENAI_API_KEY=sk-your-key
```

3. The dashboard needs Streamlit and Plotly (already declared in the project). Start it from the project root so the relative paths resolve.

## Development tips

• Use `test_enhanced_tracking.py` before committing to ensure no breaking changes to the tracking layer.  
• Keep legacy scripts in `scripts/archived/`; they are not executed by CI but may serve as examples.

## Licence

See the root `README.md`. 