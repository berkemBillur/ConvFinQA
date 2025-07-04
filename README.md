# ConvFinQA Multi-Agent Solution

This repository implements a **six-agent LLM pipeline** for **Conversational Financial Question Answering (ConvFinQA)** over SEC filings. It combines specialized agents for data extraction, numerical reasoning, validation, and answer synthesis, all orchestrated and benchmarked via a reproducible tracking framework.

## Project Highlights

- **Problem Addressed**: Multi-turn dialogues requiring table lookups, coreference resolution, and arithmetic across financial tables and narrative text.
- **Approach**: A configurable, six-tier agent system:
  1. **Manager**: Routes queries, checks cache, and triggers the pipeline only when needed.
  2. **Extractor**: Locates and normalizes numeric facts from tables.
  3. **Reasoner**: Performs step-by-step arithmetic and generates optional DSL programs.
  4. **Extraction & Calculation Critics**: Independently verify data accuracy and math correctness in parallel.
  5. **Synthesiser**: Aggregates outputs, applies feedback loops, and formats the final answer.
- **Benchmarking & Tracking**: Automated experiments with detailed logs in `experiment_tracking/`, enabling side-by-side comparison and rollback.
- **Interactive Tools**: CLI chat interface, Streamlit dashboard, and Jupyter notebooks for exploration and analysis.

## Repository Structure

```
ConvFinQA/
├── config/                    # API key templates (ignored by Git)
├── data/                      # Cleaned ConvFinQA dataset (`convfinqa_dataset.json`)
├── docs/                      # Architecture overviews and research notes
├── src/                       # Core code (agents, evaluation, utilities)
├── experiment_tracking/      # Snapshots, configurations, and results logs
├── scripts/                   # Benchmarking, dashboard, and utility scripts
├── notebooks/                 # Data exploration and analysis notebooks
├── figures/                   # Example outputs and diagrams
├── README.md                  # Project overview and getting started (this file)
└── dataset.md                 # Detailed dataset specification
```

## Getting Started

### Prerequisites
- Python 3.12+
- [UV Environment Manager](https://docs.astral.sh/) (optional) or `pip`
- OpenAI API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/berkemBillur/ConvFinQA.git
   cd ConvFinQA
   ```

2. **Set up the environment**:
   - With UV:
     ```bash
     brew install uv
     uv sync
     uv add python
     ```
   - Or with pip:
     ```bash
     pip install -r requirements.txt
     ```

3. **Configure your API key**:
   ```json
   // config/api_keys.json
   {
     "OPENAI_API_KEY": "your_api_key_here"
   }
   ```

## Usage

- **Chat via CLI**:
  ```bash
  uv run main chat <record_id>
  ```
- **Run Benchmark Suite**:
  ```bash
  python scripts/benchmark_multi_agent.py
  ```
- **Launch Dashboard**:
  ```bash
  python scripts/dashboard.py
  ```
- **Explore with Notebooks**:
  ```bash
  jupyter lab notebooks/
  ```

## Detailed Documentation

- `dataset.md` — ConvFinQA dataset details
- `docs/CODE_STRUCTURE.md` — Code organization guide
- `docs/SIX_AGENT_APPROACH.md` — Final multi-agent architecture deep dive
- `docs/ENHANCED_TRACKING.md` — Experiment tracking and metadata
- `docs/THREE_AGENT_APPROACH.md`, `docs/FOUR_AGENT_APPROACH.md` — Historical system designs

## Contributing

Contributions and feedback are welcome! Please open an issue or submit a pull request. For major changes, start a discussion via an issue.

## Security & Secrets

- **Do not commit** any secrets or API keys. Use `config/api_keys.json` (ignored by Git) for credentials.
- The repository is protected by secret-scanning rules to prevent accidental leaks. 