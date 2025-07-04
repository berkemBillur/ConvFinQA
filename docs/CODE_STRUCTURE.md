# ConvFinQA Codebase Structure

Directory organisation for the ConvFinQA challenge with enhanced tracking and multi-agent capabilities.

## Key Components

### Core Infrastructure

#### Data Layer (`src/data/`)
- **`dataset.py`**: ConvFinQA dataset loading with train/dev splits and conversation structure handling
- **`models.py`**: Pydantic models defining data structures (`ConvFinQARecord`, `Dialogue`, `Document`)
- **`utils.py`**: Table processing utilities and numerical data extraction functions

#### Evaluation Framework (`src/evaluation/`)
- **`evaluator.py`**: Main evaluation pipeline with quick/comprehensive evaluation modes
- **`metrics.py`**: Accuracy calculation, conversation-level evaluation, and result aggregation
- **`executor.py`**: DSL program execution with proper error handling and timeout management
- **`multi_agent_results.py`**: Multi-agent specific result processing and analysis

#### Configuration Management (`src/utils/`)
- **`config.py`**: Structured configuration system with path resolution and environment management
- **`enhanced_tracker.py`**: **✅ Enhanced experiment tracking system** with comprehensive parameter capture
- **`text_processing.py`**: Text processing utilities for number extraction and cleaning
- **`financial_matcher.py`**: Financial KPI matching with fuzzy search capabilities
- **`scale_normalizer.py`**: Scale-aware numerical processing for financial data

### Predictor Implementations (`src/predictors/`)

#### Multi-Agent CrewAI System
- **`multi_agent_predictor.py`**: **✅ Config-driven multi-agent predictor** (supports three-agent *and* six-agent modes; four-agent kept only in `archive/`) with enhanced tracking integration
- **`multi_agent/`**: CrewAI-specific implementation components:
  - **`predictor.py`**: Core six-agent orchestrator (manager, extractor, reasoner, two critics, synthesiser) with enhanced tracking
  - **`agents.py`**: Agent definitions and configurations
  - **`tasks.py`**: Task definitions for each agent
  - **`orchestrator.py`**: Agent coordination and workflow management
  - **`tracer.py`**: Execution tracing and monitoring
- **`tools/`**: Specialised agent tools organised by function:
  - **`supervisor_tools.py`**: Task decomposition and orchestration
  - **`extraction_tools.py`**: Data retrieval and reference resolution
  - **`calculation_tools.py`**: Financial reasoning and DSL generation
  - **`validation_tools.py`**: Quality assurance and error detection

### Enhanced Tracking System ✅

#### Core Tracking (`src/utils/enhanced_tracker.py`)
- **`EnhancedExperimentTracker`**: Comprehensive experiment tracking with SQLite database
- **Configuration Classes**: Complete parameter capture for agents, workflows, tasks, datasets, and systems
- **Performance Results**: Detailed metrics collection including agent-specific and question-type breakdowns
- **Database Storage**: Structured experiment data with indexing and querying capabilities

#### Data Storage (`experiment_tracking/`)
- **`configurations/`**: Database and configuration snapshots
  - **`experiments.db`**: SQLite database with experiment metadata and results
  - **`snapshots.json`**: Complete experiment configuration snapshots
- **`results/`**: Timestamped experiment result files with comprehensive output

### Experimentation Infrastructure

#### Current Scripts (`scripts/`)
- **`benchmark_multi_agent.py`**: **✅ Enhanced benchmark script** with comprehensive tracking integration
- **`experiment_manager.py`**: **✅ CLI management tool** for experiment analysis and comparison
- **`dashboard.py`**: **✅ Interactive Streamlit dashboard** for experiment visualization and management
- **`test_enhanced_tracking.py`**: **✅ Enhanced tracking validation** and integration testing

#### Archived Scripts (`scripts/archived/`)
- Legacy benchmarking and tracking implementations (deprecated but preserved for reference)

## Integration Patterns

### Enhanced Tracking Integration
The system provides comprehensive experiment tracking:
- **Complete Configuration Capture**: All agent parameters, workflow settings, and system configuration
- **Performance Monitoring**: Success rates, execution times, costs, and error patterns
- **Agent-Specific Metrics**: Individual agent performance breakdown and analysis
- **Question Type Analysis**: Performance categorization by calculation, lookup, comparison, etc.

### Evaluation Workflow
1. **Data Loading**: `ConvFinQADataset` handles train/dev splits
2. **Configuration Snapshot**: Enhanced tracker captures complete experiment setup
3. **Prediction**: Multi-agent predictor with automatic tracking integration
4. **Evaluation**: `ConvFinQAEvaluator` manages the evaluation pipeline
5. **Result Storage**: Comprehensive results stored in unified tracking structure
6. **Analysis**: Dashboard and CLI tools enable systematic optimization

## Development Workflow

### Running Experiments
```bash
# Enhanced benchmark with tracking
python scripts/benchmark_multi_agent.py -n 10 --random --seed 42

# View experiment results
python scripts/experiment_manager.py list --recent 5

# Launch interactive dashboard
python scripts/dashboard.py
```

### Experiment Management
- **CLI Tools**: Use `experiment_manager.py` for command-line experiment analysis
- **Interactive Dashboard**: Streamlit-based visualization and comparison tools
- **Configuration Comparison**: Compare experiments and revert to previous configurations
- **Performance Analysis**: Detailed breakdown by agent, question type, and error patterns

### Quality Assurance
- Type hints throughout following existing patterns
- Comprehensive error handling with graceful degradation
- Enhanced tracking ensures complete reproducibility
- Integration testing via validation scripts

## Current Directory Structure

```
BerkemBillur/
├── data/
│   └── convfinqa_dataset.json          # ConvFinQA dataset (3,458 conversations)
├── src/
│   ├── data/
│   │   ├── dataset.py                  # Data loading & preprocessing
│   │   ├── models.py                   # Pydantic data models
│   │   └── utils.py                    # Data utilities & table processing
│   ├── evaluation/
│   │   ├── evaluator.py                # Main evaluation pipeline
│   │   ├── metrics.py                  # Evaluation metrics & result aggregation
│   │   ├── executor.py                 # DSL program execution
│   │   └── multi_agent_results.py      # Multi-agent result processing
│   ├── predictors/
│   │   ├── multi_agent_predictor.py    # Multi-agent implementation
│   │   ├── multi_agent/                # CrewAI-specific components
│   │   │   ├── predictor.py            # Core predictor with enhanced tracking
│   │   │   ├── agents.py               # Agent definitions
│   │   │   ├── tasks.py                # Task definitions
│   │   │   ├── orchestrator.py         # Workflow management
│   │   │   └── tracer.py               # Execution tracing
│   │   └── tools/                      # Specialised agent tools
│   │       ├── calculation_tools.py    # Financial calculation tools
│   │       ├── extraction_tools.py     # Data extraction tools
│   │       ├── supervisor_tools.py     # Task orchestration tools
│   │       └── validation_tools.py     # Quality assurance tools
│   └── utils/
│       ├── config.py                   # Configuration management
│       ├── enhanced_tracker.py         # Enhanced experiment tracking
│       ├── financial_matcher.py        # Financial KPI matching
│       ├── scale_normalizer.py         # Scale-aware processing
│       └── text_processing.py          # Text utilities
├── experiment_tracking/
│   ├── configurations/                 # Tracking database and snapshots
│   │   ├── experiments.db              # SQLite experiment database
│   │   └── snapshots.json              # Configuration snapshots
│   └── results/                        # Timestamped experiment results
│       └── [YYYYMMDD_HHMMSS]/          # Individual experiment outputs
├── scripts/
│   ├── benchmark_multi_agent.py        # Enhanced benchmark script
│   ├── experiment_manager.py           # CLI experiment management
│   ├── dashboard.py                    # Interactive dashboard
│   ├── test_enhanced_tracking.py       # Tracking validation
│   └── archived/                       # Legacy scripts (preserved for reference)
├── docs/
│   ├── ENHANCED_TRACKING.md            # Enhanced tracking documentation
│   ├── MULTI_AGENT_APPROACH.md         # Multi-agent system documentation
│   ├── CODE_STRUCTURE.md               # This file
│   ├── CREWAI_SETUP.md                 # CrewAI setup and configuration
│   ├── PROMPT_GUIDE.md                 # Development guidelines
│   └── TASK_OVERVIEW.md                # High-level task description
└── config/
    └── base.json                       # System configuration
```

This structure provides a robust foundation for financial question answering research with comprehensive experiment tracking, reproducibility, and systematic optimization capabilities.
