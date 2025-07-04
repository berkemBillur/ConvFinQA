# Enhanced Experiment Tracking System

## Overview

The Enhanced Experiment Tracking System provides comprehensive capture, storage, and analysis of multi-agent experiment configurations and performance results. This system addresses the critical need for reproducibility and systematic experimentation in multi-agent conversational financial question answering.

## Key Features

### 🔍 Complete Configuration Capture
- **Agent Configurations**: Model types, temperatures, roles, goals, backstories, tools, delegation settings
- **Workflow Settings**: Process type, manager configuration, memory/cache settings, execution timeouts
- **Task Definitions**: Task types, descriptions, dependencies, agent assignments
- **Dataset Configuration**: Sampling strategy, random seeds, conversation selection, question type distribution
- **System Environment**: API settings, git commit, environment variables, rate limits

### 📊 Comprehensive Performance Tracking
- **Overall Metrics**: Accuracy rates, execution times, costs, failure rates
- **Agent-Specific Performance**: Individual agent success rates, execution times, cost breakdowns
- **Question Type Analysis**: Performance by question category (calculation, lookup, comparison, etc.)
- **Error Pattern Analysis**: Common failure types, error frequency, failure examples
- **Cost Analysis**: Per-agent costs, per-question-type costs, total experiment costs

### 🗃️ Database Storage & Management
- **SQLite Database**: Fast, reliable local storage with structured queries
- **JSON Snapshots**: Complete experiment snapshots for detailed analysis
- **Indexing**: Optimized queries by timestamp, configuration hash, accuracy
- **Data Export**: JSON export functionality for backup and sharing

### 📈 Interactive Dashboard
- **Real-time Visualization**: Performance trends, configuration comparisons
- **Filtering & Search**: By date range, model types, accuracy thresholds
- **Configuration Analysis**: Model usage patterns, performance correlations
- **Management Tools**: Experiment reversion, data export, database maintenance

## Architecture

```
Enhanced Tracking System
├── Core Components
│   ├── enhanced_tracker.py         # Main tracking logic
│   ├── Configuration Snapshots     # Complete experiment capture
│   ├── Performance Results         # Detailed metrics storage
│   └── SQLite Database            # Structured data storage
├── Integration
│   ├── benchmark_multi_agent.py    # Enhanced benchmark script
│   └── Multi-agent Predictor      # Automatic tracking integration
├── Management Tools
│   ├── experiment_manager.py      # CLI management utility
│   └── dashboard.py               # Interactive Streamlit dashboard
└── Storage
    ├── experiment_tracking/configurations/  # Database and JSON storage
    └── experiment_tracking/results/         # Timestamped result files
```

## Quick Start

### 1. Run Enhanced Benchmark
```bash
# Basic run with tracking
python scripts/benchmark_multi_agent.py -n 5

# Random sampling with seed and notes
python scripts/benchmark_multi_agent.py -n 10 --random --seed 42 --notes "Testing new configuration"

# Full run with saved results
python scripts/benchmark_multi_agent.py -n 20 --save --show-questions
```

### 2. View Experiments
```bash
# List recent experiments
python scripts/experiment_manager.py list

# Show experiment details
python scripts/experiment_manager.py show exp_20241227_143022

# Database statistics
python scripts/experiment_manager.py stats
```

### 3. Launch Dashboard
```bash
# Install dashboard dependencies (if needed)
pip install streamlit pandas plotly

# Launch interactive dashboard
python scripts/dashboard.py
```

## Configuration Capture

The system automatically captures complete experiment configurations including:

### Agent Configuration
```python
AgentConfiguration(
    name="supervisor",
    role="Financial QA Orchestrator", 
    goal="Decompose conversational financial queries...",
    backstory="You are a senior financial analyst...",
    model="gpt-4o",
    temperature=0.1,
    verbose=True,
    allow_delegation=False,
    tools=[],
    max_execution_time=None
)
```

### Dataset Configuration
```python
DatasetConfiguration(
    total_conversations=737,
    sample_size=10,
    sampling_strategy="sequential",
    random_seed=None,
    conversation_ids=["train_0", "train_1", ...],
    avg_conversation_length=3.2,
    question_types={
        "calculation": 15,
        "lookup": 8,
        "comparison": 5,
        "other": 4
    }
)
```

## Performance Tracking

### Comprehensive Metrics
```python
PerformanceResults(
    total_questions=32,
    correct_answers=28,
    accuracy_rate=0.875,
    total_execution_time=145.3,
    avg_execution_time_per_question=4.54,
    total_estimated_cost=0.0234,
    avg_cost_per_question=0.00073,
    failure_count=4,
    failure_rate=0.125,
    
    # Detailed breakdowns
    agent_performance={
        "supervisor": {"success_rate": 0.90, "avg_time": 1.2, ...},
        "extractor": {"success_rate": 0.85, "avg_time": 0.8, ...},
        ...
    },
    question_type_performance={
        "calculation": {"accuracy_rate": 0.80, "avg_time": 5.2, ...},
        "lookup": {"accuracy_rate": 0.95, "avg_time": 3.1, ...},
        ...
    },
    error_patterns={
        "TimeoutError": 2,
        "ValidationError": 1,
        "ParseError": 1
    }
)
```

## Database Schema

### Main Tables
- **experiments**: Core experiment metadata and results
- **agent_performance**: Agent-specific performance metrics
- **question_type_performance**: Performance by question category
- **error_patterns**: Error analysis and patterns

### Key Indexes
- Timestamp index for chronological queries
- Configuration hash index for duplicate detection
- Accuracy index for performance filtering

## Dashboard Features

### Overview Tab
- Key performance metrics and recent experiments
- Accuracy distribution charts and quick navigation

### Configuration Analysis Tab
- Model usage analysis and configuration comparison tools
- Performance correlation analysis and side-by-side experiment comparison

### Performance Trends Tab
- Time-series performance analysis and question type breakdown
- Cost analysis trends and execution time patterns

### Agent Analysis Tab
- Individual agent performance metrics and success rate comparisons
- Cost distribution by agent and agent-specific trends

### Management Tab
- Experiment reversion tools and configuration export/import
- Database maintenance utilities and data backup/restore

## CLI Management

### Basic Commands
```bash
# List experiments with details
python scripts/experiment_manager.py list -n 50

# Show detailed experiment information
python scripts/experiment_manager.py show exp_20241227_143022

# Compare multiple experiments
python scripts/experiment_manager.py compare exp_20241227_143022 exp_20241227_145156

# Export experiment data
python scripts/experiment_manager.py export backup_20241227.json

# Generate revert configuration
python scripts/experiment_manager.py revert exp_20241227_143022 -o revert_config.json
```

## Integration

The enhanced tracking system integrates seamlessly with existing infrastructure:

### Automatic Capture
- Hooks into `ConvFinQAMultiAgentPredictor` for automatic tracking
- Captures configuration during predictor initialization
- Records performance metrics during benchmark execution

### Backward Compatibility
- Works alongside existing systems and preserves existing result formats
- Extends rather than replaces current infrastructure

### Minimal Overhead
- Asynchronous database operations and efficient configuration fingerprinting
- Minimal impact on benchmark execution time

## Best Practices

### Experiment Documentation
- Use descriptive notes for each experiment
- Document configuration changes and hypotheses
- Tag experiments with version information

### Configuration Management
- Use configuration hashes to detect duplicates
- Version control configuration files
- Maintain baseline configurations for comparison

### Performance Analysis
- Run multiple experiments for statistical significance
- Compare similar configurations systematically
- Track performance trends over time

## Troubleshooting

### Common Issues

#### Database Lock Errors
```bash
# Check database file permissions
ls -la experiment_tracking/configurations/experiments.db

# Ensure no multiple processes accessing database
ps aux | grep benchmark_multi_agent
```


#### Configuration Capture Failures
- Verify agent configurations are properly initialized
- Check that all required configuration fields are present
- Ensure predictor is created before snapshot capture

## Future Enhancements

### Planned Features
- **Cloud Storage Integration**: AWS S3/Google Cloud storage options
- **Advanced Analytics**: Machine learning-based performance prediction
- **Real-time Monitoring**: Live experiment progress tracking
- **Integration APIs**: RESTful API for external tool integration

### Extension Points
- Custom performance metrics and additional storage backends
- Enhanced visualization options and integration with external tools

## Related Documentation

- [Multi-Agent Approach](MULTI_AGENT_APPROACH.md) - Core multi-agent implementation
- [Code Structure](CODE_STRUCTURE.md) - Overall codebase organization
- [CrewAI Setup](CREWAI_SETUP.md) - Agent configuration and setup

The enhanced tracking system provides comprehensive insight into multi-agent experiment performance while maintaining simplicity and reliability for daily use. 