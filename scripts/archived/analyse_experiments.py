#!/usr/bin/env python3
"""
Experiment analysis script for CrewAI configuration optimisation.

Provides comprehensive analysis of experiment results, configuration impact,
and optimisation recommendations for CrewAI multi-agent systems.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from utils.experiment_tracking import ExperimentTracker
from evaluation.performance_analysis import ConfigurationOptimiser, ABTestingFramework


def analyse_experiment_performance(config: Config, experiment_id: str = None) -> None:
    """
    Analyse performance of experiments.
    
    Args:
        config: Project configuration
        experiment_id: Specific experiment to analyse (optional)
    """
    tracker = ExperimentTracker(
        db_path=os.path.join(config.experiments.results_dir, "experiment_tracking.db")
    )
    
    print("=== CrewAI Experiment Analysis ===\n")
    
    # Get experiments summary
    experiments = tracker.get_experiments_summary()
    
    if not experiments:
        print("No experiments found in tracking database.")
        return
    
    # Display experiment summary
    print(f"Found {len(experiments)} experiments:")
    print("-" * 80)
    print(f"{'Experiment ID':<20} {'Name':<25} {'Success Rate':<15} {'Avg Time':<10}")
    print("-" * 80)
    
    for exp in experiments:
        success_rate = exp.get('success_rate', 0) or 0
        avg_time = exp.get('avg_execution_time', 0) or 0
        print(f"{exp['experiment_id']:<20} {exp.get('experiment_name', 'Unnamed'):<25} "
              f"{success_rate:.2%}<15} {avg_time:.2f}s<10}")
    
    print("-" * 80)
    
    # Detailed analysis for specific experiment
    if experiment_id:
        analyse_specific_experiment(tracker, experiment_id)
    else:
        # Analyse best performing experiment
        best_exp = max(experiments, key=lambda x: x.get('success_rate', 0) or 0)
        analyse_specific_experiment(tracker, best_exp['experiment_id'])


def analyse_specific_experiment(tracker: ExperimentTracker, experiment_id: str) -> None:
    """
    Provide detailed analysis of a specific experiment.
    
    Args:
        tracker: Experiment tracker
        experiment_id: Experiment to analyse
    """
    print(f"\n=== Detailed Analysis: {experiment_id} ===")
    
    # Get performance metrics
    metrics = tracker.calculate_performance_metrics(experiment_id)
    
    if not metrics:
        print("No performance data found for this experiment.")
        return
    
    print(f"Total Executions: {metrics.get('total_executions', 0)}")
    print(f"Successful Executions: {metrics.get('successful_executions', 0)}")
    print(f"Success Rate: {metrics.get('success_rate', 0):.2%}")
    print(f"Error Rate: {metrics.get('error_rate', 0):.2%}")
    print(f"Average Execution Time: {metrics.get('average_execution_time', 0):.2f}s")
    print(f"Average Cost: ${metrics.get('average_cost', 0):.4f}")
    print(f"Average Confidence: {metrics.get('average_confidence', 0):.2f}")


def generate_optimisation_recommendations(config: Config) -> None:
    """
    Generate configuration optimisation recommendations.
    
    Args:
        config: Project configuration
    """
    print("\n=== Configuration Optimisation Recommendations ===\n")
    
    tracker = ExperimentTracker(
        db_path=os.path.join(config.experiments.results_dir, "experiment_tracking.db")
    )
    
    optimiser = ConfigurationOptimiser(tracker)
    
    # Analyse key parameters
    key_parameters = ['temperature', 'model_name', 'memory', 'cache']
    
    print("Parameter Impact Analysis:")
    print("-" * 60)
    print(f"{'Parameter':<20} {'Correlation':<15} {'Significance':<15} {'Recommended':<15}")
    print("-" * 60)
    
    for param in key_parameters:
        analysis = optimiser.analyse_parameter_impact(param)
        
        if analysis.sample_size >= 2:
            print(f"{param:<20} {analysis.performance_correlation:<15.3f} "
                  f"{analysis.statistical_significance:<15.3f} {str(analysis.recommended_value):<15}")
    
    print("-" * 60)


def compare_experiments(config: Config, experiment_ids: List[str]) -> None:
    """
    Compare multiple experiments.
    
    Args:
        config: Project configuration
        experiment_ids: List of experiment IDs to compare
    """
    print(f"\n=== Comparing {len(experiment_ids)} Experiments ===\n")
    
    tracker = ExperimentTracker(
        db_path=os.path.join(config.experiments.results_dir, "experiment_tracking.db")
    )
    
    optimiser = ConfigurationOptimiser(tracker)
    
    # Generate comparison report
    report = optimiser.create_performance_report(experiment_ids)
    
    print(f"Summary Statistics:")
    print(f"  Total Experiments: {report['summary'].get('total_experiments', 0)}")
    print(f"  Average Success Rate: {report['summary'].get('avg_success_rate', 0):.2%}")
    print(f"  Best Success Rate: {report['summary'].get('best_success_rate', 0):.2%}")
    print(f"  Average Execution Time: {report['summary'].get('avg_execution_time', 0):.2f}s")
    
    print(f"\nBest Configuration:")
    best = report.get('best_configuration', {})
    print(f"  Experiment ID: {best.get('experiment_id', 'Unknown')}")
    print(f"  Name: {best.get('experiment_name', 'Unnamed')}")
    print(f"  Success Rate: {best.get('success_rate', 0):.2%}")
    
    print(f"\nParameter Analysis:")
    for param, analysis in report.get('parameter_analysis', {}).items():
        print(f"  {param}: correlation={analysis['correlation']:.3f}, "
              f"recommended={analysis['recommended_value']}")


def export_experiment_data(config: Config, output_file: str) -> None:
    """
    Export experiment data to JSON file.
    
    Args:
        config: Project configuration
        output_file: Output file path
    """
    tracker = ExperimentTracker(
        db_path=os.path.join(config.experiments.results_dir, "experiment_tracking.db")
    )
    
    # Get all experiment data
    experiments = tracker.get_experiments_summary()
    
    export_data = {
        'export_timestamp': pd.Timestamp.now().isoformat(),
        'total_experiments': len(experiments),
        'experiments': experiments
    }
    
    # Add detailed metrics for each experiment
    for exp in export_data['experiments']:
        metrics = tracker.calculate_performance_metrics(exp['experiment_id'])
        exp['detailed_metrics'] = metrics
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"Exported {len(experiments)} experiments to {output_file}")


def main():
    """Main analysis script entry point."""
    parser = argparse.ArgumentParser(description="Analyse CrewAI experiments")
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--experiment-id', '-e', help='Specific experiment to analyse')
    parser.add_argument('--compare', nargs='+', help='Compare multiple experiments')
    parser.add_argument('--optimise', action='store_true', help='Generate optimisation recommendations')
    parser.add_argument('--export', help='Export data to JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    try:
        if args.compare:
            compare_experiments(config, args.compare)
        elif args.optimise:
            generate_optimisation_recommendations(config)
        elif args.export:
            export_experiment_data(config, args.export)
        else:
            analyse_experiment_performance(config, args.experiment_id)
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
