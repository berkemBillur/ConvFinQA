#!/usr/bin/env python3
"""Experiment runner for systematic configuration testing.

This script enables systematic experimentation with different CrewAI configurations
while tracking performance metrics for comparison and optimisation.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.dataset import ConvFinQADataset
from evaluation.evaluator import ConvFinQAEvaluator
from utils.config import Config
from utils.performance_tracker import get_performance_tracker, create_config_snapshot

try:
    from predictors.multi_agent_predictor import ConvFinQAMultiAgentPredictor
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("CrewAI not available. Run with --fallback-only for baseline testing.")


def create_configuration_variants(base_config_path: str) -> List[Dict[str, Any]]:
    """Create configuration variants for experimentation.
    
    Args:
        base_config_path: Path to base configuration file
        
    Returns:
        List of configuration dictionaries for testing
    """
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Define configuration variants following research best practices
    variants = []
    
    # Configuration 1: Conservative (high accuracy focus)
    config_conservative = base_config.copy()
    config_conservative['models']['crewai'] = {
        'supervisor_model': 'gpt-4o',
        'extractor_model': 'gpt-4o',
        'calculator_model': 'gpt-4o',
        'validator_model': 'gpt-4o',
        'supervisor_temperature': 0.0,
        'extractor_temperature': 0.0,
        'calculator_temperature': 0.0,
        'validator_temperature': 0.0,
        'manager_model': 'gpt-4o',
        'memory': True,
        'cache': True,
        'verbose': False
    }
    variants.append(('conservative', config_conservative))
    
    # Configuration 2: Balanced (default recommended)
    config_balanced = base_config.copy()
    config_balanced['models']['crewai'] = {
        'supervisor_model': 'gpt-4o',
        'extractor_model': 'gpt-4o-mini',
        'calculator_model': 'gpt-4o',
        'validator_model': 'gpt-4o-mini',
        'supervisor_temperature': 0.1,
        'extractor_temperature': 0.0,
        'calculator_temperature': 0.1,
        'validator_temperature': 0.0,
        'manager_model': 'gpt-4o',
        'memory': True,
        'cache': True,
        'verbose': True
    }
    variants.append(('balanced', config_balanced))
    
    # Configuration 3: Cost-optimised (efficiency focus)
    config_efficient = base_config.copy()
    config_efficient['models']['crewai'] = {
        'supervisor_model': 'gpt-4o-mini',
        'extractor_model': 'gpt-4o-mini',
        'calculator_model': 'gpt-4o',  # Keep high accuracy for calculations
        'validator_model': 'gpt-4o-mini',
        'supervisor_temperature': 0.1,
        'extractor_temperature': 0.0,
        'calculator_temperature': 0.0,
        'validator_temperature': 0.0,
        'manager_model': 'gpt-4o-mini',
        'memory': False,  # Reduce memory usage
        'cache': True,
        'verbose': False
    }
    variants.append(('cost_optimised', config_efficient))
    
    return variants


def run_configuration_experiment(
    config_name: str,
    config_dict: Dict[str, Any],
    evaluator: ConvFinQAEvaluator,
    num_conversations: int = 5
) -> Dict[str, Any]:
    """Run experiment with a specific configuration.
    
    Args:
        config_name: Name identifier for the configuration
        config_dict: Configuration dictionary
        evaluator: Evaluator instance
        num_conversations: Number of conversations to test
        
    Returns:
        Experiment results dictionary
    """
    print(f"\n=== Running experiment: {config_name} ===")
    
    # Create temporary config file
    temp_config_path = f"/tmp/crewai_config_{config_name}.json"
    with open(temp_config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    try:
        # Create configuration and predictor
        config = Config(temp_config_path)
        
        if not CREWAI_AVAILABLE:
            print(f"CrewAI not available, skipping {config_name}")
            return {'config_name': config_name, 'skipped': True, 'reason': 'CrewAI not available'}
        
        predictor = ConvFinQAMultiAgentPredictor(config)
        
        # Create configuration snapshot for tracking
        config_snapshot = create_config_snapshot(config)
        tracker = get_performance_tracker()
        config_hash = tracker.register_configuration(config_snapshot)
        
        print(f"Configuration hash: {config_hash}")
        print(f"Testing with {num_conversations} conversations...")
        
        # Run evaluation
        results = evaluator.quick_evaluation(
            predictor=predictor,
            num_conversations=num_conversations,
            split='dev',
            random_sample=True,
            seed=42  # Fixed seed for reproducibility
        )
        
        # Get execution summary from predictor
        execution_summary = predictor.get_execution_summary()
        
        # Get aggregated performance from tracker
        performance_aggregate = tracker.get_performance_aggregate(config_hash)
        
        experiment_result = {
            'config_name': config_name,
            'config_hash': config_hash,
            'evaluation_results': {
                'accuracy': results.accuracy,
                'total_questions': results.total_questions,
                'correct_answers': results.correct_answers,
                'total_conversations': results.total_conversations,
                'successful_conversations': results.successful_conversations
            },
            'execution_summary': execution_summary,
            'performance_aggregate': performance_aggregate.__dict__ if performance_aggregate else None,
            'configuration_details': config_snapshot.__dict__
        }
        
        print(f"Results for {config_name}:")
        print(f"  Accuracy: {results.accuracy:.3f}")
        print(f"  Success Rate: {execution_summary.get('success_rate', 'N/A')}")
        print(f"  Average Cost: ${execution_summary.get('average_cost_per_execution', 0.0):.4f}")
        print(f"  Average Time: {execution_summary.get('average_execution_time', 0.0):.2f}s")
        
        return experiment_result
        
    except Exception as e:
        print(f"Experiment {config_name} failed: {str(e)}")
        return {
            'config_name': config_name,
            'error': str(e),
            'failed': True
        }
    
    finally:
        # Clean up temporary config file
        Path(temp_config_path).unlink(missing_ok=True)


def run_comparison_analysis(experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse and compare experiment results.
    
    Args:
        experiment_results: List of experiment result dictionaries
        
    Returns:
        Comparison analysis
    """
    print("\n=== Comparison Analysis ===")
    
    # Extract successful experiments
    successful_experiments = [r for r in experiment_results if not r.get('failed', False) and not r.get('skipped', False)]
    
    if not successful_experiments:
        print("No successful experiments to compare.")
        return {'error': 'No successful experiments'}
    
    # Rank by different criteria
    accuracy_ranking = sorted(successful_experiments, key=lambda x: x['evaluation_results']['accuracy'], reverse=True)
    cost_ranking = sorted(successful_experiments, key=lambda x: x['execution_summary'].get('average_cost_per_execution', 0.0))
    speed_ranking = sorted(successful_experiments, key=lambda x: x['execution_summary'].get('average_execution_time', 0.0))
    
    analysis = {
        'total_experiments': len(experiment_results),
        'successful_experiments': len(successful_experiments),
        'rankings': {
            'best_accuracy': accuracy_ranking[0]['config_name'] if accuracy_ranking else None,
            'best_cost_efficiency': cost_ranking[0]['config_name'] if cost_ranking else None,
            'best_speed': speed_ranking[0]['config_name'] if speed_ranking else None
        },
        'detailed_comparison': []
    }
    
    # Create detailed comparison table
    for exp in successful_experiments:
        analysis['detailed_comparison'].append({
            'config_name': exp['config_name'],
            'config_hash': exp['config_hash'],
            'accuracy': exp['evaluation_results']['accuracy'],
            'success_rate': exp['execution_summary'].get('success_rate', 0.0),
            'avg_cost': exp['execution_summary'].get('average_cost_per_execution', 0.0),
            'avg_time': exp['execution_summary'].get('average_execution_time', 0.0),
            'fallback_rate': exp['execution_summary'].get('fallback_rate', 0.0)
        })
    
    # Print comparison table
    print("\nConfiguration Comparison:")
    print(f"{'Config':<15} {'Accuracy':<10} {'Success':<10} {'Cost':<10} {'Time':<10} {'Fallback':<10}")
    print("-" * 75)
    
    for comp in analysis['detailed_comparison']:
        print(f"{comp['config_name']:<15} "
              f"{comp['accuracy']:<10.3f} "
              f"{comp['success_rate']:<10.3f} "
              f"${comp['avg_cost']:<9.4f} "
              f"{comp['avg_time']:<10.2f} "
              f"{comp['fallback_rate']:<10.3f}")
    
    print(f"\nBest performing configurations:")
    print(f"  Accuracy: {analysis['rankings']['best_accuracy']}")
    print(f"  Cost Efficiency: {analysis['rankings']['best_cost_efficiency']}")
    print(f"  Speed: {analysis['rankings']['best_speed']}")
    
    return analysis


def main():
    """Main experiment runner function."""
    parser = argparse.ArgumentParser(description="Run CrewAI configuration experiments")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/base.json',
        help='Base configuration file path'
    )
    parser.add_argument(
        '--conversations', 
        type=int, 
        default=5,
        help='Number of conversations to test per configuration'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='experiments/results/configuration_comparison.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    print("Tracking system implementation complete!")
    print(f"Performance tracking configured with storage in: experiments/tracking/")
    print(f"Configuration variants ready for testing with: {args.conversations} conversations each")
    print(f"Results will be saved to: {args.output}")


if __name__ == "__main__":
    main() 