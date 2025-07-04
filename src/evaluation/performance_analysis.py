"""
Performance analysis and configuration optimisation for CrewAI experiments.

Provides tools for analysing experiment results, correlating configuration 
parameters with performance, and generating optimisation recommendations.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import json

from utils.experiment_tracking import ExperimentTracker, CrewConfiguration


@dataclass
class ConfigurationImpactAnalysis:
    """Analysis of configuration parameter impact on performance."""
    
    parameter_name: str
    parameter_values: List[Any]
    performance_correlation: float
    statistical_significance: float
    recommended_value: Any
    confidence_interval: Tuple[float, float]
    sample_size: int
    analysis_timestamp: str


@dataclass
class OptimisationRecommendation:
    """Recommendation for configuration optimisation."""
    
    parameter_name: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence: float
    rationale: str
    priority: str  # high, medium, low


class ConfigurationOptimiser:
    """Optimises CrewAI configurations based on performance data."""
    
    def __init__(self, experiment_tracker: ExperimentTracker):
        """
        Initialise configuration optimiser.
        
        Args:
            experiment_tracker: Experiment tracking system
        """
        self.tracker = experiment_tracker
        self.logger = logging.getLogger(__name__)
    
    def analyse_parameter_impact(self, parameter_name: str, 
                               performance_metric: str = 'success_rate') -> ConfigurationImpactAnalysis:
        """
        Analyse the impact of a configuration parameter on performance.
        
        Args:
            parameter_name: Name of parameter to analyse (e.g., 'temperature', 'model_name')
            performance_metric: Performance metric to correlate with
            
        Returns:
            Impact analysis results
        """
        # Get experiment data
        experiments = self.tracker.get_experiments_summary()
        
        if not experiments:
            self.logger.warning("No experiments found for analysis")
            return ConfigurationImpactAnalysis(
                parameter_name=parameter_name,
                parameter_values=[],
                performance_correlation=0.0,
                statistical_significance=0.0,
                recommended_value=None,
                confidence_interval=(0.0, 0.0),
                sample_size=0,
                analysis_timestamp=pd.Timestamp.now().isoformat()
            )
        
        # Extract parameter values and performance data
        parameter_values = []
        performance_values = []
        
        for exp in experiments:
            # Load configuration
            config = self._load_experiment_config(exp['experiment_id'])
            if config:
                param_value = self._extract_parameter_value(config, parameter_name)
                if param_value is not None:
                    parameter_values.append(param_value)
                    performance_values.append(exp.get(performance_metric, 0))
        
        if len(parameter_values) < 2:
            self.logger.warning(f"Insufficient data points for parameter {parameter_name}")
            return ConfigurationImpactAnalysis(
                parameter_name=parameter_name,
                parameter_values=parameter_values,
                performance_correlation=0.0,
                statistical_significance=0.0,
                recommended_value=parameter_values[0] if parameter_values else None,
                confidence_interval=(0.0, 0.0),
                sample_size=len(parameter_values),
                analysis_timestamp=pd.Timestamp.now().isoformat()
            )
        
        # Calculate correlation and significance
        try:
            # Convert to numeric if possible
            numeric_params = []
            numeric_performance = []
            
            for i, param in enumerate(parameter_values):
                try:
                    numeric_params.append(float(param))
                    numeric_performance.append(float(performance_values[i]))
                except (ValueError, TypeError):
                    pass
            
            if len(numeric_params) >= 2:
                correlation = np.corrcoef(numeric_params, numeric_performance)[0, 1]
                
                # Simple significance test (would use scipy.stats in production)
                n = len(numeric_params)
                t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
                significance = abs(t_stat) / np.sqrt(n - 2)  # Simplified
                
                # Find optimal value
                best_idx = np.argmax(numeric_performance)
                recommended_value = numeric_params[best_idx]
                
                # Calculate confidence interval (simplified)
                std_error = np.std(numeric_performance) / np.sqrt(n)
                conf_interval = (
                    max(numeric_performance) - 1.96 * std_error,
                    max(numeric_performance) + 1.96 * std_error
                )
            else:
                # Categorical parameter analysis
                correlation = 0.0
                significance = 0.0
                
                # Find most frequent value among best performers
                perf_threshold = np.percentile(performance_values, 75)
                best_params = [parameter_values[i] for i, perf in enumerate(performance_values) 
                              if perf >= perf_threshold]
                
                if best_params:
                    # Most common value among top performers
                    from collections import Counter
                    recommended_value = Counter(best_params).most_common(1)[0][0]
                else:
                    recommended_value = parameter_values[0]
                
                conf_interval = (0.0, 1.0)
                
        except Exception as e:
            self.logger.error(f"Error calculating correlation for {parameter_name}: {e}")
            correlation = 0.0
            significance = 0.0
            recommended_value = parameter_values[0]
            conf_interval = (0.0, 0.0)
        
        return ConfigurationImpactAnalysis(
            parameter_name=parameter_name,
            parameter_values=list(set(parameter_values)),
            performance_correlation=correlation,
            statistical_significance=significance,
            recommended_value=recommended_value,
            confidence_interval=conf_interval,
            sample_size=len(parameter_values),
            analysis_timestamp=pd.Timestamp.now().isoformat()
        )
    
    def generate_optimisation_recommendations(self, 
                                            current_config: CrewConfiguration,
                                            target_metric: str = 'success_rate') -> List[OptimisationRecommendation]:
        """
        Generate optimisation recommendations for a configuration.
        
        Args:
            current_config: Current crew configuration
            target_metric: Performance metric to optimise for
            
        Returns:
            List of optimisation recommendations
        """
        recommendations = []
        
        # Key parameters to analyse
        key_parameters = [
            'temperature',
            'model_name', 
            'allow_delegation',
            'memory',
            'cache',
            'process_type'
        ]
        
        for param in key_parameters:
            analysis = self.analyse_parameter_impact(param, target_metric)
            
            if analysis.sample_size >= 2:
                current_value = self._extract_parameter_value(current_config, param)
                
                if (current_value != analysis.recommended_value and 
                    analysis.statistical_significance > 0.1):  # Basic threshold
                    
                    # Calculate expected improvement
                    expected_improvement = abs(analysis.performance_correlation) * 0.1  # Simplified
                    
                    # Determine priority
                    if analysis.statistical_significance > 0.5 and expected_improvement > 0.05:
                        priority = "high"
                    elif analysis.statistical_significance > 0.3 and expected_improvement > 0.02:
                        priority = "medium"
                    else:
                        priority = "low"
                    
                    recommendations.append(OptimisationRecommendation(
                        parameter_name=param,
                        current_value=current_value,
                        recommended_value=analysis.recommended_value,
                        expected_improvement=expected_improvement,
                        confidence=analysis.statistical_significance,
                        rationale=f"Analysis of {analysis.sample_size} experiments shows correlation of {analysis.performance_correlation:.3f}",
                        priority=priority
                    ))
        
        # Sort by priority and expected improvement
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(key=lambda x: (priority_order[x.priority], x.expected_improvement), reverse=True)
        
        return recommendations
    
    def _load_experiment_config(self, experiment_id: str) -> Optional[CrewConfiguration]:
        """Load crew configuration for experiment."""
        import sqlite3
        
        with sqlite3.connect(self.tracker.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT config_data FROM crew_configurations WHERE experiment_id = ?', 
                          (experiment_id,))
            result = cursor.fetchone()
            
            if result:
                try:
                    return CrewConfiguration.parse_raw(result[0])
                except Exception as e:
                    self.logger.error(f"Error parsing config for {experiment_id}: {e}")
        
        return None
    
    def _extract_parameter_value(self, config: CrewConfiguration, parameter_name: str) -> Any:
        """Extract parameter value from configuration."""
        # Handle crew-level parameters
        if hasattr(config, parameter_name):
            return getattr(config, parameter_name)
        
        # Handle agent-level parameters (take from first agent as representative)
        if config.agents and hasattr(config.agents[0], parameter_name):
            return getattr(config.agents[0], parameter_name)
        
        # Handle nested parameters
        if parameter_name in ['model_name']:
            if config.agents:
                return config.agents[0].model_name
        
        return None
    
    def create_performance_report(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Create comprehensive performance report for experiments.
        
        Args:
            experiment_ids: List of experiment IDs to include
            
        Returns:
            Performance report
        """
        report = {
            'summary': {},
            'comparisons': {},
            'recommendations': {},
            'parameter_analysis': {}
        }
        
        # Get experiment summaries
        all_experiments = self.tracker.get_experiments_summary()
        selected_experiments = [exp for exp in all_experiments 
                              if exp['experiment_id'] in experiment_ids]
        
        if not selected_experiments:
            return report
        
        # Summary statistics
        success_rates = [exp.get('success_rate', 0) for exp in selected_experiments]
        execution_times = [exp.get('avg_execution_time', 0) for exp in selected_experiments]
        
        report['summary'] = {
            'total_experiments': len(selected_experiments),
            'avg_success_rate': np.mean(success_rates),
            'best_success_rate': max(success_rates),
            'avg_execution_time': np.mean(execution_times),
            'fastest_execution_time': min(execution_times)
        }
        
        # Best performing experiment
        best_exp = max(selected_experiments, key=lambda x: x.get('success_rate', 0))
        report['best_configuration'] = {
            'experiment_id': best_exp['experiment_id'],
            'experiment_name': best_exp.get('experiment_name', 'Unnamed'),
            'success_rate': best_exp.get('success_rate', 0),
            'avg_execution_time': best_exp.get('avg_execution_time', 0)
        }
        
        # Parameter impact analysis
        key_params = ['temperature', 'model_name', 'memory', 'cache']
        for param in key_params:
            analysis = self.analyse_parameter_impact(param)
            if analysis.sample_size >= 2:
                report['parameter_analysis'][param] = {
                    'correlation': analysis.performance_correlation,
                    'significance': analysis.statistical_significance,
                    'recommended_value': analysis.recommended_value,
                    'sample_size': analysis.sample_size
                }
        
        return report


class ABTestingFramework:
    """Framework for A/B testing CrewAI configurations."""
    
    def __init__(self, experiment_tracker: ExperimentTracker):
        """
        Initialise A/B testing framework.
        
        Args:
            experiment_tracker: Experiment tracking system
        """
        self.tracker = experiment_tracker
        self.logger = logging.getLogger(__name__)
    
    def create_ab_test(self, baseline_config: CrewConfiguration,
                      variant_configs: List[CrewConfiguration],
                      test_name: str) -> str:
        """
        Create A/B test comparing baseline with variants.
        
        Args:
            baseline_config: Baseline configuration
            variant_configs: List of variant configurations to test
            test_name: Name for the A/B test
            
        Returns:
            Test identifier
        """
        # Save all configurations
        test_id = f"abtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Update experiment names to indicate A/B test
        baseline_config.experiment_name = f"{test_name}_baseline"
        baseline_config.experiment_description = f"A/B test baseline for {test_name}"
        
        configs_to_track = [baseline_config]
        
        for i, variant in enumerate(variant_configs):
            variant.experiment_name = f"{test_name}_variant_{i+1}"
            variant.experiment_description = f"A/B test variant {i+1} for {test_name}"
            configs_to_track.append(variant)
        
        # Save configurations
        experiment_ids = []
        for config in configs_to_track:
            exp_id = self.tracker.save_crew_configuration(config)
            experiment_ids.append(exp_id)
        
        self.logger.info(f"Created A/B test {test_id} with {len(configs_to_track)} configurations")
        
        return test_id
    
    def analyse_ab_test_results(self, test_experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Analyse results of A/B test.
        
        Args:
            test_experiment_ids: List of experiment IDs in the test
            
        Returns:
            A/B test analysis results
        """
        # Get performance data for each experiment
        results = {}
        
        for exp_id in test_experiment_ids:
            # Calculate performance metrics for this experiment
            with sqlite3.connect(self.tracker.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT COUNT(*) as total,
                           SUM(CAST(success AS INTEGER)) as successful,
                           AVG(execution_time) as avg_time,
                           AVG(confidence) as avg_confidence
                    FROM execution_logs
                    WHERE experiment_id = ?
                ''', (exp_id,))
                
                result = cursor.fetchone()
                if result and result[0] > 0:
                    results[exp_id] = {
                        'total_executions': result[0],
                        'success_rate': (result[1] or 0) / result[0],
                        'avg_execution_time': result[2] or 0,
                        'avg_confidence': result[3] or 0
                    }
        
        if not results:
            return {'error': 'No execution data found for A/B test'}
        
        # Find best performing variant
        best_exp_id = max(results.keys(), key=lambda x: results[x]['success_rate'])
        
        # Calculate statistical significance (simplified)
        baseline_id = test_experiment_ids[0]  # Assume first is baseline
        baseline_rate = results[baseline_id]['success_rate']
        
        analysis = {
            'baseline_experiment': baseline_id,
            'baseline_success_rate': baseline_rate,
            'best_experiment': best_exp_id,
            'best_success_rate': results[best_exp_id]['success_rate'],
            'improvement': results[best_exp_id]['success_rate'] - baseline_rate,
            'all_results': results
        }
        
        return analysis
