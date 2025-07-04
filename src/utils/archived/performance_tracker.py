"""Performance tracking system for ConvFinQA predictors.

This module provides configuration fingerprinting and performance correlation
capabilities for systematic experimentation with predictors, particularly
the CrewAI multi-agent implementation.
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

from .config import Config


@dataclass
class CrewConfigSnapshot:
    """Configuration snapshot for CrewAI predictor tracking.
    
    Captures essential configuration parameters that impact performance,
    following the principle of simple but effective tracking.
    """
    
    # Agent Models (primary performance impact factor)
    supervisor_model: str
    extractor_model: str
    calculator_model: str
    validator_model: str
    
    # Agent Temperatures (consistency vs creativity balance)
    supervisor_temp: float
    extractor_temp: float
    calculator_temp: float
    validator_temp: float
    
    # Crew Coordination Settings
    process_type: str
    manager_model: str
    memory_enabled: bool
    cache_enabled: bool
    
    # Execution Settings
    verbose: bool
    max_execution_time: Optional[int] = None
    
    # Prompt versioning integration
    prompt_hash: Optional[str] = None
    
    # Metadata (auto-generated)
    config_hash: str = field(init=False)
    timestamp: str = field(init=False)
    
    def __post_init__(self):
        """Generate deterministic hash and timestamp for this configuration."""
        # Create deterministic string representation
        config_str = (
            f"{self.supervisor_model}_{self.supervisor_temp}_"
            f"{self.extractor_model}_{self.extractor_temp}_"
            f"{self.calculator_model}_{self.calculator_temp}_"
            f"{self.validator_model}_{self.validator_temp}_"
            f"{self.process_type}_{self.manager_model}_"
            f"{self.memory_enabled}_{self.cache_enabled}_{self.verbose}"
        )
        
        # Generate short hash for easy identification
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        self.timestamp = datetime.now().isoformat()
    
    @classmethod
    def from_config(cls, config: Config) -> 'CrewConfigSnapshot':
        """Create configuration snapshot from Config instance.
        
        Args:
            config: Configuration instance containing CrewAI settings
            
        Returns:
            Configuration snapshot for tracking
        """
        crewai_config = config.get('models.crewai', {})
        
        return cls(
            supervisor_model=crewai_config.get('supervisor_model', 'gpt-4o'),
            extractor_model=crewai_config.get('extractor_model', 'gpt-4o-mini'),
            calculator_model=crewai_config.get('calculator_model', 'gpt-4o'),
            validator_model=crewai_config.get('validator_model', 'gpt-4o-mini'),
            supervisor_temp=crewai_config.get('supervisor_temperature', 0.1),
            extractor_temp=crewai_config.get('extractor_temperature', 0.0),
            calculator_temp=crewai_config.get('calculator_temperature', 0.1),
            validator_temp=crewai_config.get('validator_temperature', 0.0),
            process_type='hierarchical',  # Currently fixed in implementation
            manager_model=crewai_config.get('manager_model', 'gpt-4o'),
            memory_enabled=crewai_config.get('memory', True),
            cache_enabled=crewai_config.get('cache', True),
            verbose=crewai_config.get('verbose', True),
            max_execution_time=crewai_config.get('max_execution_time')
        )


@dataclass
class ExecutionRecord:
    """Single execution record for performance tracking."""
    
    config_hash: str
    record_id: str
    success: bool
    execution_time: float
    estimated_cost: float
    confidence: float
    fallback_used: bool
    error_message: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Optional metadata
    question_type: Optional[str] = None
    conversation_length: Optional[int] = None


@dataclass
class PerformanceAggregate:
    """Aggregated performance metrics for a configuration."""
    
    config_hash: str
    
    # Execution Statistics
    total_executions: int
    successful_executions: int
    success_rate: float
    
    # Timing Statistics
    average_execution_time: float
    median_execution_time: float
    
    # Cost Statistics
    total_cost: float
    average_cost_per_execution: float
    
    # Quality Statistics
    average_confidence: float
    fallback_rate: float
    
    # Metadata
    first_execution: str
    last_execution: str
    
    @classmethod
    def from_executions(cls, config_hash: str, executions: List[ExecutionRecord]) -> 'PerformanceAggregate':
        """Create aggregate from list of execution records.
        
        Args:
            config_hash: Configuration hash for these executions
            executions: List of execution records to aggregate
            
        Returns:
            Aggregated performance metrics
        """
        if not executions:
            # Return empty aggregate for no executions
            return cls(
                config_hash=config_hash,
                total_executions=0,
                successful_executions=0,
                success_rate=0.0,
                average_execution_time=0.0,
                median_execution_time=0.0,
                total_cost=0.0,
                average_cost_per_execution=0.0,
                average_confidence=0.0,
                fallback_rate=0.0,
                first_execution=datetime.now().isoformat(),
                last_execution=datetime.now().isoformat()
            )
        
        total_executions = len(executions)
        successful_executions = sum(1 for e in executions if e.success)
        success_rate = successful_executions / total_executions
        
        execution_times = [e.execution_time for e in executions]
        average_execution_time = sum(execution_times) / len(execution_times)
        median_execution_time = sorted(execution_times)[len(execution_times) // 2]
        
        total_cost = sum(e.estimated_cost for e in executions)
        average_cost_per_execution = total_cost / total_executions
        
        successful_confidences = [e.confidence for e in executions if e.success]
        average_confidence = sum(successful_confidences) / len(successful_confidences) if successful_confidences else 0.0
        
        fallback_rate = sum(1 for e in executions if e.fallback_used) / total_executions
        
        timestamps = [e.timestamp for e in executions]
        first_execution = min(timestamps)
        last_execution = max(timestamps)
        
        return cls(
            config_hash=config_hash,
            total_executions=total_executions,
            successful_executions=successful_executions,
            success_rate=success_rate,
            average_execution_time=average_execution_time,
            median_execution_time=median_execution_time,
            total_cost=total_cost,
            average_cost_per_execution=average_cost_per_execution,
            average_confidence=average_confidence,
            fallback_rate=fallback_rate,
            first_execution=first_execution,
            last_execution=last_execution
        )


class PerformanceTracker:
    """Simple file-based performance tracking system.
    
    Stores execution records and configuration snapshots in JSON format
    for easy version control and analysis integration.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialise performance tracker.
        
        Args:
            storage_dir: Directory for storing tracking data. 
                        Defaults to experiments/tracking/
        """
        if storage_dir is None:
            # Use existing experiments directory pattern
            project_root = Path(__file__).parent.parent.parent
            self.storage_dir = project_root / "experiments" / "tracking"
        else:
            self.storage_dir = Path(storage_dir)
        
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate files for different data types following single responsibility
        self.executions_file = self.storage_dir / "executions.json"
        self.configurations_file = self.storage_dir / "configurations.json"
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure storage files exist
        self._ensure_storage_files_exist()
    
    def _ensure_storage_files_exist(self) -> None:
        """Ensure storage files exist with proper structure."""
        if not self.executions_file.exists():
            with open(self.executions_file, 'w') as f:
                json.dump([], f)
        
        if not self.configurations_file.exists():
            with open(self.configurations_file, 'w') as f:
                json.dump({}, f)
    
    def register_configuration(self, config_snapshot: CrewConfigSnapshot) -> str:
        """Register a configuration snapshot for tracking.
        
        Args:
            config_snapshot: Configuration to register
            
        Returns:
            Configuration hash for future reference
        """
        configurations = self._load_configurations()
        
        config_hash = config_snapshot.config_hash
        if config_hash not in configurations:
            configurations[config_hash] = asdict(config_snapshot)
            self._save_configurations(configurations)
            self.logger.info(f"Registered new configuration: {config_hash}")
        
        return config_hash
    
    def log_execution(self, 
                     config_hash: str,
                     record_id: str,
                     success: bool,
                     execution_time: float,
                     estimated_cost: float = 0.0,
                     confidence: float = 0.0,
                     fallback_used: bool = False,
                     error_message: Optional[str] = None,
                     **metadata) -> None:
        """Log a single execution record.
        
        Args:
            config_hash: Hash of configuration used
            record_id: ID of the record being processed
            success: Whether execution was successful
            execution_time: Time taken for execution (seconds)
            estimated_cost: Estimated API cost for execution
            confidence: Confidence score if available
            fallback_used: Whether fallback mechanism was used
            error_message: Error message if execution failed
            **metadata: Additional metadata (e.g., question_type)
        """
        execution_record = ExecutionRecord(
            config_hash=config_hash,
            record_id=record_id,
            success=success,
            execution_time=execution_time,
            estimated_cost=estimated_cost,
            confidence=confidence,
            fallback_used=fallback_used,
            error_message=error_message,
            **metadata
        )
        
        executions = self._load_executions()
        executions.append(asdict(execution_record))
        self._save_executions(executions)
    
    def get_performance_aggregate(self, config_hash: str) -> Optional[PerformanceAggregate]:
        """Get aggregated performance metrics for a configuration.
        
        Args:
            config_hash: Configuration hash to analyse
            
        Returns:
            Aggregated performance metrics or None if no data exists
        """
        executions = self._load_executions()
        config_executions = [
            ExecutionRecord(**e) for e in executions 
            if e['config_hash'] == config_hash
        ]
        
        if not config_executions:
            return None
        
        return PerformanceAggregate.from_executions(config_hash, config_executions)
    
    def get_configuration_snapshot(self, config_hash: str) -> Optional[CrewConfigSnapshot]:
        """Get configuration snapshot by hash.
        
        Args:
            config_hash: Configuration hash to retrieve
            
        Returns:
            Configuration snapshot or None if not found
        """
        configurations = self._load_configurations()
        
        if config_hash not in configurations:
            return None
        
        return CrewConfigSnapshot(**configurations[config_hash])
    
    def list_configurations(self) -> List[str]:
        """List all registered configuration hashes.
        
        Returns:
            List of configuration hashes
        """
        configurations = self._load_configurations()
        return list(configurations.keys())
    
    def compare_configurations(self, config_hashes: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple configurations.
        
        Args:
            config_hashes: List of configuration hashes to compare
            
        Returns:
            Comparison data structure
        """
        comparison = {
            'configurations': {},
            'performance': {},
            'summary': {
                'best_accuracy': None,
                'best_cost_efficiency': None,
                'best_speed': None
            }
        }
        
        best_accuracy = 0.0
        best_cost_efficiency = float('inf')
        best_speed = float('inf')
        
        for config_hash in config_hashes:
            # Get configuration details
            config_snapshot = self.get_configuration_snapshot(config_hash)
            if config_snapshot:
                comparison['configurations'][config_hash] = asdict(config_snapshot)
            
            # Get performance data
            performance = self.get_performance_aggregate(config_hash)
            if performance:
                comparison['performance'][config_hash] = asdict(performance)
                
                # Track best performers
                if performance.success_rate > best_accuracy:
                    best_accuracy = performance.success_rate
                    comparison['summary']['best_accuracy'] = config_hash
                
                if performance.average_cost_per_execution < best_cost_efficiency:
                    best_cost_efficiency = performance.average_cost_per_execution
                    comparison['summary']['best_cost_efficiency'] = config_hash
                
                if performance.average_execution_time < best_speed:
                    best_speed = performance.average_execution_time
                    comparison['summary']['best_speed'] = config_hash
        
        return comparison
    
    def _load_executions(self) -> List[Dict[str, Any]]:
        """Load executions from storage file."""
        try:
            with open(self.executions_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_executions(self, executions: List[Dict[str, Any]]) -> None:
        """Save executions to storage file."""
        with open(self.executions_file, 'w') as f:
            json.dump(executions, f, indent=2)
    
    def _load_configurations(self) -> Dict[str, Any]:
        """Load configurations from storage file."""
        try:
            with open(self.configurations_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_configurations(self, configurations: Dict[str, Any]) -> None:
        """Save configurations to storage file."""
        with open(self.configurations_file, 'w') as f:
            json.dump(configurations, f, indent=2)


# Module-level convenience functions following existing patterns
_tracker_instance: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker instance.
    
    Returns:
        Global performance tracker instance
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PerformanceTracker()
    return _tracker_instance


def create_config_snapshot(config: Config) -> CrewConfigSnapshot:
    """Convenience function to create configuration snapshot.
    
    Args:
        config: Configuration instance
        
    Returns:
        Configuration snapshot for tracking (now includes prompt versioning)
    """
    # Create base configuration snapshot
    snapshot = CrewConfigSnapshot.from_config(config)
    
    # Capture current prompt version
    try:
        from .prompt_tracker import get_prompt_tracker
        tracker = get_prompt_tracker()
        prompt_snapshot = tracker.capture_current_prompts()
        prompt_hash = tracker.register_prompt_snapshot(prompt_snapshot)
        
        # Update the snapshot with prompt hash
        snapshot.prompt_hash = prompt_hash
        
        # Regenerate config hash to include prompt version
        config_str = (
            f"{snapshot.supervisor_model}_{snapshot.supervisor_temp}_"
            f"{snapshot.extractor_model}_{snapshot.extractor_temp}_"
            f"{snapshot.calculator_model}_{snapshot.calculator_temp}_"
            f"{snapshot.validator_model}_{snapshot.validator_temp}_"
            f"{snapshot.process_type}_{snapshot.manager_model}_"
            f"{snapshot.memory_enabled}_{snapshot.cache_enabled}_{snapshot.verbose}_"
            f"prompt_{prompt_hash}"
        )
        snapshot.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not capture prompt version: {e}")
    
    return snapshot 