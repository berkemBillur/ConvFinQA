"""Enhanced tracking system for comprehensive multi-agent experiment management.

This module provides complete capture of agent configurations, workflow settings,
task definitions, dataset configuration, and performance results with SQLite
database storage and interactive dashboard capabilities.
"""

import json
import hashlib
import sqlite3
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import os

from .config import Config, APIKeyManager

logger = logging.getLogger(__name__)


@dataclass
class AgentConfiguration:
    """Complete agent configuration capture."""
    name: str
    role: str
    goal: str
    backstory: str
    model: str
    temperature: float
    verbose: bool
    allow_delegation: bool
    tools: List[str]  # Tool class names
    max_execution_time: Optional[int] = None
    max_retry_limit: Optional[int] = None


@dataclass
class WorkflowConfiguration:
    """Complete workflow configuration capture."""
    process_type: str  # 'hierarchical', 'sequential', etc.
    manager_model: str
    manager_temperature: float
    memory_enabled: bool
    cache_enabled: bool
    max_execution_time: Optional[int]
    share_crew: bool = False
    step_callback: Optional[str] = None


@dataclass
class TaskConfiguration:
    """Complete task configuration capture."""
    task_type: str  # 'extraction', 'calculation', 'validation', etc.
    description_template: str  # Task description template
    expected_output: str
    agent_assignment: str
    dependencies: List[str]  # Dependent task types
    human_input: bool = False
    async_execution: bool = False


@dataclass
class DatasetConfiguration:
    """Dataset configuration and sampling strategy."""
    total_conversations: int
    sample_size: int
    sampling_strategy: str  # 'sequential', 'random'
    random_seed: Optional[int]
    conversation_ids: List[str]  # Actual IDs processed
    avg_conversation_length: float
    question_types: Dict[str, int]  # Question type distribution


@dataclass
class PerformanceResults:
    """Comprehensive performance results."""
    total_questions: int
    correct_answers: int
    accuracy_rate: float
    total_execution_time: float
    avg_execution_time_per_question: float
    total_estimated_cost: float
    avg_cost_per_question: float
    
    # Agent-specific breakdown
    agent_performance: Dict[str, Dict[str, Any]]
    
    # Question type breakdown
    question_type_performance: Dict[str, Dict[str, Any]]
    
    # Error analysis
    failure_count: int
    failure_rate: float
    error_patterns: Dict[str, int]
    common_failures: List[str]
    
    # Cost breakdown
    cost_by_agent: Dict[str, float]
    cost_by_question_type: Dict[str, float]


@dataclass
class SystemConfiguration:
    """System-level configuration."""
    openai_api_key_set: bool
    api_base_url: Optional[str]
    rate_limits: Dict[str, Any]
    timeout_settings: Dict[str, int]
    retry_policies: Dict[str, Any]
    cost_controls: Dict[str, Any]
    logging_level: str
    environment: str  # 'development', 'production', etc.


@dataclass
class CompleteExperimentSnapshot:
    """Complete experiment configuration snapshot."""
    
    # Core identifiers
    experiment_id: str
    config_hash: str
    timestamp: str
    
    # Configurations
    agents: List[AgentConfiguration]
    workflow: WorkflowConfiguration
    tasks: List[TaskConfiguration]
    dataset: DatasetConfiguration
    system: SystemConfiguration
    
    # Results (filled after execution)
    performance: Optional[PerformanceResults] = None
    
    # Metadata
    git_commit: Optional[str] = None
    environment_snapshot: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    
    def __post_init__(self):
        """Generate deterministic hash and IDs."""
        if not self.experiment_id:
            self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        
        if not self.config_hash:
            # Create deterministic hash from configuration
            config_str = self._create_config_string()
            self.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def _create_config_string(self) -> str:
        """Create deterministic string representation of configuration."""
        agent_str = "_".join([
            f"{a.name}:{a.model}:{a.temperature}:{a.role[:20]}"
            for a in sorted(self.agents, key=lambda x: x.name)
        ])
        
        workflow_str = f"{self.workflow.process_type}:{self.workflow.manager_model}:{self.workflow.manager_temperature}"
        
        task_str = "_".join([
            f"{t.task_type}:{t.agent_assignment}"
            for t in sorted(self.tasks, key=lambda x: x.task_type)
        ])
        
        dataset_str = f"{self.dataset.sample_size}:{self.dataset.sampling_strategy}:{self.dataset.random_seed}"
        
        return f"{agent_str}|{workflow_str}|{task_str}|{dataset_str}"


class EnhancedExperimentTracker:
    """Enhanced experiment tracking with SQLite database and dashboard support."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize enhanced experiment tracker with unified folder structure."""
        if storage_dir is None:
            storage_dir = "experiment_tracking/configurations"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Database and snapshot files in configurations subfolder
        self.db_path = self.storage_dir / "experiments.db"
        self.snapshots_file = self.storage_dir / "snapshots.json"
        
        # Results directory in parallel subfolder  
        self.results_dir = self.storage_dir.parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"✅ Enhanced tracker initialized:")
        logger.info(f"   📁 Configurations: {self.storage_dir}")
        logger.info(f"   📁 Results: {self.results_dir}")
        logger.info(f"   🗃️  Database: {self.db_path}")
        logger.info(f"   📸 Snapshots: {self.snapshots_file}")
        
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    config_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    git_commit TEXT,
                    notes TEXT,
                    
                    -- Agent configuration summary
                    num_agents INTEGER,
                    agent_models TEXT,  -- JSON
                    agent_temperatures TEXT,  -- JSON
                    
                    -- Workflow configuration
                    process_type TEXT,
                    manager_model TEXT,
                    manager_temperature REAL,
                    memory_enabled BOOLEAN,
                    cache_enabled BOOLEAN,
                    
                    -- Dataset configuration
                    sample_size INTEGER,
                    sampling_strategy TEXT,
                    random_seed INTEGER,
                    avg_conversation_length REAL,
                    
                    -- Performance results
                    total_questions INTEGER,
                    correct_answers INTEGER,
                    accuracy_rate REAL,
                    total_execution_time REAL,
                    avg_execution_time_per_question REAL,
                    total_estimated_cost REAL,
                    avg_cost_per_question REAL,
                    failure_rate REAL,
                    
                    -- Metadata
                    environment TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS agent_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    agent_name TEXT,
                    total_executions INTEGER,
                    successful_executions INTEGER,
                    success_rate REAL,
                    avg_execution_time REAL,
                    estimated_cost REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                );
                
                CREATE TABLE IF NOT EXISTS question_type_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    question_type TEXT,
                    total_questions INTEGER,
                    correct_answers INTEGER,
                    accuracy_rate REAL,
                    avg_execution_time REAL,
                    avg_cost REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                );
                
                CREATE TABLE IF NOT EXISTS error_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    error_type TEXT,
                    error_count INTEGER,
                    example_message TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_experiments_timestamp ON experiments(timestamp);
                CREATE INDEX IF NOT EXISTS idx_experiments_config_hash ON experiments(config_hash);
                CREATE INDEX IF NOT EXISTS idx_experiments_accuracy ON experiments(accuracy_rate);
            """)
    
    def capture_system_configuration(self) -> SystemConfiguration:
        """Capture current system configuration."""
        return SystemConfiguration(
            openai_api_key_set=bool(APIKeyManager.load_openai_key()),
            api_base_url=os.getenv('OPENAI_API_BASE'),
            rate_limits={},  # Could be enhanced to capture actual rate limits
            timeout_settings={'default': 30},
            retry_policies={'max_retries': 3},
            cost_controls={},
            logging_level=logging.getLevelName(logging.getLogger().getEffectiveLevel()),
            environment=os.getenv('ENVIRONMENT', 'development')
        )
    
    def capture_agent_configurations(self, agents: Dict[str, Any]) -> List[AgentConfiguration]:
        """Capture complete agent configurations from CrewAI agents."""
        agent_configs = []
        
        for name, agent in agents.items():
            # Extract LLM configuration
            llm_model = getattr(agent.llm, 'model_name', 'unknown')
            llm_temperature = getattr(agent.llm, 'temperature', 0.0)
            
            # Extract tools
            tool_names = [tool.__class__.__name__ for tool in getattr(agent, 'tools', [])]
            
            config = AgentConfiguration(
                name=name,
                role=getattr(agent, 'role', ''),
                goal=getattr(agent, 'goal', ''),
                backstory=getattr(agent, 'backstory', ''),
                model=llm_model,
                temperature=llm_temperature,
                verbose=getattr(agent, 'verbose', False),
                allow_delegation=getattr(agent, 'allow_delegation', False),
                tools=tool_names,
                max_execution_time=getattr(agent, 'max_execution_time', None),
                max_retry_limit=getattr(agent, 'max_retry_limit', None)
            )
            agent_configs.append(config)
        
        return agent_configs
    
    def capture_workflow_configuration(self, crew_config: Dict[str, Any]) -> WorkflowConfiguration:
        """Capture workflow configuration from crew settings."""
        return WorkflowConfiguration(
            process_type=crew_config.get('process_type', 'hierarchical'),
            manager_model=crew_config.get('manager_model', 'gpt-4o'),
            manager_temperature=crew_config.get('manager_temperature', 0.1),
            memory_enabled=crew_config.get('memory', True),
            cache_enabled=crew_config.get('cache', True),
            max_execution_time=crew_config.get('max_execution_time'),
            share_crew=crew_config.get('share_crew', False),
            step_callback=crew_config.get('step_callback')
        )
    
    def capture_task_configurations(self, task_builders: List[str]) -> List[TaskConfiguration]:
        """Capture task configuration information."""
        # This would be enhanced to capture actual task definitions
        task_configs = []
        
        task_types = ['extraction', 'calculation', 'validation', 'supervision']
        agent_assignments = ['extractor', 'calculator', 'validator', 'supervisor']
        
        for i, task_type in enumerate(task_types):
            config = TaskConfiguration(
                task_type=task_type,
                description_template=f"Template for {task_type} task",
                expected_output="JSON string" if task_type == 'extraction' else "String",
                agent_assignment=agent_assignments[i] if i < len(agent_assignments) else 'unknown',
                dependencies=['extraction'] if task_type in ['calculation', 'validation'] else [],
                human_input=False,
                async_execution=False
            )
            task_configs.append(config)
        
        return task_configs
    
    def capture_dataset_configuration(self, 
                                    total_conversations: int,
                                    sample_size: int,
                                    sampling_strategy: str,
                                    random_seed: Optional[int],
                                    conversation_ids: List[str],
                                    records: List[Any]) -> DatasetConfiguration:
        """Capture dataset configuration and statistics."""
        
        # Calculate conversation statistics
        total_questions = sum(len(record.dialogue.conv_questions) for record in records)
        avg_length = total_questions / len(records) if records else 0
        
        # Analyze question types (simplified)
        question_types = {'calculation': 0, 'lookup': 0, 'comparison': 0, 'other': 0}
        for record in records:
            for question in record.dialogue.conv_questions:
                q_lower = question.lower()
                if any(word in q_lower for word in ['calculate', 'compute', 'difference', 'change']):
                    question_types['calculation'] += 1
                elif any(word in q_lower for word in ['what was', 'what is', 'how much']):
                    question_types['lookup'] += 1
                elif any(word in q_lower for word in ['compare', 'ratio', 'versus']):
                    question_types['comparison'] += 1
                else:
                    question_types['other'] += 1
        
        return DatasetConfiguration(
            total_conversations=total_conversations,
            sample_size=sample_size,
            sampling_strategy=sampling_strategy,
            random_seed=random_seed,
            conversation_ids=conversation_ids,
            avg_conversation_length=avg_length,
            question_types=question_types
        )
    
    def create_experiment_snapshot(self,
                                 agents: Dict[str, Any],
                                 crew_config: Dict[str, Any],
                                 dataset_info: Dict[str, Any],
                                 notes: str = "") -> CompleteExperimentSnapshot:
        """Create complete experiment snapshot before execution."""
        
        # Capture all configurations
        agent_configs = self.capture_agent_configurations(agents)
        workflow_config = self.capture_workflow_configuration(crew_config)
        task_configs = self.capture_task_configurations([])  # Would be enhanced
        dataset_config = self.capture_dataset_configuration(**dataset_info)
        system_config = self.capture_system_configuration()
        
        # Get git commit if available
        git_commit = self._get_git_commit()
        
        # Create snapshot
        snapshot = CompleteExperimentSnapshot(
            experiment_id="",  # Will be auto-generated
            config_hash="",    # Will be auto-generated
            timestamp="",      # Will be auto-generated
            agents=agent_configs,
            workflow=workflow_config,
            tasks=task_configs,
            dataset=dataset_config,
            system=system_config,
            git_commit=git_commit,
            environment_snapshot=dict(os.environ),
            notes=notes
        )
        
        # Save snapshot
        self._save_snapshot(snapshot)
        
        return snapshot
    
    def update_experiment_results(self, 
                                experiment_id: str,
                                performance_results: PerformanceResults):
        """Update experiment with performance results."""
        
        # Update in-memory snapshot
        snapshot = self._load_snapshot(experiment_id)
        if snapshot:
            snapshot.performance = performance_results
            self._save_snapshot(snapshot)
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE experiments SET
                    total_questions = ?,
                    correct_answers = ?,
                    accuracy_rate = ?,
                    total_execution_time = ?,
                    avg_execution_time_per_question = ?,
                    total_estimated_cost = ?,
                    avg_cost_per_question = ?,
                    failure_rate = ?
                WHERE experiment_id = ?
            """, (
                performance_results.total_questions,
                performance_results.correct_answers,
                performance_results.accuracy_rate,
                performance_results.total_execution_time,
                performance_results.avg_execution_time_per_question,
                performance_results.total_estimated_cost,
                performance_results.avg_cost_per_question,
                performance_results.failure_rate,
                experiment_id
            ))
            
            # Save agent performance
            for agent_name, perf in performance_results.agent_performance.items():
                conn.execute("""
                    INSERT INTO agent_performance 
                    (experiment_id, agent_name, total_executions, successful_executions, 
                     success_rate, avg_execution_time, estimated_cost)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, agent_name,
                    perf.get('total_executions', 0),
                    perf.get('successful_executions', 0),
                    perf.get('success_rate', 0.0),
                    perf.get('avg_execution_time', 0.0),
                    perf.get('estimated_cost', 0.0)
                ))
            
            # Save question type performance
            for q_type, perf in performance_results.question_type_performance.items():
                conn.execute("""
                    INSERT INTO question_type_performance 
                    (experiment_id, question_type, total_questions, correct_answers,
                     accuracy_rate, avg_execution_time, avg_cost)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, q_type,
                    perf.get('total_questions', 0),
                    perf.get('correct_answers', 0),
                    perf.get('accuracy_rate', 0.0),
                    perf.get('avg_execution_time', 0.0),
                    perf.get('avg_cost', 0.0)
                ))
            
            # Save error patterns
            for error_type, count in performance_results.error_patterns.items():
                conn.execute("""
                    INSERT INTO error_patterns 
                    (experiment_id, error_type, error_count, example_message)
                    VALUES (?, ?, ?, ?)
                """, (
                    experiment_id, error_type, count,
                    performance_results.common_failures[0] if performance_results.common_failures else ""
                ))
    
    def get_experiment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get experiment history for dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM experiments 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_configuration_comparison(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare configurations across experiments."""
        snapshots = []
        for exp_id in experiment_ids:
            snapshot = self._load_snapshot(exp_id)
            if snapshot:
                snapshots.append(snapshot)
        
        if not snapshots:
            return {}
        
        # Compare configurations
        comparison = {
            'experiments': len(snapshots),
            'agent_differences': self._compare_agents([s.agents for s in snapshots]),
            'workflow_differences': self._compare_workflows([s.workflow for s in snapshots]),
            'performance_comparison': self._compare_performance([s.performance for s in snapshots if s.performance])
        }
        
        return comparison
    
    def revert_to_configuration(self, experiment_id: str) -> Dict[str, Any]:
        """Generate configuration for reverting to a specific experiment."""
        snapshot = self._load_snapshot(experiment_id)
        if not snapshot:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Generate revert configuration
        revert_config = {
            'agents': {agent.name: asdict(agent) for agent in snapshot.agents},
            'workflow': asdict(snapshot.workflow),
            'tasks': [asdict(task) for task in snapshot.tasks],
            'dataset': asdict(snapshot.dataset),
            'system': asdict(snapshot.system)
        }
        
        return revert_config
    
    def _save_snapshot(self, snapshot: CompleteExperimentSnapshot):
        """Save snapshot to JSON file and database."""
        
        # Save to JSON
        snapshots = self._load_all_snapshots()
        snapshots[snapshot.experiment_id] = asdict(snapshot)
        
        with open(self.snapshots_file, 'w') as f:
            json.dump(snapshots, f, indent=2, default=str)
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments 
                (experiment_id, config_hash, timestamp, git_commit, notes,
                 num_agents, agent_models, agent_temperatures,
                 process_type, manager_model, manager_temperature, memory_enabled, cache_enabled,
                 sample_size, sampling_strategy, random_seed, avg_conversation_length,
                 environment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.experiment_id, snapshot.config_hash, snapshot.timestamp,
                snapshot.git_commit, snapshot.notes,
                len(snapshot.agents),
                json.dumps([a.model for a in snapshot.agents]),
                json.dumps([a.temperature for a in snapshot.agents]),
                snapshot.workflow.process_type,
                snapshot.workflow.manager_model,
                snapshot.workflow.manager_temperature,
                snapshot.workflow.memory_enabled,
                snapshot.workflow.cache_enabled,
                snapshot.dataset.sample_size,
                snapshot.dataset.sampling_strategy,
                snapshot.dataset.random_seed,
                snapshot.dataset.avg_conversation_length,
                snapshot.system.environment
            ))
    
    def _load_snapshot(self, experiment_id: str) -> Optional[CompleteExperimentSnapshot]:
        """Load snapshot from JSON file."""
        snapshots = self._load_all_snapshots()
        snapshot_data = snapshots.get(experiment_id)
        
        if not snapshot_data:
            return None
        
        # Convert back to dataclass with proper reconstruction
        try:
            # Reconstruct agent configurations
            agents = []
            if 'agents' in snapshot_data:
                for agent_data in snapshot_data['agents']:
                    if isinstance(agent_data, dict):
                        agents.append(AgentConfiguration(**agent_data))
                    else:
                        agents.append(agent_data)
            
            # Reconstruct workflow configuration
            workflow = snapshot_data.get('workflow', {})
            if isinstance(workflow, dict):
                workflow = WorkflowConfiguration(**workflow)
            
            # Reconstruct task configurations
            tasks = []
            if 'tasks' in snapshot_data:
                for task_data in snapshot_data['tasks']:
                    if isinstance(task_data, dict):
                        tasks.append(TaskConfiguration(**task_data))
                    else:
                        tasks.append(task_data)
            
            # Reconstruct dataset configuration
            dataset = snapshot_data.get('dataset', {})
            if isinstance(dataset, dict):
                dataset = DatasetConfiguration(**dataset)
            
            # Reconstruct system configuration
            system = snapshot_data.get('system', {})
            if isinstance(system, dict):
                system = SystemConfiguration(**system)
            
            # Reconstruct performance results if present
            performance = None
            if 'performance' in snapshot_data and snapshot_data['performance']:
                perf_data = snapshot_data['performance']
                if isinstance(perf_data, dict):
                    performance = PerformanceResults(**perf_data)
            
            # Create snapshot with reconstructed objects
            snapshot = CompleteExperimentSnapshot(
                experiment_id=snapshot_data.get('experiment_id', experiment_id),
                config_hash=snapshot_data.get('config_hash', ''),
                timestamp=snapshot_data.get('timestamp', ''),
                agents=agents,
                workflow=workflow,
                tasks=tasks,
                dataset=dataset,
                system=system,
                performance=performance,
                git_commit=snapshot_data.get('git_commit'),
                environment_snapshot=snapshot_data.get('environment_snapshot', {}),
                notes=snapshot_data.get('notes', '')
            )
            
            return snapshot
            
        except Exception as e:
            logger.warning(f"Could not load snapshot {experiment_id}: {e}")
            return None
    
    def _load_all_snapshots(self) -> Dict[str, Any]:
        """Load all snapshots from JSON file."""
        if not self.snapshots_file.exists():
            return {}
        
        try:
            with open(self.snapshots_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load snapshots file: {e}")
            return {}
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def _compare_agents(self, agent_lists: List[List[AgentConfiguration]]) -> Dict[str, Any]:
        """Compare agent configurations across experiments."""
        if not agent_lists:
            return {}
        
        # Find differences in agent configurations
        differences = {}
        agent_names = set()
        for agents in agent_lists:
            agent_names.update(agent.name for agent in agents)
        
        for name in agent_names:
            agent_configs = []
            for agents in agent_lists:
                agent_config = next((a for a in agents if a.name == name), None)
                agent_configs.append(agent_config)
            
            # Check for differences
            if len(set(str(a) for a in agent_configs if a)) > 1:
                differences[name] = agent_configs
        
        return differences
    
    def _compare_workflows(self, workflows: List[WorkflowConfiguration]) -> Dict[str, Any]:
        """Compare workflow configurations."""
        if not workflows:
            return {}
        
        differences = {}
        first_workflow = workflows[0]
        
        for workflow in workflows[1:]:
            for field in first_workflow.__dict__:
                if getattr(first_workflow, field) != getattr(workflow, field):
                    if field not in differences:
                        differences[field] = []
                    differences[field].append({
                        'value': getattr(workflow, field),
                        'experiment': workflow  # Would need experiment ID
                    })
        
        return differences
    
    def _compare_performance(self, performances: List[PerformanceResults]) -> Dict[str, Any]:
        """Compare performance results."""
        if not performances:
            return {}
        
        comparison = {
            'accuracy_rates': [p.accuracy_rate for p in performances],
            'execution_times': [p.avg_execution_time_per_question for p in performances],
            'costs': [p.avg_cost_per_question for p in performances],
            'failure_rates': [p.failure_rate for p in performances]
        }
        
        return comparison
    
    def _get_agent_model(self, agent):
        """Get agent model handling both objects and dicts."""
        if hasattr(agent, 'model'):
            return agent.model
        elif isinstance(agent, dict):
            return agent.get('model', 'unknown')
        else:
            return 'unknown'
    
    def _get_agent_temperature(self, agent):
        """Get agent temperature handling both objects and dicts."""
        if hasattr(agent, 'temperature'):
            return agent.temperature
        elif isinstance(agent, dict):
            return agent.get('temperature', 0.0)
        else:
            return 0.0
    
    def _get_workflow_field(self, workflow, field_name):
        """Get workflow field handling both objects and dicts."""
        if hasattr(workflow, field_name):
            return getattr(workflow, field_name)
        elif isinstance(workflow, dict):
            return workflow.get(field_name)
        else:
            return None
    
    def _get_dataset_field(self, dataset, field_name):
        """Get dataset field handling both objects and dicts."""
        if hasattr(dataset, field_name):
            return getattr(dataset, field_name)
        elif isinstance(dataset, dict):
            return dataset.get(field_name)
        else:
            return None
    
    def _get_system_field(self, system, field_name):
        """Get system field handling both objects and dicts."""
        if hasattr(system, field_name):
            return getattr(system, field_name)
        elif isinstance(system, dict):
            return system.get(field_name)
        else:
            return None


# Global instance
_enhanced_tracker = None

def get_enhanced_tracker() -> EnhancedExperimentTracker:
    """Get global enhanced tracker instance."""
    global _enhanced_tracker
    if _enhanced_tracker is None:
        _enhanced_tracker = EnhancedExperimentTracker()
    return _enhanced_tracker 