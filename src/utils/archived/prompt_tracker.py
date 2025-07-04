"""Prompt versioning and tracking system for ConvFinQA multi-agent predictors.

This module provides comprehensive prompt tracking capabilities including:
- Agent backstory versioning
- Task description fingerprinting  
- Prompt change detection
- Historical prompt comparison
"""

import json
import hashlib
import inspect
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptSnapshot:
    """Complete snapshot of all prompts used in a configuration."""
    
    # Agent backstories
    supervisor_backstory: str
    extractor_backstory: str
    calculator_backstory: str
    validator_backstory: str
    
    # Agent roles and goals
    supervisor_role: str
    supervisor_goal: str
    extractor_role: str
    extractor_goal: str
    calculator_role: str
    calculator_goal: str
    validator_role: str
    validator_goal: str
    
    # Task template signatures (will be populated from tasks.py)
    extraction_task_template: str = ""
    calculation_task_template: str = ""
    validation_task_template: str = ""
    
    # Metadata
    prompt_hash: str = field(init=False)
    timestamp: str = field(init=False)
    source_files: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate prompt hash and timestamp after initialization."""
        self.timestamp = datetime.now().isoformat()
        self.prompt_hash = self._generate_prompt_hash()
    
    def _generate_prompt_hash(self) -> str:
        """Generate deterministic hash from all prompt content."""
        # Combine all prompt content for hashing
        content_parts = [
            self.supervisor_backstory,
            self.extractor_backstory, 
            self.calculator_backstory,
            self.validator_backstory,
            self.supervisor_role,
            self.supervisor_goal,
            self.extractor_role,
            self.extractor_goal,
            self.calculator_role,
            self.calculator_goal,
            self.validator_role,
            self.validator_goal,
            self.extraction_task_template,
            self.calculation_task_template,
            self.validation_task_template
        ]
        
        combined_content = "|".join(content_parts)
        return hashlib.md5(combined_content.encode()).hexdigest()[:8]


@dataclass
class PromptComparison:
    """Comparison between two prompt versions."""
    
    old_hash: str
    new_hash: str
    timestamp: str
    changes: Dict[str, Dict[str, str]]  # field -> {old: str, new: str}
    change_summary: str
    
    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()


class PromptTracker:
    """Central prompt tracking and versioning system."""
    
    def __init__(self, storage_dir: str = "experiments/tracking"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.prompts_file = self.storage_dir / "prompts.json"
        self.comparisons_file = self.storage_dir / "prompt_comparisons.json"
        
        # Load existing data
        self._prompts = self._load_prompts()
        self._comparisons = self._load_comparisons()
    
    def capture_current_prompts(self) -> PromptSnapshot:
        """Capture current prompts from agents.py and tasks.py."""
        try:
            # Import agents module to extract current prompts
            import importlib.util
            import sys
            from pathlib import Path
            
            # Dynamic import to avoid relative import issues
            agents_path = Path(__file__).parent.parent / "predictors" / "multi_agent" / "agents.py"
            spec = importlib.util.spec_from_file_location("agents", agents_path)
            if spec and spec.loader:
                agents_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(agents_module)
            else:
                raise ImportError("Could not load agents module")
            
            config_path = Path(__file__).parent / "config.py"
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec and spec.loader:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
            else:
                raise ImportError("Could not load config module")
            
            build_agents = agents_module.build_agents
            Config = config_module.Config
            
            # Build agents with default config to extract prompts
            config = Config()
            agents = build_agents(config)
            
            # Extract agent information
            snapshot = PromptSnapshot(
                supervisor_backstory=agents["supervisor"].backstory,
                extractor_backstory=agents["extractor"].backstory,
                calculator_backstory=agents["calculator"].backstory,
                validator_backstory=agents["validator"].backstory,
                supervisor_role=agents["supervisor"].role,
                supervisor_goal=agents["supervisor"].goal,
                extractor_role=agents["extractor"].role,
                extractor_goal=agents["extractor"].goal,
                calculator_role=agents["calculator"].role,
                calculator_goal=agents["calculator"].goal,
                validator_role=agents["validator"].role,
                validator_goal=agents["validator"].goal,
            )
            
            # Try to capture task templates
            try:
                snapshot.extraction_task_template = self._extract_task_template("extraction")
                snapshot.calculation_task_template = self._extract_task_template("calculation") 
                snapshot.validation_task_template = self._extract_task_template("validation")
            except Exception as e:
                logger.warning(f"Could not extract task templates: {e}")
            
            # Record source file info
            snapshot.source_files = self._get_source_file_info()
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to capture current prompts: {e}")
            raise
    
    def _extract_task_template(self, task_type: str) -> str:
        """Extract task template from tasks.py by reading source code."""
        try:
            import sys
            from pathlib import Path as PathLib
            sys.path.append(str(PathLib(__file__).parent.parent))
            
            from predictors.multi_agent import tasks
            
            # Read the tasks.py source file
            tasks_file = Path(inspect.getfile(tasks)).resolve()
            with open(tasks_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract the relevant function
            if task_type == "extraction":
                func_name = "build_extraction_task"
            elif task_type == "calculation":
                func_name = "build_calculation_task"
            elif task_type == "validation":
                func_name = "build_validation_task"
            else:
                return ""
            
            # Find function definition and extract template
            lines = content.split('\n')
            in_function = False
            template_lines = []
            
            for line in lines:
                if f"def {func_name}" in line:
                    in_function = True
                    continue
                
                if in_function:
                    if line.strip().startswith('def ') and not line.strip().startswith('def _'):
                        break  # Next function started
                    
                    # Look for desc = f""" or similar
                    if 'desc = f"""' in line or 'description=' in line:
                        template_lines.append(line.strip())
                    elif template_lines and ('"""' in line or 'expected_output' in line):
                        template_lines.append(line.strip())
                        if '"""' in line:
                            break
                    elif template_lines:
                        template_lines.append(line.strip())
            
            return '\n'.join(template_lines)[:500] + "..." if template_lines else ""
            
        except Exception as e:
            logger.warning(f"Could not extract {task_type} template: {e}")
            return ""
    
    def _get_source_file_info(self) -> Dict[str, str]:
        """Get source file modification times for tracking."""
        source_files = {}
        try:
            import sys
            from pathlib import Path as PathLib
            sys.path.append(str(PathLib(__file__).parent.parent))
            
            from predictors.multi_agent import agents, tasks
            
            agents_file = Path(inspect.getfile(agents)).resolve()
            tasks_file = Path(inspect.getfile(tasks)).resolve()
            
            source_files["agents.py"] = datetime.fromtimestamp(agents_file.stat().st_mtime).isoformat()
            source_files["tasks.py"] = datetime.fromtimestamp(tasks_file.stat().st_mtime).isoformat()
            
        except Exception as e:
            logger.warning(f"Could not get source file info: {e}")
        
        return source_files
    
    def register_prompt_snapshot(self, snapshot: PromptSnapshot) -> str:
        """Register a new prompt snapshot and return its hash."""
        prompt_hash = snapshot.prompt_hash
        
        # Check if this is a new version
        if prompt_hash not in self._prompts:
            self._prompts[prompt_hash] = asdict(snapshot)
            self._save_prompts()
            
            # Compare with previous version if exists
            if len(self._prompts) > 1:
                previous_hashes = [h for h in self._prompts.keys() if h != prompt_hash]
                if previous_hashes:
                    latest_previous = max(previous_hashes, 
                                        key=lambda h: self._prompts[h]['timestamp'])
                    comparison = self._compare_prompts(latest_previous, prompt_hash)
                    self._comparisons.append(asdict(comparison))
                    self._save_comparisons()
            
            logger.info(f"✅ New prompt version registered: {prompt_hash}")
        else:
            logger.info(f"📋 Prompt version already exists: {prompt_hash}")
        
        return prompt_hash
    
    def _compare_prompts(self, old_hash: str, new_hash: str) -> PromptComparison:
        """Compare two prompt versions and identify changes."""
        old_prompts = self._prompts[old_hash]
        new_prompts = self._prompts[new_hash]
        
        changes = {}
        change_count = 0
        
        # Compare all prompt fields
        prompt_fields = [
            'supervisor_backstory', 'extractor_backstory', 'calculator_backstory', 'validator_backstory',
            'supervisor_role', 'supervisor_goal', 'extractor_role', 'extractor_goal',
            'calculator_role', 'calculator_goal', 'validator_role', 'validator_goal',
            'extraction_task_template', 'calculation_task_template', 'validation_task_template'
        ]
        
        for field in prompt_fields:
            old_val = old_prompts.get(field, "")
            new_val = new_prompts.get(field, "")
            
            if old_val != new_val:
                changes[field] = {
                    "old": old_val[:200] + "..." if len(old_val) > 200 else old_val,
                    "new": new_val[:200] + "..." if len(new_val) > 200 else new_val
                }
                change_count += 1
        
        # Generate summary
        if change_count == 0:
            summary = "No prompt changes detected"
        elif change_count == 1:
            changed_field = list(changes.keys())[0]
            summary = f"Modified {changed_field.replace('_', ' ')}"
        else:
            summary = f"Modified {change_count} prompt fields: {', '.join(changes.keys())}"
        
        return PromptComparison(
            old_hash=old_hash,
            new_hash=new_hash,
            timestamp="",  # Will be set by __post_init__
            changes=changes,
            change_summary=summary
        )
    
    def get_prompt_snapshot(self, prompt_hash: str) -> Optional[PromptSnapshot]:
        """Retrieve a specific prompt snapshot."""
        if prompt_hash in self._prompts:
            data = self._prompts[prompt_hash]
            # Reconstruct PromptSnapshot object
            snapshot = PromptSnapshot(
                supervisor_backstory=data['supervisor_backstory'],
                extractor_backstory=data['extractor_backstory'],
                calculator_backstory=data['calculator_backstory'],
                validator_backstory=data['validator_backstory'],
                supervisor_role=data['supervisor_role'],
                supervisor_goal=data['supervisor_goal'],
                extractor_role=data['extractor_role'],
                extractor_goal=data['extractor_goal'],
                calculator_role=data['calculator_role'],
                calculator_goal=data['calculator_goal'],
                validator_role=data['validator_role'],
                validator_goal=data['validator_goal'],
            )
            snapshot.extraction_task_template = data.get('extraction_task_template', '')
            snapshot.calculation_task_template = data.get('calculation_task_template', '')
            snapshot.validation_task_template = data.get('validation_task_template', '')
            snapshot.prompt_hash = data['prompt_hash']
            snapshot.timestamp = data['timestamp']
            snapshot.source_files = data.get('source_files', {})
            return snapshot
        return None
    
    def get_latest_prompt_hash(self) -> Optional[str]:
        """Get the hash of the most recent prompt version."""
        if not self._prompts:
            return None
        
        return max(self._prompts.keys(), 
                  key=lambda h: self._prompts[h]['timestamp'])
    
    def get_prompt_history(self) -> List[Dict[str, Any]]:
        """Get chronological history of all prompt versions."""
        history = []
        for prompt_hash, data in self._prompts.items():
            history.append({
                'prompt_hash': prompt_hash,
                'timestamp': data['timestamp'],
                'source_files': data.get('source_files', {}),
                'summary': f"Prompt version {prompt_hash}"
            })
        
        return sorted(history, key=lambda x: x['timestamp'])
    
    def get_recent_changes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent prompt changes with summaries."""
        recent_comparisons = sorted(self._comparisons, 
                                  key=lambda x: x['timestamp'], 
                                  reverse=True)[:limit]
        return recent_comparisons
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompt snapshots from storage."""
        if self.prompts_file.exists():
            try:
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load prompts file: {e}")
        return {}
    
    def _save_prompts(self) -> None:
        """Save prompt snapshots to storage."""
        try:
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(self._prompts, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save prompts file: {e}")
    
    def _load_comparisons(self) -> List[Dict[str, Any]]:
        """Load prompt comparisons from storage."""
        if self.comparisons_file.exists():
            try:
                with open(self.comparisons_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load comparisons file: {e}")
        return []
    
    def _save_comparisons(self) -> None:
        """Save prompt comparisons to storage."""
        try:
            with open(self.comparisons_file, 'w', encoding='utf-8') as f:
                json.dump(self._comparisons, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save comparisons file: {e}")


# Global prompt tracker instance
_prompt_tracker = None

def get_prompt_tracker() -> PromptTracker:
    """Get the global prompt tracker instance."""
    global _prompt_tracker
    if _prompt_tracker is None:
        _prompt_tracker = PromptTracker()
    return _prompt_tracker 