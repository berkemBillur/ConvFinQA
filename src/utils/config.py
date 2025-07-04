"""Configuration management for ConvFinQA project."""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data-related configuration."""
    dataset_path: str
    cache_dir: str


@dataclass 
class EvaluationConfig:
    """Evaluation-related configuration."""
    tolerance: float
    quick_eval: Dict[str, Any]
    comprehensive_eval: Dict[str, Any]
    metrics: Dict[str, Any]


@dataclass
class DSLConfig:
    """DSL execution configuration."""
    constants: Dict[str, float]
    timeout: int


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    format: str
    file: str


@dataclass
class ModelConfig:
    """Model configuration."""
    default_type: str
    baseline: Dict[str, Any]
    hybrid_keyword: Optional[Dict[str, Any]] = None
    multi_agent: Optional[Dict[str, Any]] = None
    crewai: Optional[Dict[str, Any]] = None
    use_six_agents: bool = False


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    results_dir: str
    save_predictions: bool
    save_intermediate: bool
    config_hash_prefix: str = "multi_agent_v2"


class APIKeyManager:
    """Centralized manager for API keys with consistent priority loading."""
    
    @staticmethod
    def load_openai_key() -> Optional[str]:
        """Load OpenAI API key using consistent priority order.
        
        Priority order:
        1. Local config/api_keys.json file (most secure)
        2. Environment variable OPENAI_API_KEY
        3. Config object (if provided)
        
        Returns:
            Optional[str]: The API key if found, None otherwise
        """
        # Priority 1: Local config file
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "api_keys.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    api_config = json.load(f)
                    api_key = api_config.get('openai', {}).get('api_key')
                    if api_key:
                        # Ensure downstream libraries that only read the environment variable can access the key
                        current_env_key = os.getenv("OPENAI_API_KEY")
                        if current_env_key != api_key:
                            os.environ["OPENAI_API_KEY"] = api_key
                            logger.debug("🔄 Synchronized OPENAI_API_KEY environment variable with config/api_keys.json")
                        logger.debug("✅ API key loaded from config/api_keys.json")
                        return api_key
        except Exception as e:
            logger.debug(f"Failed to load from api_keys.json: {e}")
        
        # Priority 2: Environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'sk-your-openai-api-key-here':
            logger.debug("✅ API key loaded from environment")
            return api_key
        
        logger.warning("❌ No valid API key found in any source")
        return None


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialise configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Default to base config in config directory
            project_root = Path(__file__).parent.parent.parent
            resolved_config_path = project_root / "config" / "base.json"
        else:
            resolved_config_path = Path(config_path)
        
        self.config_path = resolved_config_path
        self._raw_config = self._load_config()
        
        # Parse into structured config objects
        self.data = DataConfig(**self._raw_config['data'])
        self.evaluation = EvaluationConfig(**self._raw_config['evaluation'])
        self.dsl = DSLConfig(**self._raw_config['dsl'])
        self.logging = LoggingConfig(**self._raw_config['logging'])
        self.models = ModelConfig(**self._raw_config['models'])
        self.experiments = ExperimentConfig(**self._raw_config['experiments'])
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Resolve relative paths to absolute paths
        config = self._resolve_paths(config)
        return config
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve relative paths to absolute paths."""
        project_root = Path(__file__).parent.parent.parent
        
        # Resolve data paths
        if 'data' in config:
            if 'dataset_path' in config['data']:
                if not Path(config['data']['dataset_path']).is_absolute():
                    config['data']['dataset_path'] = str(project_root / config['data']['dataset_path'])
            
            if 'cache_dir' in config['data']:
                if not Path(config['data']['cache_dir']).is_absolute():
                    config['data']['cache_dir'] = str(project_root / config['data']['cache_dir'])
        
        # Resolve logging paths
        if 'logging' in config and 'file' in config['logging']:
            if not Path(config['logging']['file']).is_absolute():
                config['logging']['file'] = str(project_root / config['logging']['file'])
        
        # Resolve experiment paths
        if 'experiments' in config and 'results_dir' in config['experiments']:
            if not Path(config['experiments']['results_dir']).is_absolute():
                config['experiments']['results_dir'] = str(project_root / config['experiments']['results_dir'])
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key.
        
        Args:
            key: Dot-separated key (e.g., 'evaluation.tolerance')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._raw_config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value by dot notation key.
        
        Args:
            key: Dot-separated key (e.g., 'evaluation.tolerance')
            value: New value to set
        """
        keys = key.split('.')
        config = self._raw_config
        
        # Navigate to parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Reload structured configs
        self._reload_structured_configs()
    
    def _reload_structured_configs(self) -> None:
        """Reload structured configuration objects after updates."""
        self.data = DataConfig(**self._raw_config['data'])
        self.evaluation = EvaluationConfig(**self._raw_config['evaluation'])
        self.dsl = DSLConfig(**self._raw_config['dsl'])
        self.logging = LoggingConfig(**self._raw_config['logging'])
        self.models = ModelConfig(**self._raw_config['models'])
        self.experiments = ExperimentConfig(**self._raw_config['experiments'])
    
    def save(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Args:
            output_path: Path to save config. If None, overwrites original.
        """
        if output_path is None:
            resolved_output_path = self.config_path
        else:
            resolved_output_path = Path(output_path)
        
        # Create directory if needed
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(resolved_output_path, 'w', encoding='utf-8') as f:
            json.dump(self._raw_config, f, indent=2)


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance.
    
    Args:
        config_path: Path to configuration file. Only used on first call.
        
    Returns:
        Global configuration instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload global configuration instance.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        New global configuration instance
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance 