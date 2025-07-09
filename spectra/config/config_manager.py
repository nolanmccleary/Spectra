import json
from pathlib import Path
from typing import Dict, Any, List
import yaml
from datetime import datetime
from .attack_config import AttackConfig, ExperimentConfig


class ConfigManager:
    """Manages loading and saving of attack configurations"""
    
    def __init__(self, config_dir: str = "experiments"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    

    def load_attack_config(self, config_name: str) -> AttackConfig:
        """Load standalone attack into preconfigured experiment"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return AttackConfig(**config_data['attack'])
    

    def load_experiment_config(self, config_name: str) -> ExperimentConfig:
        """Load experiment configuration from YAML file"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return ExperimentConfig(**config_data)
    

    def save_attack_config(self, config: AttackConfig, name: str) -> None:
        """Save configuration to YAML file"""
        config_path = self.config_dir / f"{name}.yaml"
        
        attack_data = json.loads(config.json())

        config_data = {
            'attack': attack_data,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    

    def save_experiment_config(self, config: ExperimentConfig, name: str) -> None:
        """Save experiment configuration to YAML file"""
        config_path = self.config_dir / f"{name}.yaml"
        
        config_data = json.loads(config.json())
        config_data['metadata'] = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    

    def list_configs(self) -> List[str]:
        """List all available configuration files"""
        configs = []
        for file in self.config_dir.glob("*.yaml"):
            configs.append(file.stem)
        return sorted(configs)
    

    def create_attack_from_dict(self, config_dict: Dict[str, Any]) -> AttackConfig:
        """Create configuration from dictionary"""
        return AttackConfig(**config_dict)
    
    
    def create_experiment_from_dict(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Create experiment configuration from dictionary"""
        return ExperimentConfig(**config_dict) 