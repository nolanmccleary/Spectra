from pydantic import BaseModel, Field, validator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from spectra.hashes import generate_ahash_batched, generate_ahash_rgb_batched, generate_dhash_batched, generate_dhash_rgb_batched, generate_phash_batched, generate_phash_rgb_batched, generate_pdq_batched
from spectra.deltagrad import Optimizer, NES_Optimizer, NES_Signed_Optimizer, Colinear_Optimizer, Optimizer_Config



class HyperparameterConfig(BaseModel):
    """Configuration for attack hyperparameters"""
    alpha: float = Field(..., gt=0, description="Alpha parameter for perturbations")
    beta: Tuple[float, Optional[float], Optional[float]] = Field(
        ..., description="Beta value or sweep parameters (start, end, step)"
    )
    step_coeff: float = Field(..., gt=0, description="Step coefficient")
    scale_factor: Tuple[float, Optional[float], Optional[float]] = Field(
        ..., description="Scale factor value or sweep parameters (start, end, step)"
    )
    
    @validator('beta', 'scale_factor')
    def validate_sweep_values(cls, v):
        if isinstance(v, tuple):
            if len(v) != 3:
                raise ValueError("Sweep parameters must be (start, end, step)")
            start, end, step = v
            if start is None or (end is not None and start >= end):
                raise ValueError("Invalid sweep range")
        return v


class AttackConfig(BaseModel):
    """Configuration for a single attack in a given experiment"""
    # Attack identification
    attack_name: Optional[str] = Field(default=None, description="Unique name for this attack")
    hash_function: Optional[str] = Field(default=None, description="Hash function to attack")
    
    # Core parameters
    hamming_threshold: Optional[int] = Field(default=None, description="Minimum hamming distance required")
    device: Optional[str] = Field(default="cpu", description="Device to use")
    
    # Attack parameters
    num_reps: Optional[int] = Field(default=None, description="Number of repetitions")
    attack_cycles: Optional[int] = Field(default=None, description="Number of attack cycles")
    
    # Optional features
    delta_scaledown: Optional[bool] = Field(default=True, description="Enable delta scaledown")
    gate: Optional[float] = Field(default=None, description="Gate function name")
    attack_verbose: Optional[bool] = Field(default=False, description="Enable verbose logging")
    deltagrad_verbose: Optional[bool] = Field(default=False, description="Enable verbose logging for deltagrad components")
    
    attack_type: Optional[str] = Field(default=None, description="Attack type")

    # Function specifications
    acceptance_func: Optional[str] = Field(default=None, description="Acceptance function name")
    quant_func: Optional[str] = Field(default=None, description="Quantization function name")
    
    # Hyperparameters
    hyperparameters: Optional[HyperparameterConfig] = Field(default=None, description="Hyperparameters")
    
    # LPIPS (optional) - can be string name or function object
    lpips_func: Optional[Union[str, Any]] = Field(default="alex", description="LPIPS function name or function object")
    
    dry_run: Optional[bool] = Field(default=False, description="Dry run, don't save output")

    resize_width: Optional[int] = Field(default=None, description="Resize width")
    resize_height: Optional[int] = Field(default=None, description="Resize height")
    colormode: Optional[str] = Field(default=None, description="Color mode")
    available_devices: Optional[List[str]] = Field(default=["cpu", "cuda", "mps"], description="Available devices")
    
    def get_hash_function(self) -> Callable:
        if self.hash_function is None:
            raise ValueError("Hash function not specified")
        hash_function_map = {
            "ahash": generate_ahash_batched,
            "ahash_rgb": generate_ahash_rgb_batched,
            "dhash": generate_dhash_batched,
            "dhash_rgb": generate_dhash_rgb_batched,
            "phash": generate_phash_batched,
            "phash_rgb": generate_phash_rgb_batched,
            "pdq": generate_pdq_batched
        }
        return hash_function_map[self.hash_function]


    def get_optimizer(self, config: Optimizer_Config) -> Optimizer:
        if self.attack_type is None:
            raise ValueError("Attack type not specified")
        optimizer_map = {
            "nes": NES_Optimizer,
            "nes_signed": NES_Signed_Optimizer,
            "colinear": Colinear_Optimizer
        }
        return optimizer_map[self.attack_type](config=config)


    def check_and_dump(self) -> Dict[str, Any]:
        """Check and dump the attack configuration, asserting all members are not None"""
        for field_name, field_value in self.__dict__.items():
            if field_name != "gate" and field_name != "lpips_func" and field_value is None:
                raise ValueError(f"AttackConfig field '{field_name}' is None. All fields must be set before dumping.")
        return self.model_dump()


    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True



class ExperimentConfig(BaseModel):
    """Configuration for a complete experiment"""
    experiment_name: Optional[str] = Field(default=None, description="Experiment name")
    experiment_description: Optional[str] = Field(default='', description="Experiment description")
    engine_verbose: Optional[bool] = Field(default=False, description="Engine verbose")

    # Attacks
    attacks: List[AttackConfig] = Field(..., min_items=1, description="List of attack configurations")
    
    # Input/Output paths (optional, can be set later)
    experiment_input_dir: Optional[str] = Field(default=None, description="Input directory path")
    experiment_output_dir: Optional[str] = Field(default=None, description="Output directory path")

    def check_and_dump(self) -> Dict[str, Any]:
        """Check and dump the experiment configuration, asserting all members are not None"""
        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                raise ValueError(f"ExperimentConfig field '{field_name}' is None. All fields must be set before dumping.")
        return self.model_dump()

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True 