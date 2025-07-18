from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Tuple, Any
from enum import Enum
from spectra.hashes import Hash_Wrapper, AHASH, DHASH, PHASH, PDQ, AHASH_RGB, DHASH_RGB, PHASH_RGB

class Device(str, Enum):
    """Supported devices for computation"""
    CPU = "cpu"
    CUDA = "cuda" 
    MPS = "mps"


class HashFunction(str, Enum):
    """Supported hash functions"""
    AHASH = "ahash"
    DHASH = "dhash"
    PHASH = "phash"
    PDQ = "pdq"
    AHASH_RGB = "ahash_rgb"
    DHASH_RGB = "dhash_rgb"
    PHASH_RGB = "phash_rgb"


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
    attack_name: str = Field(..., description="Unique name for this attack")
    hash_function: HashFunction = Field(..., description="Hash function to attack")
    
    # Core parameters
    hamming_threshold: int = Field(..., ge=0, description="Minimum hamming distance required")
    device: Device = Field(...)
    
    # Attack parameters
    num_reps: int = Field(..., gt=0, description="Number of repetitions")
    attack_cycles: int = Field(..., gt=0, description="Number of attack cycles")
    
    # Optional features
    delta_scaledown: bool = Field(default=False, description="Enable delta scaledown")
    gate: Optional[str] = Field(default=None, description="Gate function name")
    attack_verbose: bool = Field(default=False, description="Enable verbose logging")
    deltagrad_verbose: bool = Field(default=False, description="Enable verbose logging for deltagrad components")
    
    # Function specifications
    acceptance_func: str = Field(..., description="Acceptance function name")
    quant_func: Optional[str] = Field(default=None, description="Quantization function name")
    
    # Hyperparameters
    hyperparameters: HyperparameterConfig
    
    # LPIPS (optional) - can be string name or function object
    lpips_func: Optional[Union[str, Any]] = Field(default=None, description="LPIPS function name or function object")
    
    dry_run: bool = Field(default=False, description="Dry run, don't save output")
    
    def get_hash_wrapper(self) -> Hash_Wrapper:
        hash_wrapper_map = {
            HashFunction.AHASH: AHASH,
            HashFunction.DHASH: DHASH,
            HashFunction.PHASH: PHASH,
            HashFunction.PDQ: PDQ,
            HashFunction.AHASH_RGB: AHASH_RGB,
            HashFunction.DHASH_RGB: DHASH_RGB,
            HashFunction.PHASH_RGB: PHASH_RGB
        }
        return hash_wrapper_map[self.hash_function]

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True


class ExperimentConfig(BaseModel):
    """Configuration for a complete experiment"""
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(default=None, description="Experiment description")
    
    # Attacks
    attacks: List[AttackConfig] = Field(..., min_items=1, description="List of attack configurations")
    
    # Input/Output paths (optional, can be set later)
    input_dir: str = Field(..., description="Input directory path")
    output_dir: str = Field(..., description="Output directory path")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True 