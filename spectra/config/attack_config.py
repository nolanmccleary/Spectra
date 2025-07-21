from pydantic import BaseModel, Field, validator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from spectra.hashes import generate_ahash_batched, generate_ahash_rgb_batched, generate_dhash_batched, generate_dhash_rgb_batched, generate_phash_batched, generate_phash_rgb_batched, generate_pdq_batched
from spectra.deltagrad import Optimizer, NES_Optimizer, NES_Signed_Optimizer, Colinear_Optimizer, Optimizer_Config, Gaussian_Optimizer
from spectra.lpips import ALEX_IMPORT, ALEX_ONNX
from spectra.utils import bool_tensor_delta, noop, byte_quantize
from spectra.utils.acceptance import lpips_acceptance, l2_acceptance, step_acceptance, latching_acceptance, dummy_acceptance


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
                raise ValueError(f"Sweep parameters must be (start, end, step) - got: {v}")
            start, end, step = v
            if start is None or (end is not None and start >= end):
                raise ValueError(f"Invalid sweep range: {v}")
        return v


class AttackConfig(BaseModel):
    """Configuration for a single attack in a given experiment"""
    # Attack identification
    attack_name: Optional[str] = Field(default=None, description="Unique name for this attack")
    hash_function: Optional[Union[str, Callable]] = Field(default=None, description="Hash function to attack")
    
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
    
    attack_type: Optional[Union[str, Any]] = Field(default=None, description="Attack type")

    # Function specifications
    acceptance_func: Optional[Union[str, Callable]] = Field(default=dummy_acceptance, description="Acceptance function name")
    
    # Hyperparameters
    hyperparameters: Optional[HyperparameterConfig] = Field(default=None, description="Hyperparameters")
    
    # LPIPS (optional) - can be string name or function object
    lpips_func: Optional[Union[str, Any]] = Field(default='alex_import', description="LPIPS function name or function object")
    loss_func: Optional[Union[str, Any]] = Field(default=bool_tensor_delta, description="Loss function name or function object")
    quant_func: Optional[Union[str, Any]] = Field(default=noop, description="Quantization function name or function object")

    dry_run: Optional[bool] = Field(default=False, description="Dry run, don't save output")

    resize_width: Optional[int] = Field(default=None, description="Resize width")
    resize_height: Optional[int] = Field(default=None, description="Resize height")
    colormode: Optional[str] = Field(default=None, description="Color mode")
    available_devices: Optional[List[str]] = Field(default=["cpu", "cuda", "mps"], description="Available devices")
    

    def get_hash_function(self) -> Callable:
        ret = self.hash_function
        if not isinstance(self.hash_function, Callable):
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
            
            if isinstance(self.hash_function, str):
                return hash_function_map[self.hash_function]

            else:
                raise ValueError(f"Invalid hash function type: {type(self.hash_function)}")
            
            self.hash_function = ret
        
        return ret


    def get_optimizer(self, config: Optimizer_Config) -> Optimizer:
        ret = self.attack_type
        if not isinstance(self.attack_type, Optimizer):
            if self.attack_type is None:
                raise ValueError("Attack type not specified")
            
            if isinstance(self.attack_type, str):
                optimizer_map = {
                    "nes": NES_Optimizer,
                    "nes_signed": NES_Signed_Optimizer,
                    "colinear": Colinear_Optimizer,
                    "gaussian": Gaussian_Optimizer
                }
                
                if self.attack_type not in optimizer_map.keys():
                    raise ValueError(f"'{self.attack_type}' not in set of valid optimizer handles: {optimizer_map.keys()}")
                ret = optimizer_map[self.attack_type](config=config)

            else:
                raise ValueError(f"Invalid attack type type: {type(self.attack_type)}")
            
            self.attack_type = ret
        
        return ret


    def get_acceptance_func(self, ao) -> Callable:
        ret = self.acceptance_func
        if not isinstance(self.acceptance_func, Callable):
        
            if self.acceptance_func is None:
                raise ValueError("Acceptance function not specified")
            
            if isinstance(self.acceptance_func, str):
                acceptance_map = {
                    'lpips' : lpips_acceptance,
                    'l2'    : l2_acceptance,
                    'latch' : latching_acceptance,
                    'step'  : step_acceptance,
                    'dummy' : dummy_acceptance,
                }

                if self.acceptance_func not in acceptance_map.keys():
                    raise ValueError(f"'{self.acceptance_func}' not in set of valid acceptance function handles: {acceptance_map.keys()}")
                ret = acceptance_map[self.acceptance_func]
            
            else:
                raise ValueError(f"Invalid acceptance function type: {type(self.acceptance_func)}")
            
            self.acceptance_func = ret

        return ret(ao)


    def get_lpips_func(self, ao) -> Callable:
        ret = self.lpips_func
        if not isinstance(self.lpips_func, Callable):
            if isinstance(self.lpips_func, str):
                if self.lpips_func == 'alex_import':
                    ao.log("Using ALEXNET import for LPIPS!")
                    ao.lpips_model = ALEX_IMPORT(ao.device)

                elif self.lpips_func == 'alex_onnx':
                    ao.log("Using ALEXNET ONNX for LPIPS!")
                    ao.lpips_model = ALEX_ONNX(ao.device)
            
                ret = ao.lpips_model.get_lpips
            
            else:
                raise ValueError(f"Invalid LPIPS function type: {type(self.lpips_func)}")
    
            self.lpips_func = ret

        return ret

    
    def get_loss_func(self) -> Callable:
        ret = self.loss_func
        
        if not isinstance(self.loss_func, Callable):
            if isinstance(self.loss_func, str):
                loss_func_map = {
                    "bool_tensor_delta": bool_tensor_delta, #TODO: Extend this to other loss functions
                }
                
                if self.loss_func not in loss_func_map.keys():
                    raise ValueError(f"'{self.loss_func}' not in set of valid loss function handles: {loss_func_map.keys()}")
                ret = loss_func_map[self.loss_func]
            
            else:
                raise ValueError("Invalid loss function type")
            self.loss_func = ret
        
        return ret

        
    def get_quant_func(self) -> Callable:
        ret = self.quant_func
        
        if not isinstance(self.quant_func, Callable):
            if isinstance(self.quant_func, str):
                quant_table = {
                    "byte_quantize" : byte_quantize,
                    "noop" : noop
                }
                if self.quant_func not in quant_table.keys():
                    raise ValueError(f"'{self.quant_func}' not in set of valid quantization function handles: {quant_table.keys()}")
                ret = quant_table[self.quant_func]
            
            else:
                raise ValueError("Invalid quantization function type")
            
            self.quant_func = ret
        
        return ret


    def check_and_dump(self) -> Dict[str, Any]:
        """Check and dump the attack configuration, asserting all members are not None"""        
        config_dict = self.model_dump()

        for key, value in config_dict.items():
            if key not in ["gate", "quant_func"] and value is None:
                raise ValueError(f"AttackConfig field '{key}' is None. All required fields must be set before a dump is taken.")
        
        return config_dict


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