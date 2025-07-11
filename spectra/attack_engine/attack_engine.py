import json
from spectra.config import AttackConfig
from spectra.deltagrad import NES_Signed_Optimizer, NES_Optimizer
from spectra.deltagrad.utils import anal_clamp
from spectra.hashes import Hash_Wrapper
from spectra.validation import image_compare
from pathlib import Path
from PIL import Image
from spectra.utils import get_rgb_tensor, tensor_resize, to_hex, bool_tensor_delta, l2_delta, generate_acceptance, generate_conversion, generate_inversion, generate_quant, create_sweep
import torch
from torchvision.transforms import ToPILImage
from dataclasses import dataclass
from typing import List, Dict, Any
from spectra.lpips import ALEX_IMPORT, ALEX_ONNX

# Constants
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class AttackRunConfig:
    """Configuration for a single attack run"""
    images: List[str]
    input_dir: str
    output_dir: str
    attack_object: 'Attack_Object'


class Attack_Engine:
    """Manages multiple attacks and their execution"""
    
    def __init__(self, verbose: str):
        self.attacks: Dict[str, AttackRunConfig] = {}
        self.attack_log: Dict[str, Dict[str, Any]] = {}
        self.verbose = verbose


    def log(self, msg: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.verbose == "on":
            print(msg)

    
    def add_attack_from_config(self, attack_tag: str, hash_wrapper: Hash_Wrapper, config: AttackConfig) -> None:
        """Register a new attack configuration using AttackConfig object"""
        input_path = Path(config.input_dir)
        images = [
            f.name for f in input_path.iterdir() 
            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        
        attack_object = Attack_Object(hash_wrapper, config=config)
        self.attacks[attack_tag] = AttackRunConfig(
            images=images,
            input_dir=config.input_dir,
            output_dir=config.output_dir,
            attack_object=attack_object
        )


    def _calculate_averages(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average metrics from attack results"""
        successful_results = [r for r in results if r["pre_validation"]["success"] is True]
        
        if not successful_results:
            return {}
        
        metrics = {
            "phash_hamming": 0,
            "ahash_hamming": 0,
            "dhash_hamming": 0,
            "pdq_hamming": 0,
            "lpips": 0.0,
            "l2": 0.0,
            "ideal_beta": 0,
            "ideal_scale_factor": 0,
            "num_steps": 0
        }
        
        for result in successful_results:
            pre_val = result["pre_validation"]
            post_val = result["post_validation"]
            
            metrics["phash_hamming"] += int(post_val["phash_hamming"])
            metrics["ahash_hamming"] += int(post_val["ahash_hamming"])
            metrics["dhash_hamming"] += int(post_val["dhash_hamming"])
            metrics["pdq_hamming"] += int(post_val["pdq_hamming"])
            metrics["lpips"] += float(post_val["lpips"])
            metrics["l2"] += float(post_val["l2"])
            metrics["ideal_beta"] += float(pre_val["ideal_beta"])
            metrics["ideal_scale_factor"] += float(pre_val["ideal_scale_factor"])
            metrics["num_steps"] += float(pre_val["num_steps"])
        
        count = len(successful_results)
        return {f"average_{k}": v / count for k, v in metrics.items()}


    def run_attacks(self, output_name: str = "spectra_out") -> None:
        """Execute all registered attacks and save results"""
        for attack_tag, config in self.attacks.items():
            self.attack_log[attack_tag] = {
                "per_image_results": {},
                "average_results": {}
            }
            
            # Run attack on each image
            for image_name in config.images:
                input_path = f"{config.input_dir}/{image_name}"
                output_path = f"{config.output_dir}/{attack_tag}_{image_name}"
                
                result = config.attack_object.run_attack(input_path, output_path)
                self.attack_log[attack_tag]["per_image_results"][image_name] = result
            
            # Calculate averages
            results_list = list(self.attack_log[attack_tag]["per_image_results"].values())
            self.attack_log[attack_tag]["average_results"] = self._calculate_averages(results_list)

        # Save results to JSON
        json_filename = f"{output_name}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.attack_log, f, indent=4)
        print(f"Attack log saved to {json_filename}")



class Attack_Object:

    # Constants
    VALID_DEVICES = {"cpu", "cuda", "mps"}
    VALID_VERBOSITIES = {"on", "off"}
    DELTA_SCALEDOWN_STEPS = 50

    def __init__(self, hash_wrapper: Hash_Wrapper, config=None, **kwargs):
        """Initialize attack configuration
        
        Args:
            hash_wrapper: Hash function wrapper
            config: AttackConfig object (preferred) or individual parameters via kwargs
        """
        self._init_from_config(hash_wrapper, config)


    def _init_from_config(self, hash_wrapper: Hash_Wrapper, config):
        """Initialize from AttackConfig object"""
        from spectra.config import AttackConfig
        
        if not isinstance(config, AttackConfig):
            raise ValueError("config must be an AttackConfig object")
        
        # Core attack parameters
        self.hamming_threshold = config.hamming_threshold
        self.colormode = config.colormode
        self.device = config.device
        self.verbose = "on" if config.verbose else "off"
        self.delta_scaledown = config.delta_scaledown
        self.gate = config.gate
        
        # Hash function setup
        self._setup_hash_function(hash_wrapper)
        
        # Function generators
        self.acceptance_func = generate_acceptance(self, config.acceptance_func)
        self.quant_func = generate_quant(config.quant_func)
        
        # Attack parameters
        self.num_reps = config.num_reps
        self.attack_cycles = config.attack_cycles
        self.resize_flag = self.resize_height > 0 and self.resize_width > 0
        
        # LPIPS setup
        self._setup_lpips(config.lpips_func)
        self.l2_func = l2_delta
        
        # Hyperparameters
        self._setup_hyperparameters_from_config(config.hyperparameters)
        
        # Optimization packages
        self.func_package = (self.func, bool_tensor_delta, self.quant_func)
        self.device_package = (self.func_device, self.device, self.device)


    def _validate_inputs(self, device: str, verbose: str) -> None:
        """Validate input parameters"""
        if device not in self.VALID_DEVICES:
            raise ValueError(f"Invalid device '{device}'. Expected one of: {self.VALID_DEVICES}")
        if verbose not in self.VALID_VERBOSITIES:
            raise ValueError(f"Invalid verbosity '{verbose}'. Expected one of: {self.VALID_VERBOSITIES}")


    def _setup_hash_function(self, hash_wrapper: Hash_Wrapper) -> None:
        """Setup hash function and device compatibility"""
        self.func, self.resize_height, self.resize_width, available_devices = hash_wrapper.get_info()
        
        if self.device in available_devices:
            self.func_device = self.device
        else:
            self.log(f"Warning, current hash function '{hash_wrapper.get_name()}' does not support the chosen device {self.device}. Defaulting to CPU for hash function calls; this will add overhead.")
            self.func_device = "cpu"


    def _setup_lpips(self, lpips_func) -> None:
        
        """Setup LPIPS function for perceptual similarity"""
        if lpips_func is not None:
            # Handle both function objects and string names
            if callable(lpips_func):
                self.lpips_func = lpips_func
            
            else:
                print(lpips_func)
                self.lpips_model = None
                
                if lpips_func == 'alex_import':
                    self.lpips_model = ALEX_IMPORT(self.device)

                elif lpips_func == 'alex_onnx':
                    self.lpips_model = ALEX_ONNX(self.device)

                else:
                    raise KeyError("Error! Invalid LPIPS function selection")
        else:
            self.log("\nNo LPIPS selected! Defaulting to ALEXNET import!\n")
            self.lpips_model = ALEX_IMPORT(self.device)
        
        self.lpips_func = self.lpips_model.get_lpips


    def _setup_hyperparameters(self, hyperparameter_set: dict) -> None:
        """Setup hyperparameters from configuration"""
        self.alpha = hyperparameter_set["alpha"]
        self.betas = create_sweep(*hyperparameter_set["beta"])
        self.step_coeff = hyperparameter_set["step_coeff"]
        self.scale_factors = create_sweep(*hyperparameter_set["scale_factor"])
    

    def _setup_hyperparameters_from_config(self, hyperparameters) -> None:
        """Setup hyperparameters from HyperparameterConfig object"""
        self.alpha = hyperparameters.alpha
        
        # Handle beta (can be single value or sweep parameters)
        if isinstance(hyperparameters.beta, (list, tuple)):
            self.betas = create_sweep(*hyperparameters.beta)
        else:
            self.betas = [hyperparameters.beta]
        
        self.step_coeff = hyperparameters.step_coeff
        
        # Handle scale_factor (can be single value or sweep parameters)
        if isinstance(hyperparameters.scale_factor, (list, tuple)):
            self.scale_factors = create_sweep(*hyperparameters.scale_factor)
        else:
            self.scale_factors = [hyperparameters.scale_factor]


    def log(self, msg: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.verbose == "on":
            print(msg)


    def _reset_state(self) -> None:
        """Reset all state variables for a new attack"""
        self.lpips_model = None
        
        # Input tensors
        self.rgb_tensor = None
        self._tensor = None
        self.original_hash = None
        
        # Dimensions
        self.original_height = None
        self.original_width = None
        self.height = None
        self.width = None
        
        # Output tensors
        self.output_tensor = None
        self.output_hash = None
        self.output_hamming = None
        
        # Metrics
        self.output_lpips = 1.0
        self.output_l2 = 1.0
        self.min_steps = self.attack_cycles
        
        # Current state
        self.current_hash = None
        self.current_hamming = None
        self.current_lpips = 1.0
        self.current_l2 = 1.0
        
        # Attack state
        self.attack_success = False
        self.prev_step = None
        self.optimizer = None
        self.is_staged = False


    def _load_and_process_image(self, input_image_path: str) -> None:
        """Load image and setup tensors for attack"""
        with Image.open(input_image_path) as img:
            self.rgb_tensor = get_rgb_tensor(img, self.device)
            self.original_height = self.rgb_tensor.size(1)
            self.original_width = self.rgb_tensor.size(2)
            
            self.log("Setting grayscale image tensor")
            
            # Setup conversion functions
            self.inversion_func = generate_inversion(self.colormode)
            self.conversion_func = generate_conversion(self.colormode)
            
            # Convert to target color space
            self._tensor = self.conversion_func(self.rgb_tensor)
            
            # Resize if needed
            if self.resize_flag:
                self._tensor = tensor_resize(self._tensor, self.resize_height, self.resize_width)
                self.height = self.resize_height
                self.width = self.resize_width
            else:
                self.height = self.original_height
                self.width = self.original_width
            
            # Generate original hash
            self.original_hash = self.func(self._tensor.to(self.func_device))


    def stage_attack(self, input_image_path: str) -> None:
        """Prepare attack by loading image and setting up optimizer"""
        self._reset_state()
        self.log("Staging attack...\n")
        
        # Load and process image
        self._load_and_process_image(input_image_path)
        
        # Setup optimizer
        self.optimizer = NES_Optimizer(
            func_package=self.func_package, 
            device_package=self.device_package, 
            tensor=self._tensor, 
            vecMin=0.0, 
            vecMax=1.0
        )
        
        # Calculate number of perturbations (must be even)
        self.num_perturbations = self.alpha
        for k in self._tensor.shape:
            self.num_perturbations *= k
        self.num_perturbations = (int(self.num_perturbations) // 2) * 2


    def _apply_delta_scaledown(self, rgb_delta: torch.Tensor) -> None:
        """Apply delta scaledown to fine-tune the attack"""
        scale_factors = torch.linspace(0.0, 1.0, steps=self.DELTA_SCALEDOWN_STEPS)
        
        for scale in scale_factors:
            cand_delta = rgb_delta * scale
            cand_tensor = self.rgb_tensor + cand_delta
            cand_targ = self.conversion_func(cand_tensor.clone())

            if self.resize_flag:
                cand_targ = tensor_resize(cand_targ, self.resize_height, self.resize_width)

            cand_tensor = self.quant_func(cand_tensor)
            cand_targ = self.quant_func(cand_targ)

            cand_hash = self.func(cand_targ.to(self.func_device))
            cand_ham = cand_hash.ne(self.original_hash).sum().item()
            if cand_ham >= self.hamming_threshold:
                self.output_tensor = cand_tensor
                self.output_hash = cand_hash
                self.output_hamming = cand_ham
                self.attack_success = True
                break


    def run_attack(self, input_image_path: str, output_image_path: str) -> Dict[str, Any]:
        self.stage_attack(input_image_path)
        self.log("Running attack...\n")

        ret_set = (None, None, None, None)

        self.log(f"Beta sweep across: {self.betas}\n")
        self.log(f"Perturbation scale factor sweep across: {self.scale_factors}\n")

        for beta in self.betas:
            for scale_factor in self.scale_factors:

                for rep in range(self.num_reps):
                    step_count, curr_delta, accepted = self.optimizer.get_delta(
                        step_coeff=self.step_coeff,
                        num_steps=self.attack_cycles,
                        perturbation_scale_factor=scale_factor,
                        num_perturbations=self.num_perturbations,
                        beta=beta, 
                        acceptance_func=self.acceptance_func
                    )
                    
                    if accepted or ret_set[0] is None:  # We get the acceptance best out of our entire sweep space for our output tensor
                        ret_set = (curr_delta, step_count, beta, scale_factor)
                        self.log(f"Accepted: steps={ret_set[1]}, beta={ret_set[2]}, scale={ret_set[3]}")


        ################################ RTQ - FROM HASH SPACE TO IMAGE SPACE #####################
        output_delta = ret_set[0]
        
        if output_delta is not None:

            if self.resize_flag:
                optimal_delta = output_delta.view(3 if self.colormode == "rgb" else 1, self.height, self.width)
                output_delta = tensor_resize(optimal_delta, self.original_height, self.original_width)
            
            rgb_delta = self.inversion_func(self.rgb_tensor, output_delta)
            safe_scale = anal_clamp(self.rgb_tensor, rgb_delta, 0.0, 1.0)

            self.output_tensor = self.rgb_tensor + rgb_delta * safe_scale

            cand_targ = self.conversion_func(self.output_tensor)
            if self.resize_flag:
                cand_targ = tensor_resize(cand_targ, self.resize_height, self.resize_width)

            self.output_hash = self.func(self.quant_func(cand_targ))
            self.output_hamming = self.original_hash.ne(self.output_hash.to(self.original_hash.device)).sum().item()
            self.attack_success = self.output_hamming >= self.hamming_threshold

            # Optional delta scaledown for fine-tuning
            if self.delta_scaledown:
                self._apply_delta_scaledown(rgb_delta)

            self.output_lpips = self.lpips_func(self.rgb_tensor, self.output_tensor)
            self.output_l2 = self.l2_func(self.rgb_tensor, self.output_tensor)
            
            out = self.output_tensor.detach()
            output_image = ToPILImage()(out)
            output_image.save(output_image_path)
            
            self.log(f"Saved attacked image to {output_image_path}")
            self.log(f"Success status: {self.attack_success}")


        def null_guard(input):
            if input is None:
                return "N/A"
            else:
                return input

        out_log = {
            "pre_validation": {
                "success"               : null_guard(self.attack_success),
                "original_hash"         : null_guard(to_hex(self.original_hash)),
                "output_hash"           : null_guard(to_hex(self.output_hash) if self.output_hash is not None else None),
                "hamming_distance"      : null_guard(self.output_hamming),
                "lpips"                 : null_guard(self.output_lpips),
                "l2"                    : null_guard(self.output_l2),
                "num_steps"             : null_guard(ret_set[1]),
                "ideal_scale_factor"    : null_guard(ret_set[3]),
                "ideal_beta"            : null_guard(ret_set[2])
            },
            "post_validation": image_compare(input_image_path, output_image_path, self.lpips_func, self.device, verbose = "off")
        }

        self.log(out_log)
        return out_log