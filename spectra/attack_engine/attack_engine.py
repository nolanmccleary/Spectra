import json
import os
from spectra.config import AttackConfig, ExperimentConfig
from spectra.deltagrad import Optimizer_Config, Delta_Config
from spectra.deltagrad.utils import anal_clamp
from spectra.validation import image_compare
from pathlib import Path
from PIL import Image
from spectra.utils import get_rgb_tensor, tensor_resize, to_hex, l2_delta, generate_conversion, generate_inversion, create_sweep, make_json_serializable
import torch
from torchvision.transforms import ToPILImage
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Constants
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class InputTensors:
    """Input tensor state"""
    rgb_tensor: Optional[torch.Tensor] = None
    working_tensor: Optional[torch.Tensor] = None
    original_hash: Optional[torch.Tensor] = None


@dataclass
class Dimensions:
    """Image dimensions"""
    original_height: Optional[int] = None
    original_width: Optional[int] = None
    working_height: Optional[int] = None
    working_width: Optional[int] = None


@dataclass
class OutputTensors:
    """Output tensor state"""
    output_tensor: Optional[torch.Tensor] = None
    output_hash: Optional[torch.Tensor] = None
    output_hamming: Optional[int] = None


@dataclass
class Metrics:
    """Attack metrics"""
    min_steps: int # Default to attack_cycles
    output_lpips: float = 1.0
    output_l2: float = 1.0
    current_hash: Optional[torch.Tensor] = None
    current_hamming: Optional[int] = None
    current_lpips: float = 1.0
    current_l2: float = 1.0


@dataclass
class AttackRunConfig:
    """Configuration for a single attack run"""
    images: List[str]
    input_dir: str
    output_dir: str
    attack_object: 'Attack_Object'


@dataclass
class AttackRetSet:
    ideal_delta: torch.Tensor
    step_count: int
    ideal_alpha: float
    ideal_step_coeff: float
    ideal_beta: float
    ideal_scale_factor: float



class Attack_Engine:
    """Manages multiple attacks and their execution"""
    
    def __init__(self, verbose: str):
        self.attacks: Dict[str, AttackRunConfig] = {}
        self.attack_log: Dict[str, Dict[str, Any]] = {}
        self.experiment_configs: List[ExperimentConfig] = []
        self.current_experiment = None

    def log(self, experiment_index: int, msg: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.experiment_configs[experiment_index].engine_verbose == "on":
            print(msg)


    def load_experiment_from_config(self, experiment_index: int, config: ExperimentConfig, **overrides) -> None:
        """Load an experiment from a configuration object with optional parameter overrides"""
        self.experiment_configs.append(config)
        
        direct_params = ['experiment_name', 'experiment_description', 'experiment_input_dir', 'experiment_output_dir', 'engine_verbose']
        for param in direct_params:
            override_key = f'force_{param}'
            if override_key in overrides and overrides[override_key] is not None:
                self.log(experiment_index, f"Overriding {param} to {overrides[override_key]}")
                setattr(self.experiment_configs[experiment_index], param, overrides[override_key])

        for attack_config in config.attacks:
            # Apply attack-level overrides
            self._apply_overrides_to_config(experiment_index, attack_config, overrides)
            self.add_attack_from_config(experiment_index, attack_config)


    def _apply_overrides_to_config(self, experiment_index: int, config: AttackConfig, overrides: dict) -> None:
        """Apply overrides to attack configuration"""
        # Direct parameter overrides
        direct_params = ['device', 'dry_run',
                        'attack_cycles', 'num_reps', 'gate', 'acceptance_func', 
                        'quant_func', 'lpips_func', 'attack_verbose', 'deltagrad_verbose', 'loss_func',
                        'hamming_threshold', 'attack_type', 'attack_name', 'hash_function', 'delta_scaledown',
                        'resize_width', 'resize_height', 'colormode', 'available_devices']
        
        for param in direct_params:
            override_key = f'force_{param}'
            if override_key in overrides and overrides[override_key] is not None:
                self.log(experiment_index, f"Overriding {param} to {overrides[override_key]}")
                setattr(config, param, overrides[override_key])
        
        # Hyperparameter overrides
        hyperparam_mapping = {
            'force_hyperparameters_alpha': 'alpha',
            'force_hyperparameters_beta': 'beta', 
            'force_hyperparameters_step_coeff': 'step_coeff',
            'force_hyperparameters_scale_factor': 'scale_factor'
        }
        for override_key, hyperparam in hyperparam_mapping.items():
            if override_key in overrides and overrides[override_key] is not None:
                setattr(config.hyperparameters, hyperparam, overrides[override_key])


    def add_attack_from_config(self, experiment_index: int, config: AttackConfig) -> None:
        """Register a new attack configuration using AttackConfig object with optional verbosity and device overrides"""
        assert self.experiment_configs[experiment_index] is not None, "Experiment not loaded"
        
        self.log(experiment_index, f"Experiment input dir: {self.experiment_configs[experiment_index].experiment_input_dir}")
        input_path = Path(str(self.experiment_configs[experiment_index].experiment_input_dir))
        images = [
            f.name for f in input_path.iterdir() 
            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        
        attack_object = Attack_Object(config=config)
        
        self.attacks[config.attack_name] = AttackRunConfig(
            images=images,
            input_dir=self.experiment_configs[experiment_index].experiment_input_dir,
            output_dir=self.experiment_configs[experiment_index].experiment_output_dir,
            attack_object=attack_object
        )


    def _calculate_averages(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average metrics from attack results"""
        successful_results = [r for r in results if r["pre_validation"]["success"] is True]
        
        if not successful_results:
            return {}
        
        metrics = {
            "ahash_hamming_torch": 0,
            "dhash_hamming_torch": 0,
            "phash_hamming_torch": 0,
            "pdq_hamming_torch": 0,
            "ahash_hamming_cannonical": 0,
            "dhash_hamming_cannonical": 0,
            "pdq_hamming_cannonical": 0,
            "phash_hamming_cannonical": 0,
            "ahash_discrepency": 0,
            "dhash_discrepency": 0,
            "phash_discrepency": 0,
            "pdq_discrepency": 0,
            "lpips": 0.0,
            "l2": 0.0,
            "ideal_step_coeff": 0,
            "ideal_alpha": 0,
            "ideal_beta": 0,
            "ideal_scale_factor": 0,
            "num_steps": 0
        }
        
        for result in successful_results:
            pre_val = result["pre_validation"]
            post_val = result["post_validation"]
            
            metrics["ahash_hamming_torch"] += int(post_val["ahash_hamming_torch"])
            metrics["dhash_hamming_torch"] += int(post_val["dhash_hamming_torch"])
            metrics["pdq_hamming_torch"] += int(post_val["pdq_hamming_torch"])
            metrics["phash_hamming_torch"] += int(post_val["phash_hamming_torch"])
            metrics["ahash_hamming_cannonical"] += int(post_val["ahash_hamming_cannonical"])
            metrics["dhash_hamming_cannonical"] += int(post_val["dhash_hamming_cannonical"])
            metrics["pdq_hamming_cannonical"] += int(post_val["pdq_hamming_cannonical"])
            metrics["phash_hamming_cannonical"] += int(post_val["phash_hamming_cannonical"])
            metrics["ahash_discrepency"] += int(post_val["ahash_discrepency"])
            metrics["dhash_discrepency"] += int(post_val["dhash_discrepency"])
            metrics["phash_discrepency"] += int(post_val["phash_discrepency"])
            metrics["pdq_discrepency"] += int(post_val["pdq_discrepency"])
            metrics["lpips"] += float(post_val["lpips"])
            metrics["l2"] += float(post_val["l2"])
            metrics["ideal_step_coeff"] += float(pre_val["ideal_step_coeff"])
            metrics["ideal_alpha"] += float(pre_val["ideal_alpha"])
            metrics["ideal_beta"] += float(pre_val["ideal_beta"])
            metrics["ideal_scale_factor"] += float(pre_val["ideal_scale_factor"])
            metrics["num_steps"] += float(pre_val["num_steps"])
        
        count = len(successful_results)
        return {f"average_{k}": v / count for k, v in metrics.items()}


    def run_experiment(self, experiment_index: int) -> None:
        """Execute all registered attacks and save results"""
        self.current_experiment = experiment_index
        
        assert self.experiment_configs[self.current_experiment] is not None, "Experiment not loaded"
        
        
        experiment_date = datetime.now().strftime("%Y-%m-%d")
        experiment_time = datetime.now().strftime("%H:%M:%S")

        # Create output directories
        experiment_dir = f"{self.experiment_configs[self.current_experiment].experiment_output_dir}/{self.experiment_configs[self.current_experiment].experiment_name}_{str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))}"
        images_dir = f"{experiment_dir}/images"
        results_dir = f"{experiment_dir}/results"
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        self.attack_log = {"experiment_config": self.experiment_configs[self.current_experiment].check_and_dump(), "attacks": {}}
        
        for attack_tag, config in self.attacks.items():
            self.attack_log["attacks"][attack_tag] = {
                "attack_config": config.attack_object.config.check_and_dump(),
                "per_image_results": {},
                "average_results": {}
            }
            
            # Run attack on each image
            for image_name in config.images:
                input_path = f"{self.experiment_configs[self.current_experiment].experiment_input_dir}/{image_name}"
                output_path = f"{images_dir}/{attack_tag}_{image_name}"
                result = config.attack_object.run_attack(input_path, output_path)
                self.attack_log["attacks"][attack_tag]["per_image_results"][image_name] = result
            
            # Calculate averages
            results_list = list(self.attack_log["attacks"][attack_tag]["per_image_results"].values())
            if config.attack_object.config.dry_run == False:
                self.attack_log["attacks"][attack_tag]["average_results"] = self._calculate_averages(results_list)

        # Save results to JSON
        json_filename = f"{results_dir}/results.json"

        experiment_endtime = datetime.now()
        experiment_start_time = datetime.strptime(experiment_date + " " + experiment_time, "%Y-%m-%d %H:%M:%S")
        experiment_runtime = experiment_endtime - experiment_start_time

        
        self.attack_log["metadata"] = {
            "experiment_name": self.experiment_configs[self.current_experiment].experiment_name,
            "experiment_description": self.experiment_configs[self.current_experiment].experiment_description,
            "experiment_date": experiment_date,
            "experiment_time": experiment_time,
            "experiment_runtime": str(experiment_runtime)
        }

        with open(json_filename, 'w') as f:
            json.dump(make_json_serializable(self.attack_log), f, indent=4)
    
        self.log(self.current_experiment, f"Attack log saved to {json_filename}")



class Attack_Object:

    # Constants
    VALID_DEVICES = {"cpu", "cuda", "mps"}
    VALID_VERBOSITIES = {"on", "off"}
    DELTA_SCALEDOWN_STEPS = 50

    def __init__(self, config: AttackConfig = None, **kwargs):
        """Initialize attack configuration
        
        Args:
            hash_wrapper: Hash function wrapper
            config: AttackConfig object (preferred) or individual parameters via kwargs
        """
        self._init_from_config(config)


    def _init_from_config(self, config: AttackConfig):
        """Initialize from AttackConfig object"""
        if not isinstance(config, AttackConfig):
            raise ValueError("config must be an AttackConfig object")
        
        self.config = config

        # Core attack parameters
        self.hamming_threshold = config.hamming_threshold
        self.device = config.device
        self.verbose = "on" if config.attack_verbose else "off"
        self.deltagrad_verbose = "on" if config.deltagrad_verbose else "off"
        self.delta_scaledown = config.delta_scaledown
        self.gate = config.gate
        
        # Hash function setup
        self._setup_hash_params()
        
        # Function generators
        self.acceptance_func = self.config.get_acceptance_func(self)
        self.quant_func = self.config.get_quant_func()
        self.loss_func = self.config.get_loss_func()
        self.quant_func_device = self.device
        self.loss_func_device = self.device

        # Attack parameters
        self.num_reps = config.num_reps
        self.attack_cycles = config.attack_cycles
        self.resize_flag = self.resize_height > 0 and self.resize_width > 0
        
        # LPIPS setup
        self.lpips_func = self.config.get_lpips_func(self)
        self.l2_func = l2_delta
        
        # Hyperparameters
        self._setup_hyperparameters_from_config(config)
        

    def _validate_inputs(self, device: str, verbose: str) -> None:
        """Validate input parameters"""
        if device not in self.VALID_DEVICES:
            raise ValueError(f"Invalid device '{device}'. Expected one of: {self.VALID_DEVICES}")
        if verbose not in self.VALID_VERBOSITIES:
            raise ValueError(f"Invalid verbosity '{verbose}'. Expected one of: {self.VALID_VERBOSITIES}")


    def _setup_hash_params(self) -> None:
        """Setup hash function and device compatibility"""
        self.hash_func = self.config.get_hash_function()
        self.resize_height = self.config.resize_height
        self.resize_width = self.config.resize_width
        available_devices = self.config.available_devices
        self.colormode = self.config.colormode
        
        if self.device in available_devices:
            self.hash_func_device = self.device
        else:
            self.log(f"Warning, current hash function '{self.config.hash_function}' does not support the chosen device {self.device}. Defaulting to CPU for hash function calls; this will add overhead.")
            self.hash_func_device = "cpu"



    def _setup_hyperparameters_from_config(self, config: AttackConfig) -> None:
        """Setup hyperparameters from HyperparameterConfig object"""
        self.alphas = create_sweep(config.hyperparameters.alpha)
        self.step_coeffs = create_sweep(config.hyperparameters.step_coeff)
        self.betas = create_sweep(config.hyperparameters.beta)
        self.scale_factors = create_sweep(config.hyperparameters.scale_factor)


    def log(self, msg: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.verbose == "on":
            print(msg)


    def _reset_state(self) -> None:
        """Reset all state variables for a new attack"""
        self.lpips_model = None
        self.input_tensors = InputTensors()
        self.dimensions = Dimensions()
        self.output_tensors = OutputTensors()
        self.metrics = Metrics(min_steps=self.attack_cycles)
        self.input_tensors = InputTensors()
        self.dimensions = Dimensions()
        self.attack_success = False
        self.prev_step = None
        self.optimizer = None
        self.is_staged = False


    def _load_and_process_image(self, input_image_path: str) -> None:
        """Load image and setup tensors for attack"""
        with Image.open(input_image_path) as img:
            self.input_tensors.rgb_tensor = get_rgb_tensor(img, self.device)
            self.dimensions.original_height = self.input_tensors.rgb_tensor.size(1)
            self.dimensions.original_width = self.input_tensors.rgb_tensor.size(2)
            
            self.log("Setting grayscale image tensor")
            
            # Setup conversion functions
            self.inversion_func = generate_inversion(self.colormode)
            self.conversion_func = generate_conversion(self.colormode)
            
            # Convert to target color space
            self.input_tensors.working_tensor = self.conversion_func(self.input_tensors.rgb_tensor)
            
            # Resize if needed
            if self.resize_flag:
                self.input_tensors.working_tensor = tensor_resize(self.input_tensors.working_tensor, self.resize_height, self.resize_width)
                self.dimensions.working_height = self.resize_height
                self.dimensions.working_width = self.resize_width
            else:
                self.dimensions.working_height = self.dimensions.original_height
                self.dimensions.working_width = self.dimensions.original_width
            
            # Generate original hash
            self.input_tensors.original_hash = self.hash_func(self.input_tensors.working_tensor.to(self.hash_func_device))


    def stage_attack(self, input_image_path: str) -> None:
        """Prepare attack by loading image and setting up optimizer"""
        self._reset_state()
        self.log("Staging attack...\n")
        
        # Load and process image
        self._load_and_process_image(input_image_path)

        optimizer_config = Optimizer_Config(
            func=self.hash_func,
            loss_func=self.loss_func,
            quant_func=self.quant_func,
            func_device=self.hash_func_device,
            loss_func_device=self.loss_func_device,
            quant_func_device=self.quant_func_device,
            verbose=self.deltagrad_verbose)

        # Setup optimizer
        self.optimizer = self.config.get_optimizer(optimizer_config)


    def _apply_delta_scaledown(self, rgb_delta: torch.Tensor) -> None:
        """Apply delta scaledown to fine-tune the attack"""
        scale_factors = torch.linspace(0.0, 1.0, steps=self.DELTA_SCALEDOWN_STEPS)
        
        for scale in scale_factors:
            cand_delta = rgb_delta * scale
            cand_tensor = self.input_tensors.rgb_tensor + cand_delta
            cand_targ = self.conversion_func(cand_tensor.clone())

            if self.resize_flag:
                cand_targ = tensor_resize(cand_targ, self.resize_height, self.resize_width)

            cand_tensor = self.quant_func(cand_tensor)
            cand_targ = self.quant_func(cand_targ)

            cand_hash = self.hash_func(cand_targ.to(self.hash_func_device))
            cand_ham = cand_hash.ne(self.input_tensors.original_hash).sum().item()
            if cand_ham >= self.hamming_threshold:
                self.output_tensors.output_tensor = cand_tensor
                self.output_tensors.output_hash = cand_hash
                self.output_tensors.output_hamming = cand_ham
                self.attack_success = True
                break


    def run_attack(self, input_image_path: str, output_image_path: str) -> Dict[str, Any]:
        """Run attack on a single image"""
        
        self.stage_attack(input_image_path)
        self.log("Running attack...\n")

        ret_set = AttackRetSet(None, 0, 0, 0, 0, 0)

        self.log(f"Step coefficient sweep across: {self.step_coeffs}\n")
        self.log(f"Alpha sweep across: {self.alphas}\n")
        self.log(f"Beta sweep across: {self.betas}\n")
        self.log(f"Perturbation scale factor sweep across: {self.scale_factors}\n")

        for alpha in self.alphas:
            num_perturbations = alpha
            for k in self.input_tensors.working_tensor.shape:
                num_perturbations *= k
            num_perturbations = (int(num_perturbations) // 2) * 2
            
            for step_coeff in self.step_coeffs:
                for beta in self.betas:
                    for scale_factor in self.scale_factors:
                        for rep in range(self.num_reps):
                            step_count, curr_delta, accepted = self.optimizer.get_delta(
                                tensor=self.input_tensors.working_tensor,
                                config=Delta_Config(
                                    step_coeff=step_coeff,
                                    num_steps=self.attack_cycles,
                                    perturbation_scale_factor=scale_factor,
                                    num_perturbations=num_perturbations,
                                    beta=beta, 
                                    acceptance_func=self.acceptance_func,
                                    vecMin=0.0,
                                    vecMax=1.0
                                )
                            )
                            
                            if accepted or ret_set.ideal_delta is None:  # We get the acceptance best out of our entire sweep space for our output tensor
                                ret_set.ideal_delta = curr_delta
                                ret_set.step_count = step_count
                                ret_set.ideal_beta = beta
                                ret_set.ideal_scale_factor = scale_factor
                                ret_set.ideal_step_coeff = step_coeff
                                ret_set.ideal_alpha = alpha
                            
                            self.log(f"Accepted={accepted}, steps={step_count}, beta={ret_set.ideal_beta}, scale={ret_set.ideal_scale_factor}, step_coeff={ret_set.ideal_step_coeff}, alpha={ret_set.ideal_alpha}")


        ################################ RTQ - FROM HASH SPACE TO IMAGE SPACE #####################
        output_delta = ret_set.ideal_delta.clone()
        if output_delta is not None:

            ret_delta = output_delta.clone()

            if self.resize_flag:
                optimal_delta = output_delta.view(3 if self.colormode == "rgb" else 1, self.dimensions.working_height, self.dimensions.working_width)
                ret_delta = tensor_resize(optimal_delta, self.dimensions.original_height, self.dimensions.original_width)

            rgb_delta = self.inversion_func(self.input_tensors.rgb_tensor, ret_delta).to(self.device)
            safe_scale = anal_clamp(self.input_tensors.rgb_tensor, rgb_delta, 0.0, 1.0).to(self.device)

            self.output_tensors.output_tensor = self.input_tensors.rgb_tensor + rgb_delta * safe_scale

            self.log(f"Cosine similarity between input and output: {torch.cosine_similarity(self.output_tensors.output_tensor.flatten(), self.input_tensors.rgb_tensor.flatten(), dim=0):.10f}")

            cand_targ = self.conversion_func(self.output_tensors.output_tensor)
            if self.resize_flag:
                cand_targ = tensor_resize(cand_targ, self.resize_height, self.resize_width)

            self.output_tensors.output_hash = self.hash_func(self.quant_func(cand_targ))

            self.output_tensors.output_hamming = self.input_tensors.original_hash.ne(self.output_tensors.output_hash.to(self.input_tensors.original_hash.device)).sum().item()
            self.attack_success = self.output_tensors.output_hamming >= self.hamming_threshold

            # Optional delta scaledown for fine-tuning
            if self.delta_scaledown:
                self._apply_delta_scaledown(rgb_delta)

            self.metrics.current_lpips = self.lpips_func(self.input_tensors.rgb_tensor, self.output_tensors.output_tensor)
            self.metrics.current_l2 = self.l2_func(self.input_tensors.rgb_tensor, self.output_tensors.output_tensor)
            
            if not self.config.dry_run:
                out = self.output_tensors.output_tensor.detach()
                output_image = ToPILImage()(out)
                output_image.save(output_image_path)
            
                self.log(f"Saved attacked image to {output_image_path}")
                self.log(f"Success status: {self.attack_success}")

        def null_guard(input):
            if input is None:
                return "N/A"
            else:
                return input

        # For dry runs, we can't do post-validation since we don't save the image
        post_validation = {}
        if self.config.dry_run == False:
            post_validation = image_compare(input_image_path, output_image_path, self.lpips_func, self.device, verbose = "off")

        out_log = {
            "pre_validation": {
                "success"               : null_guard(self.attack_success),
                "original_hash"         : null_guard(to_hex(self.input_tensors.original_hash)),
                "output_hash"           : null_guard(to_hex(self.output_tensors.output_hash) if self.output_tensors.output_hash is not None else None),
                "hamming_distance"      : null_guard(self.output_tensors.output_hamming),
                "lpips"                 : null_guard(self.metrics.current_lpips),
                "l2"                    : null_guard(self.metrics.current_l2),
                "num_steps"             : null_guard(ret_set.step_count),
                "ideal_scale_factor"    : null_guard(ret_set.ideal_scale_factor),
                "ideal_beta"            : null_guard(ret_set.ideal_beta),
                "ideal_step_coeff"      : null_guard(ret_set.ideal_step_coeff),
                "ideal_alpha"           : null_guard(ret_set.ideal_alpha)
            },
            "post_validation": post_validation
        }

        self.log(out_log)
        return out_log