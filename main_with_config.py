import os
import sys
import time
import torch
from spectra import Attack_Engine, PHASH, AHASH, DHASH, PDQ
from spectra.config import ConfigManager, ExperimentConfig
from models import ALEX_IMPORT


def attack_sequence_with_config(dev):
    """Run attacks using the new configuration system"""
    engine = Attack_Engine(verbose="on")
    
    # Setup LPIPS
    LPIPS_MODEL = ALEX_IMPORT(device=dev)
    F_LPIPS = LPIPS_MODEL.get_lpips
    
    # Load experiment configuration
    config_manager = ConfigManager()

    # Try to load the full experiment config
    experiment_config = config_manager.load_experiment_config("example_experiment")
    
    # Add attacks from configuration
    hash_wrappers = [AHASH, DHASH, PHASH, PDQ]
    attack_names = ["ahash_attack", "dhash_attack", "phash_attack", "pdq_attack"]
    
    for i, (hash_wrapper, attack_name) in enumerate(zip(hash_wrappers, attack_names)):
        if i < len(experiment_config.attacks):
            config = experiment_config.attacks[i]
            # Pass LPIPS function directly to add_attack_from_config
            engine.add_attack_from_config(attack_name, hash_wrapper, config)
    
    engine.add_attack_from_config("ahash_attack_2", AHASH, config_manager.load_attack_config("ahash_example")) #Can explicitly add additional attacks to the experiment if desired

    print(f"Loaded experiment: {experiment_config.name}")
    print(f"Description: {experiment_config.description}")
        
    t1 = time.time()
    engine.run_attacks()
    time_delta = time.time() - t1
    
    print(f"\nTest sequence completed in {time_delta:.2f} seconds")


def create_example_configs():
    """Create example configuration files"""
    config_manager = ConfigManager()
    
    # Create AHASH attack config
    from spectra.config import AttackConfig, HyperparameterConfig
    
    ahash_config = AttackConfig(
        hamming_threshold=24,
        colormode="grayscale",
        device="cpu",
        num_reps=1,
        verbose=True,
        lpips_func = 'alex_import',
        attack_cycles=10000,
        delta_scaledown=True,
        acceptance_func="lpips",
        input_dir="sample_images2",
        output_dir="output",
        quant_func=None,
        hyperparameters=HyperparameterConfig(
            alpha=2.9,
            beta=0.9,
            step_coeff=0.0001,
            scale_factor=0.4
        )
    )
    
    config_manager.save_attack_config(ahash_config, "ahash_example")
    print("Created ahash_example.yaml")
    
    # Create full experiment config
    from spectra.config import ExperimentConfig
    
    experiment_config = ExperimentConfig(
        name="Example Experiment",
        description="Example experiment with all hash functions",
        device="cpu",
        verbose=True,
        attacks=[
            AttackConfig(
                hamming_threshold=24,
                colormode="grayscale",
                device="cpu",
                num_reps=1,
                attack_cycles=10000,
                verbose=True,
                delta_scaledown=True,
                acceptance_func="lpips",
                quant_func=None,
                input_dir="sample_images3",
                output_dir="output",
                hyperparameters=HyperparameterConfig(
                    alpha=2.9,
                    beta=(0.9, None, None),
                    step_coeff=0.0001,
                    scale_factor=(0.4, None, None)
                )
            )
        ]
    )
    
    config_manager.save_experiment_config(experiment_config, "example_experiment")
    print("Created example_experiment.yaml")


if __name__ == '__main__':
    # Set default dtype
    torch.set_default_dtype(torch.float32)
    sys.path.append(os.path.join(os.path.dirname(__file__), "spectra/"))
    
    # Create example configs if they don't exist
    create_example_configs()
    
    # Run attack sequence
    attack_sequence_with_config("cpu") 