import argparse
import time
import torch
import sys
import os
import ast
from typing import Union, Tuple, Optional
from spectra import Attack_Engine
from spectra.config import ConfigManager


def parse_float_or_tuple(value: Union[str, float]) -> Union[float, Tuple[float, Optional[float], Optional[float]]]:
    """Parse a string as either a float or a tuple of floats"""
    try:
        ret = None
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            tuple_str = value[1:-1]
            parts = [part.strip() for part in tuple_str.split(',')]
            
            if len(parts) == 3:
                ret = (float(parts[0]), float(parts[1]), float(parts[2]))
            else:
                raise ValueError(f"Invalid tuple format: {value}")
        else:
            ret = (float(value), None, None)
    
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid value '{value}': {e}")
    
    return ret


def run_attacks(args):
    """Run attacks from experiment configuration files"""
    engine = Attack_Engine(verbose="on")
    
    # Load experiment configuration
    config_manager = ConfigManager()
    
    total_experiments = len(args.files)
    
    for i, experiment_file in enumerate(args.files):
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{total_experiments}: {experiment_file}")
        print(f"{'='*60}")
        
        # Load experiment configuration
        experiment_config = config_manager.load_experiment_config(experiment_file)
        
        # Add attacks from configuration with verbosity and device overrides
        overrides = {
            'force_engine_verbose': args.v1,
            'force_attack_verbose': args.v2,
            'force_deltagrad_verbose': args.v3,
            'force_device': args.device,
            'force_dry_run': args.dry,
            'force_attack_cycles': args.attack_cycles,
            'force_num_reps': args.num_reps,
            'force_hamming_threshold': args.hamming_threshold,
            'force_gate': args.gate,
            'force_acceptance_func': args.acceptance_func,
            'force_quant_func': args.quant_func,
            'force_lpips_func': args.lpips_func,
            'force_hyperparameters_alpha': args.alpha,
            'force_hyperparameters_beta': args.beta,
            'force_hyperparameters_step_coeff': args.step_coeff,
            'force_hyperparameters_scale_factor': args.scale_factor,
            'force_attack_type': args.attack_type,
            'force_attack_name': args.attack_name,
            'force_hash_function': args.hash_function,
            'force_delta_scaledown': args.delta_scaledown,
            'force_experiment_name': args.experiment_name,
            'force_experiment_description': args.experiment_description,
            'force_experiment_input_dir': args.input_dir,
            'force_experiment_output_dir': args.output_dir,
            'force_resize_width': args.resize_width,
            'force_resize_height': args.resize_height,
            'force_colormode': args.colormode,
            'force_available_devices': args.available_devices
        }
        
        # Filter out None values to avoid overriding with None
        overrides = {k: v for k, v in overrides.items() if v is not None}
        
        exp = engine.load_experiment_from_config(experiment_config, **overrides)

        print(f"Experiment: {engine.experiment.experiment_name}")
        if engine.experiment.experiment_description:
            print(f"Description: {engine.experiment.experiment_description}")
        print(f"Number of attacks: {len(engine.experiment.attacks)}")

        # Run attacks for this experiment
        print(f"\nStarting attacks...")
        start_time = time.time()
        engine.run_attacks()
        end_time = time.time()
        print(f"\nExperiment completed in {end_time - start_time:.2f} seconds")
            
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")



if __name__ == '__main__':
    # Set default dtype
    torch.set_default_dtype(torch.float64)
    sys.path.append(os.path.join(os.path.dirname(__file__), "spectra/"))
    parser = argparse.ArgumentParser(description='Run adversarial attacks from experiment configurations')
    parser.add_argument('-d', '--device', type=str, default=None, help="Override device setting in config (cpu, cuda, mps)")
    parser.add_argument('-f', '--files', nargs='+', required=True, help='Target experiment YAML files (without .yaml extension)')
    parser.add_argument('-experiment_name', type=str, default=None, help='Override experiment name in config')
    parser.add_argument('-experiment_description', type=str, default=None, help='Override experiment description in config')
    parser.add_argument('-input_dir', type=str, default=None, help='Override experiment input directory in config')
    parser.add_argument('-output_dir', type=str, default=None, help='Override experiment output directory in config')
    parser.add_argument('-v1', '--v1', action='store_true', help='Force attack engine verbosity to high')
    parser.add_argument('-v2', '--v2', action='store_true', help='Force attack verbosity to high')
    parser.add_argument('-v3', '--v3', action='store_true', help='Force deltagrad verbosity to high')
    parser.add_argument('-attack_cycles', type=int, default=None, help='Override attack cycles in config')
    parser.add_argument('-num_reps', type=int, default=None, help='Override num_reps in config')
    parser.add_argument('-hamming_threshold', type=int, default=None, help='Override hamming threshold in config')
    parser.add_argument('-gate', type=float, default=None, help='Override gate in config')
    parser.add_argument('-acceptance_func', type=str, default=None, help='Override acceptance function in config')
    parser.add_argument('-quant_func', type=str, default=None, help='Override quantization function in config')
    parser.add_argument('-lpips_func', type=str, default=None, help='Override lpips function in config')
    parser.add_argument('-alpha', type=float, default=None, help='Override alpha in hyperparameters in config')
    parser.add_argument('-beta', type=parse_float_or_tuple, default=None, help='Override beta in hyperparameters in config')
    parser.add_argument('-step_coeff', type=float, default=None, help='Override step_coeff in hyperparameters in config')
    parser.add_argument('-scale_factor', type=parse_float_or_tuple, default=None, help='Override scale_factor in hyperparameters in config')
    parser.add_argument('-attack_type', type=str, default=None, help='Override attack type in config')
    parser.add_argument('-dry', action='store_true', help="dry run, don't save output")
    parser.add_argument('-attack_name', type=str, default=None, help='Override attack name in config')
    parser.add_argument('-hash_function', type=str, default=None, help='Override hash function in config')
    parser.add_argument('-delta_scaledown', action='store_true', help='Override delta scaledown in config')
    parser.add_argument('-resize_width', type=int, default=None, help='Override resize width in config')
    parser.add_argument('-resize_height', type=int, default=None, help='Override resize height in config')
    parser.add_argument('-colormode', type=str, default=None, help='Override colormode in config')
    parser.add_argument('-available_devices', type=str, nargs='+', default=None, help='Override available devices in config')

    args = parser.parse_args()
    
    run_attacks(args)