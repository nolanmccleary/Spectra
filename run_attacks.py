import argparse
import time
import torch
import sys
import os
import traceback
from spectra import Attack_Engine, PHASH, AHASH, DHASH, PDQ
from spectra.config import ConfigManager, HashFunction


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
        
        try:
            # Load experiment configuration
            experiment_config = config_manager.load_experiment_config(experiment_file)
            
            print(f"Experiment: {experiment_config.name}")
            if experiment_config.description:
                print(f"Description: {experiment_config.description}")
            print(f"Device: {experiment_config.device}")
            print(f"Number of attacks: {len(experiment_config.attacks)}")
            
            # Add attacks from configuration with verbosity and device overrides
            engine.load_experiment_from_config(experiment_config, 
                                             force_engine_verbose=args.v1,
                                             force_attack_verbose=args.v2,
                                             force_deltagrad_verbose=args.v3,
                                             force_device=args.device,
                                             force_dry_run=args.dry)
            
            # Run attacks for this experiment
            print(f"\nStarting attacks...")
            start_time = time.time()
            engine.run_attacks()
            end_time = time.time()
            print(f"\nExperiment completed in {end_time - start_time:.2f} seconds")
            
        except FileNotFoundError:
            print(f"Error: Configuration file '{experiment_file}' not found")
            continue

        except Exception as e:
            print(f"Error running experiment '{experiment_file}': {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            continue
    
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
    parser.add_argument('-v1', '--v1', action='store_true', help='Force attack engine verbosity to high')
    parser.add_argument('-v2', '--v2', action='store_true', help='Force attack verbosity to high')
    parser.add_argument('-v3', '--v3', action='store_true', help='Force deltagrad verbosity to high')
    parser.add_argument('--dry', action='store_true', help="dry run, don't save output")
    args = parser.parse_args()
    
    run_attacks(args)