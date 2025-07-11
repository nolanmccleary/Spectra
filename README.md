# Spectra: Adversarially Testing Perceptual Hash Functions

Spectra is a comprehensive toolkit designed to help evaluate the robusteness of perceptual hash algorithms against gradient-based adversarial attacks.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nolanmccleary/spectra.git
cd spectra

# Install dependencies
pip install -r requirements.txt
```

### Basic Use

Run attacks using the configuration system:

```bash
# Run a single experiment
python run_attacks.py -f full_attack_suite

# Run multiple experiments
python run_attacks.py -f example_experiment ahash_attack_experiment

# Run with custom device (if available)
python run_attacks.py -f full_attack_suite -d cuda
```

## Configuration System

The system uses a hierarchical configuration approach with YAML files:

### Experiment Configuration

Experiments are defined in `experiments/` directory. Each experiment contains one or more attack configurations:

```yaml
name: "Full Attack Suite Experiment"
description: "Comprehensive adversarial attacks on all hash functions"
device: "cpu"
verbose: true
attacks:
  - attack_name: "ahash_attack"
    hash_function: "ahash"
    hamming_threshold: 24
    colormode: "grayscale"
    device: "cpu"
    num_reps: 1
    attack_cycles: 10000
    delta_scaledown: true
    acceptance_func: "lpips"
    input_dir: "sample_images"
    output_dir: "output"
    hyperparameters:
      alpha: 2.9
      beta: [0.9, null, null]
      step_coeff: 0.0001
      scale_factor: [0.4, null, null]
```

### Attack Configuration Fields

| Field | Description | Required |
|-------|-------------|----------|
| `attack_name` | Unique identifier for the attack | Yes |
| `hash_function` | Target hash function (`ahash`, `dhash`, `phash`, `pdq`) | Yes |
| `hamming_threshold` | Minimum hamming distance required | Yes |
| `colormode` | Color processing mode (`grayscale`, `rgb`, `luma`) | Yes |
| `device` | Computation device (`cpu`, `cuda`, `mps`) | Yes |
| `num_reps` | Number of attack repetitions | Yes |
| `attack_cycles` | Maximum attack iterations | Yes |
| `delta_scaledown` | Enable delta scaling | Yes |
| `acceptance_func` | Acceptance function (`lpips`, `l2`) | Yes |
| `input_dir` | Directory containing input images | Yes |
| `output_dir` | Directory for output images | Yes |
| `hyperparameters` | Attack optimization parameters | Yes |

### Hyperparameters

Each attack configuration includes hyperparameters that control the optimization:

```yaml
hyperparameters:
  alpha: 2.9          # Momentum coefficient
  beta: [0.9, null, null]  # Current step weight (or sweep range)
  step_coeff: 0.0001  # Gradient step size
  scale_factor: [0.4, null, null]  # Perturbation scale (or sweep range)
```

## Supported Hash Functions

The system supports four major perceptual hash algorithms:

1. **AHASH** - Average Hash
2. **DHASH** - Difference Hash  
3. **PHASH** - Perceptual Hash
4. **PDQ** - DCT-Based Perceptual Hash, Similar to PHash

## Input/Output

### Input Images
- Supported formats: JPEG, JPG, PNG
- Images are automatically resized and converted to grayscale/RGB as needed by the chosen hash algorithm
- Place images in the directory specified by `input_dir`

### Output
- **Attacked images**: Saved to `output_dir` with naming pattern `{attack_name}_{original_filename}`
- **Results log**: JSON file with detailed attack metrics
- **Validation data**: Post-attack hash comparisons across all algorithms

### Output Metrics
- Hamming distance between original and attacked hashes
- LPIPS perceptual similarity score
- L2 distance (Euclidean)
- Number of optimization steps required
- Success/failure status

## Advanced Usage

### Creating Custom Configurations

```python
from spectra.config import AttackConfig, ExperimentConfig, HyperparameterConfig

# Create attack configuration
attack_config = AttackConfig(
    attack_name="custom_attack",
    hash_function="phash",
    hamming_threshold=32,
    colormode="grayscale",
    device="cpu",
    num_reps=3,
    attack_cycles=15000,
    delta_scaledown=True,
    acceptance_func="lpips",
    input_dir="my_images",
    output_dir="results",
    hyperparameters=HyperparameterConfig(
        alpha=2.9,
        beta=0.85,
        step_coeff=0.00005,
        scale_factor=0.3
    )
)

# Save configuration
from spectra.config import ConfigManager
config_manager = ConfigManager()
config_manager.save_attack_config(attack_config, "my_custom_attack")
```

### Hyperparameter Tuning

The system supports hyperparameter sweeps:

```yaml
hyperparameters:
  alpha: 2.9
  beta: [0.8, 0.95, 0.05]  # Sweep from 0.8 to 0.95 in steps of 0.05
  step_coeff: 0.0001
  scale_factor: [0.3, 0.6, 0.1]  # Sweep from 0.3 to 0.6 in steps of 0.1
```

### Batch Processing

Process multiple experiments in sequence:

```bash
# Create a batch script
python run_attacks.py -f experiment1 experiment2 experiment3
```

## Technical Details

### Natural Evolution Strategies (NES)

The core attack algorithm uses NES to estimate gradients for non-differentiable hash functions:

1. **Perturbation Generation**: Creates normally distributed perturbation vectors
2. **Gradient Estimation**: Computes gradient approximation from hash differences
3. **Gradient Step**: Updates image in gradient direction with momentum (momentum is optional)

### Algorithm Parameters

- **`alpha`**: Proportionality constant relating perturbation vector count with image size (2.9 is recommended)
- **`beta`**: Weight for current step vs. momentum
- **`step_coeff`**: Scale factor for gradient steps
- **`scale_factor`**: Perturbation (N(0, 1)) scale constant for gradient estimation

### Performance Considerations

- **Device Selection**: Use CUDA for GPU acceleration when available. MPS will run but torch's support is limited. 
- **Image Resolution**: Higher resolution images require more compute
- **Hash Complexity**: Different hashes will have different compute requirements. PDQ's compute requirement is comparatively very high. 

## Research Applications

This toolkit is designed for:

1. **Hash Robustness Evaluation**: Compare vulnerability of different hash algorithms
2. **Defense Development**: Test countermeasures against gradient-based attacks
3. **Algorithm Comparison**: Evaluate hash function complementarity
4. **Hash Algorithm Development**: Development of adversarially robust hashing algorithms

## File Structure

```
gradient_injection/
├── spectra/                 # Core attack engine
│   ├── config/             # Configuration system
│   ├── utils/              # Utility functions
    ├── deltagrad/          # Gradient compute engine
│   └── attack_engine.py    # Main attack orchestration
├── experiments/            # Configuration files
├── sample_images/          # Test images
├── output/                 # Attack results
├── validation/             # Validation scripts and results
├── run_attacks.py          # Main execution script
└── requirements.txt        # Dependencies
```

## Dependencies

- PyTorch (>= 1.9.0)
- NumPy
- Pillow
- PyYAML
- Pydantic
- LPIPS (for perceptual similarity)

## License

This project is released under a Restricted Research License. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{spectra2024,
  title={Spectra: Adversarially Testing Perceptual Hash Functions},
  author={Nolan McCleary},
  year={2025},
  url={https://github.com/nolanmccleary/spectra}
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Disclaimer

This tool is intended for research purposes to evaluate the robustness of perceptual hash algorithms. Users are responsible for ensuring compliance with applicable laws and ethical guidelines when using this software.