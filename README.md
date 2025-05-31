# Spectra: A Stress‐Test System for Perceptual Hash Pipelines

Spectra is a modular framework designed to rigorously evaluate and stress‐test perceptual image‐hash pipelines. By generating perceptually‐similar adversarial examples that intentionally “break” a variety of hash algorithms (e.g., pHash, aHash, dHash, PDQ), Spectra helps researchers and engineers measure robustness, uncover weaknesses, and fine‐tune hyperparameters for real‐world image‐hashing applications.

---

## Table of Contents

1. [Features](#features)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [Exporting LPIPS to ONNX](#exporting‐lpips‐to‐onnx)  
5. [ONNX‐based LPIPS Wrapper](#onnx‐based‐lpips‐wrapper)  
6. [Usage](#usage)  
   - [Exporting LPIPS with `make_lpips_onnx.py`](#exporting‐lpips‐with‐makelpipsonnxpy)  
   - [Running the Attack Engine](#running‐the‐attack‐engine)  
7. [Directory Structure](#directory‐structure)  
8. [Configuration & Hyperparameters](#configuration‐–‐hyperparameters)  
9. [How It Works](#how‐it‐works)  
10. [Extending Spectra](#extending‐spectra)  
11. [License](#license)  

---

## Features

- **Modular Attack Engine**  
  - Define, register, and run multiple adversarial attacks (pHash, aHash, dHash, PDQ, etc.) against any perceptual‐hash function.  
  - Support for custom hash‐wrapper objects (implementing a unified `get_info()` interface).  
  - Pluggable acceptance criteria based on LPIPS or L₂ distance to ensure perceptual fidelity.

- **Hyperparameter Injection**  
  - Easily configure α (number of perturbations), β (momentum), step‐coeff (gradient step size), scale‐factor, and other NES‐optimizer parameters at runtime.  
  - Compare differing hyperparameter suites (e.g., `DEFAULT_HYPERPARAMETERS`) or YAML/JSON‐driven sets.

- **ONNX‐based LPIPS for Efficiency**  
  - Export the PyTorch LPIPS (AlexNet backbone) model to ONNX once, then perform inference via `onnxruntime` for faster, lightweight perceptual distance evaluation.  
  - Include an `ALEX_ONNX` wrapper class to seamlessly replace the native PyTorch LPIPS module.

- **Comprehensive Post‐Validation**  
  - After generating adversarial examples, re‐compute LPIPS, L₂, and multiple hash distances (pHash, aHash, dHash, PDQ) to quantify attack success and perceptual change.  
  - Output a JSON‐formatted “attack log” detailing both pre‐validation (adversarial generation) and post‐validation (hash distances, LPIPS, L₂).

- **Flexible Input/Output Paths**  
  - Dynamically control “input_image_dir” and “output_image_dir” so you can run batch attacks on any folder of images.  
  - Automatic file naming convention: `<attack_name>_<original_filename>.<ext>`

---

## Requirements

- **Python 3.9+** (tested on 3.10 – 3.13)  
- **PyTorch 1.12+** (for LPIPS export and internal tensor ops)  
- **Torchvision 0.13+** (for model weight loading warnings—optional)  
- **lpips 0.1+** (for LPIPS PyTorch model)  
- **onnx 1.15+** (must support `load_model_from_string`)  
- **onnxruntime 1.15+** (or `onnxruntime‐gpu` for CUDA support; `onnxruntime‐mps` for Apple Silicon MPS)  
- **NumPy 1.21+**  
- **Pillow 8.0+**  

**Install with pip**:

```bash
pip install -r requirements.txt
```

If you want GPU acceleration for ONNX Runtime:
```bash
pip install onnxruntime‐gpu
```