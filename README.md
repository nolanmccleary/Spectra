# Spectra: Stress‐Testing Perceptual Hash Pipelines

Spectra is a framework for evaluating and stress‐testing perceptual image‐hash functions. By creating nearly imperceptible adversarial examples that “break” various hash algorithms (pHash, aHash, dHash, PDQ, etc.), Spectra lets you measure each algorithm’s robustness, find weak spots, and adjust hyperparameters to improve real‐world performance.

---

## Table of Contents

1. [Features](#features)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [Exporting LPIPS to ONNX](#exporting‐lpips‐to‐onnx)  
5. [ONNX‐Based LPIPS Wrapper](#onnx‐based‐lpips‐wrapper)  
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
  - Register and run multiple adversarial attacks (pHash, aHash, dHash, PDQ, etc.) against any perceptual‐hash function.  
  - Support for custom hash wrappers (implement a simple `get_info()` interface).  
  - Flexible acceptance criteria (LPIPS or L₂) to ensure adversarial images remain visually similar.

- **Hyperparameter Injection**  
  - Adjust NES‐optimizer settings (α, β, step coefficient, scale factor) at runtime.  
  - Compare different hyperparameter sets (e.g. `DEFAULT_HYPERPARAMETERS`) or load custom configurations.

- **ONNX‐Based LPIPS for Speed**  
  - Export the PyTorch LPIPS (AlexNet backbone) model to ONNX once, then run inference with `onnxruntime` for faster distance calculations.  
  - Provides an `ALEX_ONNX` wrapper class to swap in ONNX‐based LPIPS in place of the PyTorch version.

- **Detailed Post‐Validation**  
  - After generating adversarial images, compute LPIPS, L₂, and multiple hash distances (pHash, aHash, dHash, PDQ) to confirm both attack success and visual similarity.  
  - Produce a JSON “attack log” that records pre‐validation (Hamming distance, LPIPS, L₂) and post‐validation (hash distances, LPIPS, L₂) for each image.

- **Customizable Input/Output Paths**  
  - Specify any folder of images to attack.  
  - Automatically name output files as `<attack_name>_<original_filename>.<ext>` in your chosen output directory.

---

## Requirements

- **Python 3.9 or higher**  
- **PyTorch 1.12 or higher** (for LPIPS export and internal tensor operations)  
- **Torchvision 0.13 or higher** (used by LPIPS; warnings don’t affect functionality)  
- **lpips 0.1 or higher** (for the LPIPS model)  
- **onnx 1.15 or higher** (must include `load_model_from_string`)  
- **onnxruntime 1.15 or higher** (or `onnxruntime‐gpu` for CUDA; `onnxruntime‐mps` for Apple Metal)  
- **NumPy 1.21 or higher**  
- **Pillow 8.0 or higher**  

Install them all with:

```bash
pip install -r requirements.txt
```

If you need GPU support for ONNX Runtime:

```bash
pip install onnxruntime‐gpu
```
