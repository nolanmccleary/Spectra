import os
import json
import sys
import time
import torch
from spectra import Attack_Engine, PHASH, PHASH_RGB, AHASH, AHASH_RGB, DHASH, DHASH_RGB, PDQ
from spectra.validation import directory_compare
from models import ALEX_ONNX, ALEX_IMPORT



#TODO: 
# 1) Fix faulty l2/lpips regulator - DONE
# 2) Smart scaleup lpips - DONE
# 3) Fast DCT - DONE
# 4.1) LPIPS constraint - DONE, can gate any acceptance param
# 4.2) Validate percpetion gate - Discrepency due to grayscale vs RGB LPIPS deltas
# 4.3 Better batch guards - DONE
# 4.3 Add PDQ - DONE (Need to see if i can run it on GPU cluster)
# 4) Pareto integrator
# 5) CUDA port - Done
# 6) Cluster integration
# 7) Mass analysis and HP training
# 8) Backtrack algorithm



'''
AHASH_HYPERPARAMETERS = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.0001,
    "scale_factor"  : 0.5
}
DHASH_HYPERPARAMETERS = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.0001,
    "scale_factor"  : 5.5
}
PHASH_HYPERPARAMETERS = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.0001,
    "scale_factor"  : 0.5
}
PDQ_HYPERPARAMETERS = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.0001,
    "scale_factor"  : 0.5
}
AHASH_HYPERPARAMETERS_FINE = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.000001,
    "scale_factor"  : 0.4
}
DHASH_HYPERPARAMETERS_FINE = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.000001,
    "scale_factor"  : 5.5
}
PHASH_HYPERPARAMETERS_FINE = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.000001,
    "scale_factor"  : 0.5
}
PDQ_HYPERPARAMETERS_FINE = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.00001,
    "scale_factor"  : 0.5
}
'''



AHASH_HYPERPARAMETERS_SWEEP = {
    "alpha"         : 2.9,
    "beta"          : (1.0, None, None),
    "step_coeff"    : 0.0001,
    "scale_factor"  : (0.51, 1.0, 0.03)
}

DHASH_HYPERPARAMETERS_SWEEP = {
    "alpha"         : 2.9,
    "beta"          : (1.0, None, None),
    "step_coeff"    : 0.0001,
    "scale_factor"  : (0.01, 1.0, 0.03)
}

PHASH_HYPERPARAMETERS_SWEEP = {
    "alpha"         : 2.9,
    "beta"          : (1.0, None, None),
    "step_coeff"    : 0.0001,
    "scale_factor"  : (0.01, 1.0, 0.03)
}

PDQ_HYPERPARAMETERS_SWEEP = {
    "alpha"         : 2.9,
    "beta"          : (1.0, None, None),
    "step_coeff"    : 0.0001,
    "scale_factor"  : (0.51, 1.0, 0.03)
}




'''
AHASH_HYPERPARAMETERS_SWEEP_FINE = {
    "alpha"         : 2.9,
    "beta"          : (0.8, 1.0, 0.05),
    "step_coeff"    : 0.000001,
    "scale_factor"  : (0.5, 1.0, 0.05)
}

DHASH_HYPERPARAMETERS_SWEEP_FINE = {
    "alpha"         : 2.9,
    "beta"          : (0.8, 1.0, 0.05),
    "step_coeff"    : 0.000001,
    "scale_factor"  : (5.0, 8.0, 0.1)
}

PHASH_HYPERPARAMETERS_SWEEP_FINE = {
    "alpha"         : 2.9,
    "beta"          : (0.8, 1.0, 0.05),
    "step_coeff"    : 0.000001,
    "scale_factor"  : (0.4, 0.7, 0.05)
}

PDQ_HYPERPARAMETERS_SWEEP_FINE = {
    "alpha"         : 2.9,
    "beta"          : (0.8, 1.0, 0.05),
    "step_coeff"    : 0.000001,
    "scale_factor"  : (0.4, 0.7, 0.05)
}
'''





torch.set_default_dtype(torch.float32)
sys.path.append(os.path.join(os.path.dirname(__file__), "spectra/"))


def attack_sequence(dev):
    engine = Attack_Engine(verbose="on")
    image_input_dir = 'sample_images'
    image_output_dir = 'output'

    #LPIPS_MODEL = ALEX_ONNX(device=dev)
    LPIPS_MODEL = ALEX_IMPORT(device=dev)
    F_LPIPS = LPIPS_MODEL.get_lpips

    engine.add_attack("ahash_attack", image_input_dir, image_output_dir, AHASH, AHASH_HYPERPARAMETERS_SWEEP, hamming_threshold=24, colormode="grayscale", acceptance_func="lpips", quant_func=None, lpips_func=F_LPIPS, num_reps=1, attack_cycles=50, device=dev, delta_scaledown=False)
    #engine.add_attack("dhash_attack", image_input_dir, image_output_dir, DHASH, DHASH_HYPERPARAMETERS_SWEEP, hamming_threshold=24, colormode="grayscale", acceptance_func="step", quant_func=None, lpips_func=F_LPIPS, num_reps=10, attack_cycles=50, device=dev, delta_scaledown=True)
    #engine.add_attack("phash_attack", image_input_dir, image_output_dir, PHASH, PHASH_HYPERPARAMETERS_SWEEP, hamming_threshold=28, colormode="grayscale", acceptance_func="step", quant_func=None, lpips_func=F_LPIPS, num_reps=10, attack_cycles=50, device=dev, delta_scaledown=True)
    #engine.add_attack("pdq_attack", image_input_dir, image_output_dir, PDQ, PDQ_HYPERPARAMETERS_SWEEP, hamming_threshold=80, colormode="grayscale", acceptance_func="step", quant_func=None, lpips_func=F_LPIPS, num_reps=10, attack_cycles=50, device=dev, delta_scaledown=True)

    t1 = time.time()
    engine.run_attacks()
    time_delta = time.time() - t1

    print(f"\nTest sequence completed in {time_delta:.2f} seconds")

    post_validation = directory_compare(image_input_dir, image_output_dir, F_LPIPS, dev)
    json_filename = "post_validation.json"

    with open(json_filename, 'w') as f:
        json.dump(post_validation, f, indent=4)


if __name__ == '__main__':
    attack_sequence("cpu")
