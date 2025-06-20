import os
import sys
import time
import torch
from spectra import Attack_Engine, PHASH, PHASH_RGB, AHASH, AHASH_RGB, DHASH, DHASH_RGB, PDQ
from models import ALEX_ONNX, ALEX_IMPORT
#import lpips


#F_LPIPS = lpips.LPIPS(net='alex').to(dev)



#TODO: 
# 1) Fix faulty l2/lpips regulator - DONE
# 2) Smart scaleup lpips - DONE
# 3) Fast DCT - DONE
# 4.1) LPIPS constraint - DONE, can gate any acceptance param
# 4.2) Validate percpetion gate - Discrepency due to grayscale vs RGB LPIPS deltas
# 4.3 Better batch guards - DONE
# 4.3 Add PDQ
# 4) Pareto integrator
# 5) CUDA port - Done
# 6) Cluster integration
# 7) Mass analysis and HP training
# 8) Backtrack algorithm




DEFAULT_ALPHA = 2.9
DEFAULT_BETA = 0.9  # Hah, Beta.
DEFAULT_STEP_COEFF = 0.008
DEFAULT_SCALE_FACTOR = 6 #DEFAULTS OPTIMIZED FOR PHASH


PHASH_HYPERPARAMETERS = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.000002,
    "scale_factor"  : 0.5
}

AHASH_HYPERPARAMETERS = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.000007,
    "scale_factor"  : 0.4
}

DHASH_HYPERPARAMETERS = {
    "alpha"         : 2.9,
    "beta"          : 0.85,
    "step_coeff"    : 0.0000013,
    "scale_factor"  : 0.3
}


torch.set_default_dtype(torch.float32)
sys.path.append(os.path.join(os.path.dirname(__file__), "spectra/"))


def attack_sequence(dev):
    engine = Attack_Engine(verbose="on")
    images = ['dizzy1.jpeg']
    image_input_dir = 'sample_images2'
    image_output_dir = 'output'

    #LPIPS_MODEL = ALEX_ONNX(device=dev)
    LPIPS_MODEL = ALEX_IMPORT(device=dev)
    F_LPIPS = LPIPS_MODEL.get_lpips

    engine.add_attack("phash_attack", image_input_dir, image_output_dir, PHASH, PHASH_HYPERPARAMETERS, hamming_threshold=20, colormode="grayscale", acceptance_func="lpips", quant_func="byte_quantize", lpips_func=F_LPIPS, num_reps=1, attack_cycles=10000, device=dev, delta_scaledown=False)
    #engine.add_attack("pdq_attack", image_input_dir, image_output_dir, PDQ, PHASH_HYPERPARAMETERS, hamming_threshold=20, colormode="grayscale", acceptance_func="lpips", quant_func="byte_quantize", lpips_func=F_LPIPS, num_reps=1, attack_cycles=1, device=dev, delta_scaledown=True)
    engine.add_attack("ahash_attack", image_input_dir, image_output_dir, AHASH, AHASH_HYPERPARAMETERS, hamming_threshold=20, colormode="grayscale", acceptance_func="lpips", quant_func="byte_quantize", lpips_func=F_LPIPS, num_reps=1, attack_cycles=10000, device=dev, delta_scaledown=False)
    engine.add_attack("dhash_attack", image_input_dir, image_output_dir, DHASH, DHASH_HYPERPARAMETERS, hamming_threshold=20, colormode="grayscale", acceptance_func="lpips", quant_func="byte_quantize", lpips_func=F_LPIPS, num_reps=1, attack_cycles=10000, device=dev, delta_scaledown=False)

    t1 = time.time()
    engine.run_attacks()
    time_delta = time.time() - t1

    print(f"\nTest sequence completed in {time_delta:.2f} seconds")



if __name__ == '__main__':
    attack_sequence("cpu")
