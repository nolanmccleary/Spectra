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
# 4.2) Validate percpetion gate
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


DEFAULT_HYPERPARAMETERS = {
    "alpha"         : 2.9,
    "beta"          : 0.9,
    "step_coeff"    : 0.008,
    "scale_factor"  : 6
}

AHASH_HYPERPARAMETERS = {
    "alpha"         : 2.5,
    "beta"          : 0.9,
    "step_coeff"    : 0.001,
    "scale_factor"  : 6
}



torch.set_default_dtype(torch.float32)
sys.path.append(os.path.join(os.path.dirname(__file__), "spectra/"))

def attack_sequence(dev):
    engine = Attack_Engine(verbose="on")
    images = ['peppers.png', 'imagehash.png']
    image_input_dir = 'sample_images'
    image_output_dir = 'output'

    #LPIPS_MODEL = ALEX_ONNX(device=dev)
    LPIPS_MODEL = ALEX_IMPORT(device=dev)
    F_LPIPS = LPIPS_MODEL.get_lpips

    #engine.add_attack("phash_attack", images, image_input_dir, image_output_dir, PHASH, DEFAULT_HYPERPARAMETERS, 30, "lpips", 5, 1000, dev, lpips_func = F_LPIPS, delta_scaledown=False, gate=0.04)
    #engine.add_attack("pdq_attack", images, image_input_dir, image_output_dir, PDQ, DEFAULT_HYPERPARAMETERS, 1, "lpips", 1, 1, dev, lpips_func = F_LPIPS, delta_scaledown=False, gate=0.04)

    #engine.add_attack("phash_attack_scaledown", images, image_input_dir, image_output_dir, PHASH, DEFAULT_HYPERPARAMETERS, 20, "lpips", 10, 1000, dev, lpips_func = F_LPIPS, delta_scaledown=True)
    engine.add_attack("ahash_attack", images, image_input_dir, image_output_dir, AHASH, AHASH_HYPERPARAMETERS, 10, "lpips", 10, 1000, dev, lpips_func = F_LPIPS, delta_scaledown=False, gate=0.05)
    #engine.add_attack("dhash_attack", images, image_input_dir, image_output_dir, DHASH, DEFAULT_HYPERPARAMETERS, 20, "lpips", 10, 1000, dev, lpips_func = F_LPIPS, delta_scaledown=False, gate=0.05)

    t1 = time.time()
    engine.run_attacks()
    time_delta = time.time() - t1

    print(f"\nTest sequence completed in {time_delta:.2f} seconds")



if __name__ == '__main__':
    attack_sequence("cpu")
