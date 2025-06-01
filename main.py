import os
import sys
import time
import torch
from spectra import Attack_Engine, PHASH, PHASH_RGB, AHASH, AHASH_RGB, DHASH, DHASH_RGB
from models import ALEX_ONNX, ALEX_IMPORT
#import lpips


#F_LPIPS = lpips.LPIPS(net='alex').to(dev)

DEFAULT_ALPHA = 2.9
DEFAULT_BETA = 0.9  # Hah, Beta.
DEFAULT_STEP_COEFF = 0.008
DEFAULT_SCALE_FACTOR = 6 #DEFAULTS OPTIMIZED FOR PHASH


DEFAULT_HYPERPARAMETERS = {
    "alpha"         : DEFAULT_ALPHA,
    "beta"          : DEFAULT_BETA,
    "step_coeff"    : DEFAULT_STEP_COEFF,
    "scale_factor"  : DEFAULT_SCALE_FACTOR
}


torch.set_default_dtype(torch.float32)
sys.path.append(os.path.join(os.path.dirname(__file__), "spectra/"))

def attack_sequence(dev):
    engine = Attack_Engine(verbose="on")
    images = ['peppers.png', 'peppers.jpeg', 'imagehash.png']
    image_input_dir = 'sample_images'
    image_output_dir = 'output'

    #LPIPS_MODEL = ALEX_ONNX(device=dev)
    LPIPS_MODEL = ALEX_IMPORT(device=dev)
    F_LPIPS = LPIPS_MODEL.get_lpips

    engine.add_attack("phash_attack", images, image_input_dir, image_output_dir, PHASH, DEFAULT_HYPERPARAMETERS, 24, "lpips", 100, dev, lpips_func = F_LPIPS)
    engine.add_attack("ahash_attack", images, image_input_dir, image_output_dir, AHASH, DEFAULT_HYPERPARAMETERS, 24, "l2", 100, dev, lpips_func = F_LPIPS)
    engine.add_attack("dhash_attack", images, image_input_dir, image_output_dir, DHASH, DEFAULT_HYPERPARAMETERS, 24, "l2", 100, dev, lpips_func = F_LPIPS)

    t1 = time.time()
    engine.run_attacks()
    time_delta = time.time() - t1

    print(f"\nAttack sequence completed in {time_delta:.2f} seconds")



if __name__ == '__main__':
    attack_sequence("cpu")
