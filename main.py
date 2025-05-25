import lpips
import os
import sys
import torch
from spectra import Attack_Engine, PHASH, PHASH_RGB, AHASH, AHASH_RGB, DHASH, DHASH_RGB



torch.set_default_dtype(torch.float32)


sys.path.append(os.path.join(os.path.dirname(__file__), "spectra/"))

def attack_sequence(dev):
    engine = Attack_Engine(verbose="on")
    images = [('sample_images/peppers.png', 'output/peppers_attacked.png'), ('sample_images/peppers.jpeg', 'output/peppers_attacked.jpeg'), ('sample_images/imagehash.png', 'output/imagehash_attacked.png')]
    
    F_LPIPS = lpips.LPIPS(net='alex').to(dev)

    engine.add_attack("phash_attack", images, PHASH, 24, "lpips", 100, "cpu", verbose="off", lpips_func = F_LPIPS)
    engine.add_attack("ahash_attack", images, AHASH, 24, "l2", 100, "cpu", verbose="off", lpips_func = F_LPIPS)
    engine.add_attack("dhash_attack", images, DHASH, 24, "l2", 100, dev, verbose="off", lpips_func = F_LPIPS)

    engine.run_attacks()




if __name__ == '__main__':
    attack_sequence("cpu")
