import os
import sys
import torch
from spectra import Attack_Engine, PHASH, PHASH_RGB



torch.set_default_dtype(torch.float64)


sys.path.append(os.path.join(os.path.dirname(__file__), "spectra/"))

def attack_sequence():
    engine = Attack_Engine(verbose="on")
    images = [('sample_images/peppers.png', 'output/peppers_attacked.png'), ('sample_images/peppers.jpeg', 'output/peppers_attacked.jpeg'), ('sample_images/imagehash.png', 'output/imagehash_attacked.png')]
    engine.add_attack("phash_attack", images, PHASH, 24, "lpips", 40, "cpu", verbose="off")
    engine.run_attacks()




if __name__ == '__main__':
    attack_sequence()
