from spectra import Attack_Engine, PHASH

def phash_attack():
    engine = Attack_Engine()
    images = [('sample_images/peppers.png', 'output/peppers_attacked.png'), ('sample_images/imagehash.png', 'output/imagehash_attacked.png')]

    engine.add_attack(images, PHASH, 3, 100, "cpu", verbose="on")

    engine.run_attacks()



if __name__ == '__main__':
    phash_attack()