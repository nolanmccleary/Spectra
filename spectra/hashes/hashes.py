from .hash_wrapper import Hash_Wrapper, PDQ_Wrapper
from .hash_algos import generate_phash, generate_phash_rgb, generate_phash_batched

PHASH = Hash_Wrapper(name="phash", func=generate_phash_batched, colormode="grayscale", resize_height=32, resize_width=32, available_devices={"cpu"}) #Canonical Phash applies DCT on 32x32 downsample

PHASH_RGB = Hash_Wrapper(name="phash_rgb", func=generate_phash_rgb, colormode="rgb", resize_height=32, resize_width=32, available_devices={"cpu"})

PDQ_HASH = PDQ_Wrapper(name="pdq_hash", func=None, colormode="luma", resize_height=512, resize_width=512, available_devices={"cpu"})    #Canonical PDQ applies DCT on 512x512 downsample