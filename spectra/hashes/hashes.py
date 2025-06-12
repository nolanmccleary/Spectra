from .hash_wrapper import Hash_Wrapper, PDQ_Wrapper
from .hash_algos import generate_phash_batched, generate_phash_torch_batched, generate_phash_rgb_batched, generate_ahash_batched, generate_ahash_rgb_batched, generate_dhash_batched, generate_dhash_rgb_batched

AHASH = Hash_Wrapper(name="ahash", func=generate_ahash_batched, colormode="grayscale", resize_height=8, resize_width=8, available_devices={"cpu", "cuda", "mps"}) #Cannonical ahash resizes to 8x8

AHASH_RGB = Hash_Wrapper(name="ahash_rgb", func=generate_ahash_rgb_batched, colormode="rgb", resize_height=8, resize_width=8, available_devices={"cpu", "cuda", "mps"})

DHASH = Hash_Wrapper(name="dhash", func=generate_dhash_batched, colormode="grayscale", resize_height=8, resize_width=8, available_devices={"cpu", "cuda", "mps"}) #Cannonical dhash resizes to 8x8

DHASH_RGB = Hash_Wrapper(name="dhash_rgb", func=generate_dhash_rgb_batched, colormode="rgb", resize_height=8, resize_width=8, available_devices={"cpu", "cuda", "mps"})

PHASH = Hash_Wrapper(name="phash", func=generate_phash_torch_batched, colormode="grayscale", resize_height=32, resize_width=32, available_devices={"cpu", "cuda", "mps"}) #Canonical Phash applies DCT on 32x32 downsample

PHASH_RGB = Hash_Wrapper(name="phash_rgb", func=generate_phash_rgb_batched, colormode="rgb", resize_height=32, resize_width=32, available_devices={"cpu"})

PDQ_HASH = PDQ_Wrapper(name="pdq_hash", func=None, colormode="luma", resize_height=512, resize_width=512, available_devices={"cpu"})  #Canonical PDQ applies DCT on 512x512 downsample