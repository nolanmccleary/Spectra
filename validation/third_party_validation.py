from PIL import Image
import imagehash
from spectra.hashes.PDQ import Hash256



def phash_compare(img1, img2):
    hash1 = imagehash.phash(Image.open(img1))
    hash2 = imagehash.phash(Image.open(img2))
    return {"original_hash" : str(hash1), "output_hash" : str(hash2), "hamming_distance" : str(hash1 - hash2)}

