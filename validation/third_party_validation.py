from PIL import Image
import imagehash
from spectra.hashes.PDQ import PDQHasher
from spectra.hashes.PDQ import Hash256


def phash_compare(img1, img2):
    hash1 = imagehash.phash(Image.open(img1))
    hash2 = imagehash.phash(Image.open(img2))
    return {"PHASH ----- original_hash" : str(hash1), "output_hash" : str(hash2), "hamming_distance" : str(hash1 - hash2)}


def PDQ_compare(img1, img2):
    pdq = PDQHasher()
    hash1 = pdq.fromFile(img1)
    hash2 = pdq.fromFile(img2)

    hash1 = hash1.getHash()
    hash2 = hash2.getHash()

    return {"PDQ ----- original_hash" : hash1, "output_hash" : hash2, "hamming" : hash1.hammingDistance(hash2)}