from PIL import Image
import imagehash
from spectra.hashes.PDQ import PDQHasher


def ahash_compare(img1, img2):
    hash1 = imagehash.phash(Image.open(img1))
    hash2 = imagehash.phash(Image.open(img2))
    return {"original" : str(hash1), "output" : str(hash2), "hamming" : str(hash1 - hash2)}


def phash_compare(img1, img2):
    hash1 = imagehash.phash(Image.open(img1))
    hash2 = imagehash.phash(Image.open(img2))
    return {"original" : str(hash1), "output" : str(hash2), "hamming" : str(hash1 - hash2)}


def PDQ_compare(img1, img2):
    pdq = PDQHasher()
    hash1 = pdq.fromFile(img1)
    hash2 = pdq.fromFile(img2)
    hash1 = hash1.getHash()
    hash2 = hash2.getHash()
    return {"original" : hash1, "output" : hash2, "hamming" : hash1.hammingDistance(hash2)}