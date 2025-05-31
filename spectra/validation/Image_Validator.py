from PIL import Image
from spectra.utils import get_rgb_tensor, l2_delta
import imagehash
from spectra.hashes.PDQ import PDQHasher


def ahash_compare(img1, img2):
    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)
    return {"original" : str(hash1), "output" : str(hash2), "hamming" : str(hash1 - hash2)}


def dhash_compare(img1, img2):
    hash1 = imagehash.dhash(img1)
    hash2 = imagehash.dhash(img2)
    return {"original" : str(hash1), "output" : str(hash2), "hamming" : str(hash1 - hash2)}


def phash_compare(img1, img2):
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    return {"original" : str(hash1), "output" : str(hash2), "hamming" : str(hash1 - hash2)}


def PDQ_compare(img1, img2):
    pdq = PDQHasher()
    hash1 = pdq.fromFile(img1)
    hash2 = pdq.fromFile(img2)
    hash1 = hash1.getHash()
    hash2 = hash2.getHash()
    return {"original" : hash1, "output" : hash2, "hamming" : hash1.hammingDistance(hash2)}



def image_compare(img_path_1, img_path_2, lpips_func, device="cpu"):
    
    img_1 = None
    img_2 = None

    with Image.open(img_path_1) as img:
        img_1 = get_rgb_tensor(img, device)
    
    with Image.open(img_path_2) as img:
        img_2 = get_rgb_tensor(img, device)
    
    lpips_score = lpips_func(img_1, img_2)
    l2_score = l2_delta(img_1, img_2)

    img1 = Image.open(img_path_1)
    img2 = Image.open(img_path_2)

    ahash_delta = ahash_compare(img1, img2)["hamming"]
    dhash_delta = dhash_compare(img1, img2)["hamming"]
    phash_delta = phash_compare(img1, img2)["hamming"]
    pdq_delta = PDQ_compare(img_path_1, img_path_2)["hamming"]

    return {
        "lpips" : str(lpips_score),
        "l2" : str(l2_score),
        "ahash_hamming" : str(ahash_delta),
        "dhash_hamming" : str(dhash_delta),
        "phash_hamming" : str(phash_delta),
        "pdq_hamming" : str(pdq_delta)
    }