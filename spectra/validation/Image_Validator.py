import lpips
from PIL import Image
from spectra.utils import get_rgb_tensor, lpips_rgb, l2_delta
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



def image_compare(image_pair: tuple[str], device="cpu"):
    
    input_image = None
    output_image = None

    with Image.open(image_pair[0]) as img:
        input_image = get_rgb_tensor(img, device)
    
    with Image.open(image_pair[1]) as img:
        output_image = get_rgb_tensor(img, device)
    
    lpips_score = lpips_rgb(input_image, output_image, lpips.LPIPS(net='alex').to(device))
    l2_score = l2_delta(input_image, output_image)

    img_path1, img_path2 = image_pair
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    ahash_delta = ahash_compare(img1, img2)["hamming"]
    dhash_delta = dhash_compare(img1, img2)["hamming"]
    phash_delta = phash_compare(img1, img2)["hamming"]
    pdq_delta = PDQ_compare(img_path1, img_path2)["hamming"]

    return {
        "lpips" : str(lpips_score),
        "l2" : str(l2_score),
        "ahash_hamming" : str(ahash_delta),
        "dhash_hamming" : str(dhash_delta),
        "phash_hamming" : str(phash_delta),
        "pdq_hamming" : str(pdq_delta)
    }