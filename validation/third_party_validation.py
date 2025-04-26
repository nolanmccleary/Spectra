from PIL import Image
import imagehash



def phash_compare(img1, img2):
    hash1 = imagehash.phash(Image.open(img1))
    hash2 = imagehash.phash(Image.open(img2))
    print(hash1)
    print(hash2)
    print(str(hash1 - hash2))