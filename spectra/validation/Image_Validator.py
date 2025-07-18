import imagehash
from pathlib import Path
from PIL import Image
import struct
import zlib
from spectra.utils import get_rgb_tensor, l2_delta
from spectra.hashes.PDQ import PDQHasher
from spectra.hashes.hash_algos import _generate_ahash, _generate_dhash, _generate_phash, _generate_pdq
from spectra.utils.transform import tensor_resize, rgb_to_grayscale


########################################## HASH COMPARISON ######################################################

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


def ahash_compare_torch(img1, img2):
    img1 = tensor_resize(img1, 8, 8)
    img2 = tensor_resize(img2, 8, 8)
    gray1 = rgb_to_grayscale(img1)
    gray2 = rgb_to_grayscale(img2)
    hash1 = _generate_ahash(gray1)
    hash2 = _generate_ahash(gray2)
    return {"original" : hash1, "output" : hash2, "hamming" : hash1.ne(hash2).sum().item()}


def dhash_compare_torch(img1, img2):
    img1 = tensor_resize(img1, 8, 8)
    img2 = tensor_resize(img2, 8, 8)
    gray1 = rgb_to_grayscale(img1)
    gray2 = rgb_to_grayscale(img2)
    hash1 = _generate_dhash(gray1)
    hash2 = _generate_dhash(gray2)
    return {"original" : hash1, "output" : hash2, "hamming" : hash1.ne(hash2).sum().item()}


def phash_compare_torch(img1, img2):
    img1 = tensor_resize(img1, 8, 8)
    img2 = tensor_resize(img2, 8, 8)
    gray1 = rgb_to_grayscale(img1)
    gray2 = rgb_to_grayscale(img2)
    hash1 = _generate_phash(gray1)
    hash2 = _generate_phash(gray2)
    return {"original" : hash1, "output" : hash2, "hamming" : hash1.ne(hash2).sum().item()}


def pdq_compare_torch(img1, img2):
    img1 = tensor_resize(img1, 256, 256)
    img2 = tensor_resize(img2, 256, 256)
    gray1 = rgb_to_grayscale(img1)
    gray2 = rgb_to_grayscale(img2)
    hash1 = _generate_pdq(gray1, 16)
    hash2 = _generate_pdq(gray2, 16)
    return {"original" : hash1, "output" : hash2, "hamming" : hash1.ne(hash2).sum().item()}




################################## FILE METADATA COMPARISON ######################################################


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def detect_format(path):
    with open(path, 'rb') as f:
        sig = f.read(8)
        if sig == b'\x89PNG\r\n\x1a\n':
            return 'png'
        elif sig.startswith(b'\xFF\xD8'):
            return 'jpeg'
        else:
            return 'unknown'


def read_png_chunk(fp):
    length_bytes = fp.read(4)
    if not length_bytes:
        return None, None, None
    length = struct.unpack(">I", length_bytes)[0]
    chunk_type = fp.read(4)
    data = fp.read(length)
    crc = fp.read(4)  # CRC is ignored for now
    return chunk_type, data, length


def parse_png_metadata(png_path):
    metadata = {}
    with open(png_path, "rb") as f:
        assert f.read(8) == PNG_MAGIC, "Not a valid PNG file"

        while True:
            chunk_type, data, length = read_png_chunk(f)
            if chunk_type is None:
                break

            if chunk_type == b'IHDR':
                width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", data)
                metadata['IHDR'] = {
                    'width': width,
                    'height': height,
                    'bit_depth': bit_depth,
                    'color_type': color_type,
                    'interlace': interlace
                }

            elif chunk_type == b'gAMA':
                gamma = struct.unpack(">I", data)[0] / 100000.0
                metadata['gAMA'] = gamma

            elif chunk_type == b'sRGB':
                rendering_intent = struct.unpack("B", data)[0]
                metadata['sRGB'] = rendering_intent

            elif chunk_type == b'iCCP':
                name_end = data.index(b'\x00')
                compression_method = data[name_end + 1]
                compressed = data[name_end + 2:]
                profile = zlib.decompress(compressed)
                metadata['iCCP'] = {
                    'name': data[:name_end].decode('latin1'),
                    'profile': profile
                }

            elif chunk_type == b'PLTE':
                metadata['PLTE'] = data  # raw palette data

            elif chunk_type == b'tRNS':
                metadata['tRNS'] = data  # transparency values

            elif chunk_type == b'cHRM':
                values = struct.unpack(">8I", data)
                metadata['cHRM'] = {
                    'white_point': (values[0], values[1]),
                    'red': (values[2], values[3]),
                    'green': (values[4], values[5]),
                    'blue': (values[6], values[7]),
                }

            elif chunk_type == b'bKGD':
                metadata['bKGD'] = data  # format depends on color type

            elif chunk_type == b'pHYs':
                x_ppu, y_ppu, unit = struct.unpack(">IIB", data)
                metadata['pHYs'] = {
                    'x_ppu': x_ppu,
                    'y_ppu': y_ppu,
                    'unit': unit
                }

            elif chunk_type == b'IEND':
                break  # done parsing

    return metadata


def parse_jpeg_metadata(jpeg_path):
    metadata = {}
    with open(jpeg_path, "rb") as f:
        assert f.read(2) == b'\xFF\xD8', "Not a valid JPEG file"

        while True:
            # Skip any non-marker padding bytes
            byte = f.read(1)
            if not byte:
                break
            if byte != b'\xFF':
                continue

            # Skip repeated 0xFFs (padding)
            while True:
                marker = f.read(1)
                if not marker:
                    return metadata
                if marker != b'\xFF':
                    break

            # Standalone markers without length/data
            if marker in [b'\xD9', b'\xDA']:  # EOI or SOS
                break

            # Read segment length
            length_bytes = f.read(2)
            if len(length_bytes) != 2:
                break  # Malformed
            length = struct.unpack(">H", length_bytes)[0]
            data = f.read(length - 2)
            if len(data) != length - 2:
                break  # Malformed

            m = marker[0]

            if m == 0xE0:  # APP0 (JFIF)
                metadata['APP0_JFIF'] = {
                    "dpi_x": struct.unpack(">H", data[7:9])[0],
                    "dpi_y": struct.unpack(">H", data[9:11])[0],
                    "units": data[6]
                }

            elif m == 0xE1:  # APP1 (EXIF)
                metadata['APP1_Exif'] = data[:16]  # Partial EXIF fingerprint

            elif 0xE2 <= m <= 0xEF:  # Other APP segments
                tag = f'APP{m - 0xE0}'
                metadata[tag] = data[:16]  # You can extend this as needed

            elif m in (0xC0, 0xC2):  # SOF0 (baseline) / SOF2 (progressive)
                precision = data[0]
                height = struct.unpack(">H", data[1:3])[0]
                width = struct.unpack(">H", data[3:5])[0]
                num_components = data[5]
                metadata[f'SOF{m - 0xC0}'] = {
                    "precision": precision,
                    "width": width,
                    "height": height,
                    "components": num_components
                }

            elif m == 0xDB:  # DQT (quantization tables)
                metadata.setdefault('DQT', []).append(data)

            elif m == 0xFE:  # COM (comment)
                metadata.setdefault('COM', []).append(data[:32])

            elif m == 0xC4:  # DHT (Huffman table)
                metadata.setdefault('DHT', []).append(data[:32])

    return metadata



##########################################################################################################


def image_compare(img_path_1, img_path_2, lpips_func, device, verbose):
    
    img_1 = None
    img_2 = None

    with Image.open(img_path_1) as img:
        img_1 = get_rgb_tensor(img, device)
    
    with Image.open(img_path_2) as img:
        img_2 = get_rgb_tensor(img, device)
    
    lpips_score = lpips_func(img_1, img_2)
    l2_score = l2_delta(img_1, img_2)

    '''
    md1 = {}
    md2 = {}

    sig1 = detect_format(img_path_1)
    sig2 = detect_format(img_path_2)

    error_log = ""

    if sig1 == sig2:
        if sig1 == "png":
            md1 = parse_png_metadata(img_path_1)
            md2 = parse_png_metadata(img_path_2)

        elif sig1 == "jpeg":
            md1 = parse_jpeg_metadata(img_path_1)
            md2 = parse_jpeg_metadata(img_path_2)

        else:
            error_log += f"\nVALIDATION ERROR:: Metadata validation for {sig1} not supported!"

    else:
        error_log += f"\nWARNING:: Image formats do not match - First Image: {sig1}, Second Image: {sig2}"


    if md1 != md2:
        for key in md1.keys():
            if key not in md2.keys():
                error_log += f"\nKEY NOT FOUND:: {key} not found in output image metadata"

            elif md1[key] != md2[key]:
                error_log += f"\nMISMATCH:: First Image: {md1[key]}, Second Image: {md2[key]}"
    '''
    img1 = Image.open(img_path_1)
    img2 = Image.open(img_path_2)

    ahash_delta = ahash_compare(img1, img2)["hamming"]
    dhash_delta = dhash_compare(img1, img2)["hamming"]
    phash_delta = phash_compare(img1, img2)["hamming"]
    pdq_delta = PDQ_compare(img_path_1, img_path_2)["hamming"]

    ahash_delta_torch = ahash_compare_torch(img_1, img_2)["hamming"]
    dhash_delta_torch = dhash_compare_torch(img_1, img_2)["hamming"]
    phash_delta_torch = phash_compare_torch(img_1, img_2)["hamming"]
    pdq_delta_torch = pdq_compare_torch(img_1, img_2)["hamming"]

    '''
    if verbose == "on":
        output_msg = f"{img_path_1} - {img_path_2}" + "{" + error_log + "}"
        #print(output_msg)
    '''

    return {
        "lpips" : str(lpips_score),
        "l2" : str(l2_score),
        "ahash_hamming_imagehash" : str(ahash_delta),
        "dhash_hamming_imagehash" : str(dhash_delta),
        "pdq_hamming_imagehash"   : str(pdq_delta),
        "phash_hamming_imagehash" : str(phash_delta),
        "ahash_hamming_torch" : str(ahash_delta_torch),
        "dhash_hamming_torch" : str(dhash_delta_torch),
        "pdq_hamming_torch"   : str(pdq_delta_torch),
        "phash_hamming_torch" : str(phash_delta_torch),
        "ahash_discrepency" : str(int(ahash_delta_torch) - int(ahash_delta)),
        "dhash_discrepency" : str(int(dhash_delta_torch) - int(dhash_delta)),
        "phash_discrepency" : str(int(phash_delta_torch) - int(phash_delta)),
        "pdq_discrepency"   : str(int(pdq_delta_torch) - int(pdq_delta))
    }   #Add error log if desired



def directory_compare(input_dir, output_dir, lpips_func, device, verbose="off"):
    """
    Compare every output image in output_dir against its corresponding
    input in input_dir. Outputs are assumed named <prefix>_<input_filename>.
    Returns a dict of the form:
      {
        "<prefix>": {
          "<input_filename>": { …results of image_compare… },
          …,
          "average": {
            "lpips": …,
            "l2": …,
            "ahash_hamming": …,
            "dhash_hamming": …,
            "phash_hamming": …,
            "pdq_hamming": …
          }
        },
        …
      }
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    input_files = {f.name: f for f in input_dir.iterdir() if f.is_file()}

    results: dict[str, dict[str, dict]] = {}

    for out_path in output_dir.iterdir():
        if not out_path.is_file():
            continue
        out_name = out_path.name

        match = None
        for in_name in input_files:
            suffix = f"_{in_name}"
            if out_name.endswith(suffix):
                prefix = out_name[: -len(suffix)]
                match  = in_name
                break

        if match is None:
            if verbose == "on":
                print(f"Skipping unmatched output: {out_name}")
            continue

        in_path = input_files[match]
        cmp_res = image_compare(
            str(in_path),
            str(out_path),
            lpips_func,
            device,
            verbose
        )

        results.setdefault(prefix, {})[match] = cmp_res

    for prefix, entries in results.items():
        sums = {}
        count = 0

        for img_name, metrics in entries.items():
            if img_name == "average":
                continue
            count += 1
            for key, val in metrics.items():
                if key == "error_log":
                    continue
                num = float(val)
                sums[key] = sums.get(key, 0.0) + num

        avg_metrics = { key: (sums[key] / count) for key in sums }
        entries["average"] = avg_metrics

    return results


