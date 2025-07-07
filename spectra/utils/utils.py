import numpy as np
import torch
import torch.nn.functional as F
from spectra.utils.acceptance import create_acceptance as _create_acceptance
from spectra.utils.transform import (
    rgb_to_grayscale,
    rgb_to_luma,
    no_conversion,
    generate_conversion,
    generate_inversion,
    inverse_delta,
    inverse_delta_local,
    inverse_luma,
    no_inversion,
)


def tensor_resize(input_tensor, height, width):
    tensor = input_tensor.clone().unsqueeze(0)      #[{3,1}, H, W] -> [1, {3, 1}, H, W]
    tensor_resized = F.interpolate(                 #Interpolate needs to know batch and channel dimensions thus a 4-d tensor is required
        tensor,
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )
    return tensor_resized.squeeze(0)                #[1, {3, 1}, H, W] -> [{3,1}, H, W]


def get_rgb_tensor(image_object, rgb_device):
    if image_object.mode == 'RGBA':
        image_object = image_object.convert('RGB')
    arr = np.array(image_object).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(rgb_device)
    return tensor


def generate_quant(quant_str):
    if quant_str is None:
        return noop
    else:
        quant_table = {"byte_quantize" : byte_quantize}
        if quant_str not in quant_table.keys():
            raise ValueError(f"'{quant_str}' not in set of valid acceptance function handles: {quant_table.keys()}")
    return quant_table[quant_str]


'''
def generate_seed_perturbation(dim, start_scalar, device):
    return torch.rand((1, dim), dtype=torch.float32, device=device) * start_scalar                                        #Mirror to preserve distribution
'''


def to_hex(hash_bool):
    arr = hash_bool.view(-1).cpu().numpy().astype(np.uint8)
    packed = np.packbits(arr)                                   #Convert to byte array
    return '0x' + ''.join(f'{b:02x}' for b in packed.tolist())  #Format to hex


def bool_tensor_delta(a, b):
    return a.ne(b)


def byte_quantize(tensor):
    return torch.round(tensor * 255.0) / (255.0)


def l2_delta(a, b):
    return torch.sqrt(torch.mean((a - b).pow(2))).item()


def generate_acceptance(self, acceptance_str):
    """Compatibility wrapper – delegates to spectra.utils.acceptance.create_acceptance.
    Keeps the original public symbol so existing import paths remain valid.
    """
    return _create_acceptance(self, acceptance_str)


def noop(tensor):
    return tensor


def create_sweep(start, stop, step):
    if stop == None or step == None:
        ret = [start]
    else:     
        ret = [start + step * i for i in range(int((stop + step - start) / step + 1E-6))]
    return ret