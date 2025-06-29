import numpy as np
import torch
import torch.nn.functional as F


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


def rgb_to_grayscale(rgb_tensor):
        return torch.mean(rgb_tensor, dim = 0, keepdim=True)
def rgb_to_luma(rgb_tensor):        #[C, H, W] -> [1, H, W]
    r, g, b = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return gray.unsqueeze(0)
def no_conversion(tensor):
    return tensor 
def generate_conversion(conversion_str: str):
    conversion_table = {"grayscale" : rgb_to_grayscale, "grayscale_local" : rgb_to_grayscale, "luma" : rgb_to_luma, "noinvert" : no_conversion}
    if conversion_str not in conversion_table.keys():
        raise ValueError(f"'{conversion_str}' not in set of valid acceptance function handles: {conversion_table.keys()}")

    return conversion_table[conversion_str]


def generate_inversion(inversion_str: str):

    def inverse_delta(tensor, delta, eps=1e-6):
        C, H, W = tensor.shape
        if delta.shape == (C, H, W):
            return delta

        rgb_mean = tensor.mean()        # [1, H, W]
        gd = delta.unsqueeze(0)         # [1, H, W]
        
        # Avoid division by zero
        delta = torch.where(
            gd <= 0,
            gd * tensor / (rgb_mean + eps),
            gd * (1 - tensor) / ((1 - rgb_mean) + eps)
        )
        return delta.view(C, H, W)

    def inverse_delta_local(tensor, delta, eps=1e-6):
        C, H, W = tensor.shape
        if delta.shape == (C, H, W):
            return delta

        rgb_mean = tensor.mean(dim=0, keepdim=True)     # [1, H, W]
        gd = delta.unsqueeze(0)                         # [1, H, W]
        
        # Avoid division by zero
        delta = torch.where(
            gd <= 0,
            gd * tensor / (rgb_mean + eps),
            gd * (1 - tensor) / ((1 - rgb_mean) + eps)
        )
        return delta.view(C, H, W)

    def inverse_luma(tensor, delta):
        if delta.dim() == 2:               # delta is (H, W)
            delta = delta.unsqueeze(0)     # -> (1, H, W)

        r, g, b = tensor[0], tensor[1], tensor[2]

        luma = (0.2126*r + 0.7152*g + 0.0722*b).unsqueeze(0)   # (1, H, W)
        new_luma = torch.clamp(luma + delta, 0.0, 1.0)         # broadcast OK
        ratio    = (new_luma+1e-6) / (luma+1e-6)               # (1,H,W)
        perturbed = tensor * ratio                             # (3,H,W)
        delta_rgb = perturbed - tensor                         # (3,H,W)
        return delta_rgb

    def no_inversion(tensor, delta):
        return delta

    inversion_table = {"grayscale" : inverse_delta, "grayscale_local" : inverse_delta_local, "luma" : inverse_luma, "noinvert" : no_inversion} #TODO: Add inverse luma
    if inversion_str not in inversion_table.keys():
        raise ValueError(f"'{inversion_str}' not in set of valid acceptance function handles: {inversion_table.keys()}")

    return inversion_table[inversion_str]


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

    def lpips_acceptance_func(tensor):
        self.current_hash = self.func(tensor.to(self.func_device))
        self.current_hamming = int((self.original_hash != self.current_hash).sum().item())
        self.current_lpips = self.lpips_func(self._tensor, tensor)
        self.current_l2 = l2_delta(self._tensor, tensor)

        break_loop, accepted = False, False

        if self.gate is not None:
            if self.current_lpips >= self.gate:
                break_loop = True
        
        if self.current_hamming >= self.hamming_threshold:
            if self.current_lpips < self.output_lpips:
                self.output_lpips = self.current_lpips
                self.output_l2 = self.current_l2
                self.output_hash = self.current_hash
                self.output_hamming = self.current_hamming
                accepted = True

            break_loop = True

        return break_loop, accepted


    def l2_acceptance_func(tensor):
        self.current_hash = self.func(tensor.to(self.func_device))
        self.current_hamming = int((self.original_hash != self.current_hash).sum().item())
        self.current_lpips = self.lpips_func(self._tensor, tensor)
        self.current_l2 = l2_delta(self._tensor, tensor)

        break_loop, accepted = False, False

        if self.gate is not None:
            if self.current_l2 >= self.gate:
                break_loop = True

        if self.current_hamming >= self.hamming_threshold:
            if self.current_l2 < self.output_l2:
                self.output_l2 = self.current_l2
                self.output_lpips = self.current_lpips
                self.output_hash = self.current_hash
                self.output_hamming = self.current_hamming
                accepted = True

            break_loop = True

        return break_loop, accepted

    def latching_acceptance_func(tensor):
        self.current_hash = self.func(tensor.to(self.func_device))
        self.current_hamming = int((self.original_hash != self.current_hash).sum().item())
        self.current_lpips = self.lpips_func(self._tensor, tensor)
        self.current_l2 = l2_delta(self._tensor, tensor)

        if self.current_hamming >= self.hamming_threshold:
            return True, True
        
        return False, False
        

    acceptance_table = {"lpips" : lpips_acceptance_func, "l2" : l2_acceptance_func, "latch" : latching_acceptance_func}
    if acceptance_str not in acceptance_table.keys():
        raise ValueError(f"'{acceptance_str}' not in set of valid acceptance function handles: {acceptance_table.keys()}")

    return acceptance_table[acceptance_str]


def noop(tensor):
    return tensor


def create_sweep(start, stop, step):
    return [start + step * i for i in range(int((stop + step - start) / step + 1E-6))]