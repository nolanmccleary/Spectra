import numpy as np
import torch
import torch.nn.functional as F



def get_rgb_tensor(image_object, rgb_device):
    if image_object.mode == 'RGBA':
        image_object = image_object.convert('RGB')
    arr = np.array(image_object).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(rgb_device)
    return tensor



def rgb_to_grayscale(rgb_tensor): #[C, H, W] -> [1, H, W]
    return torch.mean(rgb_tensor, dim=0, keepdim=True)



def rgb_to_luma(rgb_tensor): #[C, H, W] -> [1, H, W]
    r, g, b = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return gray.unsqueeze(0)



def tensor_resize(input_tensor, height, width):
    tensor = input_tensor.clone().unsqueeze(0) #[{3,1}, H, W] -> [1, {3, 1}, H, W]
    tensor_resized = F.interpolate(   #Interpolate needs to know batch and channel dimensions thus a 4-d tensor is required
        tensor,
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )
    return tensor_resized.squeeze(0) #[1, {3, 1}, H, W] -> [{3,1}, H, W]



def inverse_delta(rgb_tensor, delta, eps=1e-6):
    C, H, W = rgb_tensor.shape

    if delta.shape == (C, H, W):
        return delta

    rgb_mean = rgb_tensor.mean()  # [1, H, W]
    gd = delta.unsqueeze(0)              # [1, H, W]
    
    # Avoid division by zero
    delta = torch.where(
        gd <= 0,
        gd * rgb_tensor / (rgb_mean + eps),
        gd * (1 - rgb_tensor) / ((1 - rgb_mean) + eps)
    )

    return delta.view(C, H, W)



def generate_seed_perturbation(dim, start_scalar, device):
    return torch.rand((1, dim), dtype=torch.float32, device=device) * start_scalar



def generate_perturbation_vectors(num_perturbations, shape, device):
    base = torch.randn((num_perturbations // 2, *shape), dtype=torch.float32, device=device) 
    absmax = base.abs().amax(dim=1, keepdim=True)
    scale = torch.where(absmax > 0, 1.0 / absmax, torch.tensor(1.0, device = device)) #scale tensor to get max val of each generated perturbation so that we can normalize
    base = base * scale
    return torch.cat([base, -base], dim=0) #Mirror to preserve distribution



def to_hex(hash_bool):
    arr = hash_bool.view(-1).cpu().numpy().astype(np.uint8)
    packed = np.packbits(arr)  #Convert to byte array
    return '0x' + ''.join(f'{b:02x}' for b in packed.tolist()) #Format to hex



def bool_tensor_delta(a, b):
    return a.ne(b)



def byte_quantize(tensor):
    return torch.round(tensor * 255.0) / (255.0)



def l2_delta(a, b):
    return torch.sqrt(torch.mean((a - b).pow(2))).item()



def make_acceptance_func(self, acceptance_str):
    
    def lpips_acceptance_func(tensor, delta):
        self.current_hash = self.func(tensor.to(self.func_device))
        self.current_hamming = int((self.original_hash != self.current_hash).sum().item())

        if self.current_hamming >= self.hamming_threshold:
            lpips_distance = self.lpips_func(self.tensor, tensor)

            if lpips_distance < self.output_lpips:
                self.output_lpips = lpips_distance
                return True
            
            else:   #Lpips distance more or less increases monotonically so once we know it isn't better than our current best we may as well re-start; <- NEED TO TEST THIS
                delta.zero_()
                tensor = self.tensor.clone()
        return False

    def l2_acceptance_func(tensor, delta):
        self.current_hash = self.func(tensor.to(self.func_device))
        self.current_hamming = int((self.original_hash != self.current_hash).sum().item())

        if self.current_hamming >= self.hamming_threshold:
            l2_distance = l2_delta(self.tensor, tensor)

            if l2_distance < self.output_l2:
                self.output_l2 = l2_distance
                return True
            
            else:   
                delta.zero_()
                tensor = self.tensor.clone()
        return False
    
    def latching_acceptance_func(tensor, delta):
        self.current_hash = self.func(tensor.to(self.func_device))
        self.current_hamming = int((self.original_hash != self.current_hash).sum().item())
        return True

    acceptance_table = {"lpips" : lpips_acceptance_func, "l2" : l2_acceptance_func, "latch" : latching_acceptance_func}
    if acceptance_str not in acceptance_table.keys():
        raise ValueError(f"'{acceptance_str}' not in set of valid acceptance function handles: {acceptance_table.keys()}")

    return acceptance_table[acceptance_str]
