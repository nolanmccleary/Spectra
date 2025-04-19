import torch
from utils import generate_perturbation_vectors_1d




def make_gradient_engine(func, tensor, device, func_device, num_perturbations):
    if tensor.dim() == 1:
        return Grayscale_Engine(func, tensor, device, func_device, num_perturbations)
    elif tensor.dim() == 3:
        print("Warning! RGB gradient calculation not yet supported")
        return RGB_Engine(func, tensor, device, num_perturbations)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")




class Gradient_Engine:

    def __init__(self, func, tensor, device, func_device, num_perturbations):    #device parameter needs to be the same as the tensor and the func's respective devices
        self.func = func
        self.tensor = tensor.clone().to(device)
        self.device = device
        self.func_device = func_device
        self.num_perturbations = num_perturbations
        self.gradient = torch.zeros_like(self.tensor)


    def compute_gradient(self, old_hash, scale_factor):
        raise NotImplementedError("Subclasses must override gradient compute ops")


    def update_tensor(self, step):
        self.tensor.add_(step)


    def set_tensor(self, tensor):
        assert tensor.dim() == self.tensor.dim(), f"Error: Input tensor dimensionality {tensor.dim()} does not match engine tensor dimensionality {self.tensor.dim()}"
        self.tensor = tensor.clone().to(self.device)


    def l2_delta_from_engine_tensor(self, tensor):
        raise NotImplementedError("Subclasses must override gradient compute ops")




class Grayscale_Engine(Gradient_Engine):

    def __init__(self, func, tensor, device, num_perturbations):    #device parameter needs to be the same as the tensor and the func's respective devices
        self.tensor_image_size = tensor.numel()
        assert self.tensor.dim() == 1, f"Expected 1D tensor, got {self.tensor.dim()}D tensor."
        super().__init__(func, tensor, device, num_perturbations)


    def compute_gradient(self, old_hash, scale_factor):
        perturbations = generate_perturbation_vectors_1d(self.num_perturbations, self.tensor_image_size // 2, self.device) #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        batch_pert = perturbations.mul_(scale_factor)   #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] = c[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        
        cand_batch = (self.tensor + batch_pert).to(self.func_device) #[t1, t2, t3] + [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] -> [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]] where cxy = t[y] + p[x,y]

        new_hashes = torch.tensor([self.func(v) for v in cand_batch], dtype=torch.int32, device=self.device)     #[f[c11, c12, c13], f[c21, c22, c23], f[c31, c32, c33]] -> [h1, h2, h3]
        orig_hash = torch.tensor(int(old_hash, 16), dtype=torch.int32, device=self.device) # h_old -> [h_old]
        
        x = orig_hash ^ new_hashes  #[h_old], [h1, h2, h3] -> [x1, x2, x3]
        hamming_deltas = x.bit_count().to(cand_batch.dtype) #[x1, x2, x3] -> [d1, d2, d3]

        gradient = (hamming_deltas.unsqueeze(1) * batch_pert).sum(dim=0).to(self.device)  #[d1, d2, d3] -> VecSum([[d1], [d2], [d3]] * [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]) -> [g1, g2, g3] where gx = [dx] * [px1, px2, px3]
        return gradient


    def l2_delta_from_engine_tensor(self, tensor):
        diff = self.tensor - tensor
        return torch.linalg.vector_norm(diff, ord=2).item() / diff.numel()





#DUMMY CLASS FOR NOW
class RGB_Engine(Gradient_Engine):

    def __init__(self, func, tensor, device, num_perturbations):
        self.height = tensor.size(1)
        self.width = tensor.size(2)
        self.tensor_image_size = self.height * self.width
        assert self.tensor.dim() == 3, f"Expected 3D tensor, got {self.tensor.dim()}D tensor."
        super().__init__(func, tensor, device, num_perturbations)


    #TODO: Make these RGB-friendly
    def compute_gradient(self, scale_factor, old_hash):
        print("RGB Gradient compute!")
        perturbations = generate_perturbation_vectors_1d(self.num_perturbations, self.tensor_image_size // 2, self.device) #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        batch_pert = perturbations.mul_(scale_factor)   #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] = c[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        
        base = self.tensor.view(1, -1)
        cand_batch = base + batch_pert #[t1, t2, t3] + [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] -> [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]] where cxy = t[y] + p[x,y]

        new_hashes = torch.tensor([self.func(v.view_as(self.tensor), device=self.device) for v in cand_batch], dtype=torch.int32, device=self.device)     #[f[c11, c12, c13], f[c21, c22, c23], f[c31, c32, c33]] -> [h1, h2, h3]
        orig_hash = torch.tensor(int(old_hash, 16), dtype=torch.int32, device=self.device) # h_old -> [h_old]
        
        x = orig_hash ^ new_hashes  #[h_old], [h1, h2, h3] -> [x1, x2, x3]
        hamming_deltas = x.bit_count().to(cand_batch.dtype) #[x1, x2, x3] -> [d1, d2, d3]

        gradient = (hamming_deltas.unsqueeze(1) * batch_pert).sum(dim=0).to(self.device)  #[d1, d2, d3] -> VecSum([[d1], [d2], [d3]] * [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]) -> [g1, g2, g3] where gx = [dx] * [px1, px2, px3]
        return gradient


    def l2_delta_from_engine_tensor(self, tensor):
        C, H, W = tensor.shape
        diff = (tensor - self.tensor).view(C, -1) #Flatten each color's matrix to 1-d array
        return torch.norm(diff, p=2, dim=0).mean().item() #Convert 3xN arrays to 3N array
