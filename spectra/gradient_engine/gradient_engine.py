import torch
from spectra.utils import generate_perturbation_vectors_1d

EPS = 1e-6

def make_gradient_engine(func, tensor, device, func_device, num_perturbations, height, width, loss_func, is_grayscale=True):
    if is_grayscale:
        return Grayscale_Engine(func, tensor, device, func_device, num_perturbations, height, width, loss_func)
    else:
        print("Warning! RGB gradient calculation not yet supported")
        return RGB_Engine(func, tensor, device, num_perturbations)


class Gradient_Engine:

    def __init__(self, func, tensor, device, func_device, num_perturbations, height, width, loss_func):    #device parameter needs to be the same as the tensor and the func's respective devices
        self.func = func
        self.device = device
        #self.tensor = tensor.clone().to(self.device)
        self.func_device = func_device
        self.num_perturbations = num_perturbations
        self.height = height
        self.width = width
        self.loss_func = loss_func
        #self.gradient = torch.zeros_like(self.tensor)


    def compute_gradient(self, old_hash, scale_factor):
        raise NotImplementedError("Subclasses must override gradient compute ops")


    def update_tensor(self, step):
        self.tensor.add_(step)


    def set_tensor(self, tensor):
        raise NotImplementedError("Subclasses must override tensor setting")


    def lpips_delta_from_engine_tensor(self, tensor):
        raise NotImplementedError("Subclasses must override LPIPS delta compute")




class Grayscale_Engine(Gradient_Engine):



    def __init__(self, func, tensor, device, func_device, num_perturbations, height, width, loss_func):    #device parameter needs to be the same as the tensor and the func's respective devices
        
        super().__init__(func, tensor, device, func_device, num_perturbations, height, width, loss_func)
        
        self.tensor = tensor.to(self.device)
        self.gradient = torch.zeros_like(self.tensor)
        self.tensor_image_size = self.tensor.numel()



    def compute_gradient(self, last_hash, scale_factor):
        perturbations = generate_perturbation_vectors_1d(self.num_perturbations, self.height, self.width, self.device) #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        
        batch_pert = perturbations.mul(scale_factor)   #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] = c[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]    
        

        pos_scale = torch.where(
            batch_pert > 0,
            (1.0 - self.tensor) / (batch_pert + EPS),
            torch.tensor(1.0, device=self.device),
        )
        neg_scale = torch.where(
            batch_pert < 0,
            (0.0 - self.tensor) / (batch_pert - EPS),
            torch.tensor(1.0, device=self.device),
        )


        safe_scale = torch.min(pos_scale, neg_scale).clamp(max=1.0)
        batch_pert.mul(safe_scale)

        cand_batch = (self.tensor + batch_pert).to(self.func_device).clamp(0.0, 1.0) #[t1, t2, t3] + [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] -> [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]] where cxy = t[y] + p[x,y]
        quant = torch.round(cand_batch * 255.0) / 255.0


        new_hashes = torch.stack([self.func(v, self.height, self.width) for v in quant], dim=0).to(self.device) #[NUM_PERTURBATIONS, N_BITS]
        diffs = new_hashes.ne(last_hash)
        hamming_deltas = diffs.sum(dim=1).to(self.tensor.dtype)


        gradient = (hamming_deltas.view(self.num_perturbations, 1, 1) * batch_pert.to(self.device)).sum(dim=0).to(self.device).view(1, self.height, self.width)  #[d1, d2, d3] -> VecSum([[d1], [d2], [d3]] * [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]) -> [g1, g2, g3] where gx = [dx] * [px1, px2, px3]
        return gradient



    def lpips_delta_from_engine_tensor(self, new_tensor):

        a = self.tensor.view(1, 1, self.height, self.width) * 2.0 - 1.0
        b = new_tensor.view(1, 1, self.height, self.width) * 2.0 - 1.0

        a3 = a.repeat(1, 3, 1, 1)
        b3 = b.repeat(1, 3, 1, 1)

        return self.loss_func(a3, b3).item()
            
        

        


#DUMMY CLASS FOR NOW
class RGB_Engine(Gradient_Engine):

    def __init__(self, func, tensor, device, func_device, num_perturbations):
        '''self.height = tensor.size(1)
        self.width = tensor.size(2)
        self.tensor_image_size = self.height * self.width
        assert self.tensor.dim() == 3, f"Expected 3D tensor, got {self.tensor.dim()}D tensor."
        super().__init__(func, tensor, device, func_device, num_perturbations)'''
        pass

#DO NOT EDIT!
    #TODO: Make these RGB-friendly
    def compute_gradient(self, old_hash, scale_factor):
        '''
        perturbations = generate_perturbation_vectors_1d(self.num_perturbations, self.tensor_image_size // 2, self.device) #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        batch_pert = perturbations.mul_(scale_factor)   #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] = c[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        
        base = self.tensor.view(1, -1)
        cand_batch = base + batch_pert #[t1, t2, t3] + [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] -> [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]] where cxy = t[y] + p[x,y]

        new_hashes = torch.tensor([to_signed_int64(self.func(v.view_as(self.tensor)), device=self.device) for v in cand_batch], dtype=torch.int64, device=self.device)     #[f[c11, c12, c13], f[c21, c22, c23], f[c31, c32, c33]] -> [h1, h2, h3]
        orig_hash = torch.tensor(int(old_hash, 16), dtype=torch.int64, device=self.device) # h_old -> [h_old]
        
        x = orig_hash ^ new_hashes  #[h_old], [h1, h2, h3] -> [x1, x2, x3]
        hamming_deltas = x.bit_count().to(cand_batch.dtype) #[x1, x2, x3] -> [d1, d2, d3]

        gradient = (hamming_deltas.unsqueeze(1) * batch_pert).sum(dim=0).to(self.device)  #[d1, d2, d3] -> VecSum([[d1], [d2], [d3]] * [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]) -> [g1, g2, g3] where gx = [dx] * [px1, px2, px3]
        return gradient'''
        return

#DO NOT EDIT!
    def lpips_delta_from_engine_tensor(self, tensor):
        '''C, H, W = tensor.shape
        diff = (tensor - self.tensor).view(C, -1) #Flatten each color's matrix to 1-d array
        return torch.norm(diff, p=2, dim=0).mean().item() #Convert 3xN arrays to 3N array'''
        return
