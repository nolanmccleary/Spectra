import torch
from spectra.utils import generate_perturbation_vectors

EPS = 1e-6



class Gradient_Engine:

    def __init__(self, func, tensor, device, func_device, num_perturbations, height, width, loss_func):    #device parameter needs to be the same as the tensor and the func's respective devices
        self.func = func
        self.device = device
        self.func_device = func_device
        self.num_perturbations = num_perturbations
        self.height = height
        self.width = width
        self.loss_func = loss_func
        self.tensor = tensor.to(self.device)
        self.num_channels = self.tensor.shape[0]
        self.gradient = torch.zeros_like(self.tensor)
        self.tensor_image_size = self.tensor.numel()
        


    def compute_gradient(self, last_hash, scale_factor):
        perturbations = generate_perturbation_vectors(self.num_perturbations, self.num_channels, self.height, self.width, self.device) #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
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

        new_hashes = torch.stack([self.func(v) for v in quant], dim=0).to(self.device) #[NUM_PERTURBATIONS, N_BITS]
        
        diffs = new_hashes.ne(last_hash)
        hamming_deltas = diffs.sum(dim=1).to(self.tensor.dtype).view(self.num_perturbations, 1, 1, 1)

        gradient = (hamming_deltas * batch_pert.to(self.device)).sum(dim=0).to(self.device).view(self.num_channels, self.height, self.width)  #[d1, d2, d3] -> VecSum([[d1], [d2], [d3]] * [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]) -> [g1, g2, g3] where gx = [dx] * [px1, px2, px3]
        return gradient



    def lpips_delta_from_engine_tensor(self, new_tensor):
        a3 = None
        b3 = None

        if self.num_channels == 1:
            a = self.tensor.view(1, 1, self.height, self.width) * 2.0 - 1.0
            b = new_tensor.view(1, 1, self.height, self.width) * 2.0 - 1.0

            a3 = a.repeat(1, 3, 1, 1)
            b3 = b.repeat(1, 3, 1, 1)

        else:
            a3 = self.tensor.unsqueeze(0)
            b3 = new_tensor.unsqueeze(0)
        
        return self.loss_func(a3, b3).item()
