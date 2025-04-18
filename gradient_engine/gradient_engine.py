import torch
import utils




class Grayscale_Engine():
    def __init__(self, func, tensor, num_perturbations, device):    #device parameter needs to be the same as the tensor and the func's respective devices
        self.func = func
        self.tensor = tensor.clone().squeeze(0).to(device)
        self.tensor_size = self.tensor.numel()
        self.num_perturbations = num_perturbations
        self.device = device
        self.gradient = torch.zeros_like(self.tensor)
        assert self.tensor.dim() == 1, f"Expected 1D tensor, got {self.tensor.dim()}D tensor."



    def compute_gradient(self, scale_factor, old_hash):
        perturbations = utils.generate_perturbation_vectors_1d(self.num_perturbations, self.tensor_size // 2, self.device) #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        batch_pert = perturbations.mul_(scale_factor)   #[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] = c[[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]
        
        cand_batch = self.tensor + batch_pert #[t1, t2, t3] + [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]] -> [[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]] where cxy = t[y] + p[x,y]

        new_hashes = torch.tensor([self.func(v, device=self.device) for v in cand_batch], dtype=torch.int32, device=self.device)     #[f[c11, c12, c13], f[c21, c22, c23], f[c31, c32, c33]] -> [h1, h2, h3]
        orig_hash = torch.tensor(int(old_hash, 16), dtype=torch.int32, device=self.device) # h_old -> [h_old]
        
        x = orig_hash ^ new_hashes  #[h_old], [h1, h2, h3] -> [x1, x2, x3]
        hamming_deltas = x.bit_count().to(cand_batch.dtype) #[x1, x2, x3] -> [d1, d2, d3]

        gradient = (hamming_deltas.unsqueeze(1) * batch_pert).sum(dim=0).to(self.device)  #[d1, d2, d3] -> VecSum([[d1], [d2], [d3]] * [[p11, p12, p13], [p21, p22, p23], [p31, p32, p33]]) -> [g1, g2, g3] where gx = [dx] * [px1, px2, px3]
        return gradient



    def update_tensor(self, step):
        self.tensor.add_(step)








#TODO: Get rgb engine behaviour working
class RGB_Engine():
    def __init__(self, func, tensor, num_perturbations, device):
        self.func = func
        self.tensor = tensor.clone().squeeze(0) #Squeeze to handle RGBA format if present
        self.tensor_size = tensor.numel()
        self.height = tensor.size(1)
        self.width = tensor.size(2)
        self.num_perturbations = num_perturbations
        assert self.tensor.dim() == 3, f"Expected 4D tensor, got {self.tensor.dim()}D tensor."