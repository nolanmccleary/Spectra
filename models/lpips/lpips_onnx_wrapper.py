import onnxruntime as ort
import numpy as np
import torch


class ALEX_ONNX:
    def __init__(self, model_path="lpips_alex.onnx", device="cpu"):
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps" and "MPSExecutionProvider" in ort.get_available_providers():
            providers = ["MPSExecutionProvider", "MPSExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        self.session = ort.InferenceSession(model_path, providers=providers)


    def get_lpips(self, old_tensor: torch.Tensor, new_tensor: torch.Tensor) -> float:
        a3 = None
        b3 = None

        C, H, W = old_tensor.shape

        if C == 1:
            a = old_tensor.view(1, 1, H, W) * 2.0 - 1.0
            b = new_tensor.view(1, 1, H, W) * 2.0 - 1.0

            a3 = a.repeat(1, 3, 1, 1)
            b3 = b.repeat(1, 3, 1, 1)

        else:
            a3 = old_tensor.unsqueeze(0)
            b3 = new_tensor.unsqueeze(0)

        npA = a3.detach().cpu().numpy().astype(np.float32)
        npB = b3.detach().cpu().numpy().astype(np.float32)

        outputs = self.session.run("lpips_out", {"inA": npA, "inB": npB})
        val = float(outputs[0].item())  # shape (1,1,1,1) -> scalar
        return val