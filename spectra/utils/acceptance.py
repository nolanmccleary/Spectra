from typing import Callable, Tuple
import torch

__all__ = [
    'create_acceptance'
]

def _lpips_acceptance(ao) -> Callable[[torch.Tensor, int], Tuple[bool, bool]]:
    """LPIPS-based acceptance closure bound to an `Attack_Object`."""
    def _fn(tensor: torch.Tensor, step_number: int):
        ao.metrics.current_hash    = ao.hash_func(tensor.to(ao.hash_func_device))
        ao.metrics.current_hamming = int((ao.input_tensors.original_hash != ao.metrics.current_hash).sum().item())
        ao.metrics.current_lpips   = ao.lpips_func(ao.input_tensors.working_tensor, tensor)
        ao.metrics.current_l2      = ao.l2_func(ao.input_tensors.working_tensor, tensor)

        break_loop, accepted = False, False
        if ao.gate is not None and ao.metrics.current_lpips >= ao.gate:
            break_loop = True

        if ao.metrics.current_hamming >= ao.hamming_threshold:
            if ao.metrics.current_lpips < ao.metrics.output_lpips:
                ao.metrics.output_lpips   = ao.metrics.current_lpips
                ao.metrics.output_l2      = ao.metrics.current_l2
                ao.metrics.output_hash    = ao.metrics.current_hash
                ao.metrics.output_hamming = ao.metrics.current_hamming
                accepted = True
            break_loop = True
        return break_loop, accepted
    return _fn


def _l2_acceptance(ao):
    def _fn(tensor: torch.Tensor, step_number: int):
        ao.current_hash    = ao.hash_func(tensor.to(ao.hash_func_device))
        ao.current_hamming = int((ao.original_hash != ao.current_hash).sum().item())
        ao.metrics.current_lpips   = ao.lpips_func(ao.input_tensors.working_tensor, tensor)
        ao.metrics.current_l2      = ao.l2_func(ao.input_tensors.working_tensor, tensor)

        break_loop, accepted = False, False
        if ao.gate is not None and ao.current_l2 >= ao.gate:
            break_loop = True

        if ao.current_hamming >= ao.hamming_threshold:
            if ao.current_l2 < ao.metrics.output_l2:
                ao.metrics.output_l2      = ao.metrics.current_l2
                ao.metrics.output_lpips   = ao.metrics.current_lpips
                ao.metrics.output_hash    = ao.metrics.current_hash
                ao.metrics.output_hamming = ao.metrics.current_hamming
                accepted = True
            break_loop = True
        return break_loop, accepted
    return _fn


def _latching_acceptance(ao):
    def _fn(tensor: torch.Tensor, step_number: int):
        ao.current_hash    = ao.hash_func(tensor.to(ao.hash_func_device))
        ao.current_hamming = int((ao.original_hash != ao.current_hash).sum().item())
        ao.metrics.current_lpips   = ao.lpips_func(ao.input_tensors.working_tensor, tensor)
        ao.metrics.current_l2      = ao.l2_func(ao.input_tensors.working_tensor, tensor)
        if ao.current_hamming >= ao.hamming_threshold:
            return True, True
        return False, False
    return _fn


def _step_acceptance(ao):
    def _fn(tensor: torch.Tensor, step_number: int):
        ao.current_hash    = ao.hash_func(tensor.to(ao.hash_func_device))
        ao.current_hamming = int((ao.original_hash != ao.current_hash).sum().item())
        ao.metrics.current_lpips   = ao.lpips_func(ao.input_tensors.working_tensor, tensor)
        ao.metrics.current_l2      = ao.l2_func(ao.input_tensors.working_tensor, tensor)
        if ao.current_hamming >= ao.hamming_threshold:
            if step_number < ao.min_steps:
                ao.min_steps = step_number
                return True, True
            return True, False
        return False, False
    return _fn


_ACCEPTANCE_MAP = {
    'lpips' : _lpips_acceptance,
    'l2'    : _l2_acceptance,
    'latch' : _latching_acceptance,
    'step'  : _step_acceptance,
}


def create_acceptance(attack_obj, kind: str):
    """Return an acceptance-criterion closure bound to `attack_obj`."""
    if kind not in _ACCEPTANCE_MAP:
        raise ValueError(
            f"Unknown acceptance kind '{kind}'. Valid kinds: {list(_ACCEPTANCE_MAP)}")
    return _ACCEPTANCE_MAP[kind](attack_obj) 