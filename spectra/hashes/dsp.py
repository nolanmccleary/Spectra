import math
import torch
import torch.nn.functional as F



PI = math.pi
_dct_cache = {}
_window_cache = {}

def create_dct_matrix(N, device, dtype): #For 2D Orthnormal DCT
    global _dct_cache
    key = (N, device, dtype)
    if key not in _dct_cache:

        n = torch.arange(N, device=device, dtype=dtype)
        k = n.unsqueeze(0) #[1, N]

        basis = torch.cos(PI * (2 * n + 1).unsqueeze(1) * k / (2 * N)) #[N, 1] * [1, N] -> [N, N]; broadcast across k so we have N dct row vectors of length N
        basis = basis.to(device)

        _dct_cache[key] = basis

    return _dct_cache[key]


def _get_window(N, w, device):
    global _window_cache
    key = (N, w, device)

    if key not in _window_cache:
        half = (w + 2) // 2
        
        p2 = w - half + 1
        p3 = N - w
        
        j = torch.arange(N, device=device)
        l = torch.empty(N, dtype=torch.long, device=device)
        r = torch.empty(N, dtype=torch.long, device=device)

        mask1 = j < p2

        l[mask1] = 0
        r[mask1] = (half - 1) + j[mask1]

        mask2 = (j >= p2) & (j < p2 + p3)
        i2 = j[mask2] - p2
        l[mask2] = i2 + 1
        r[mask2] = i2 + w

        mask3 = j >= (p2 + p3)
        l[mask3] = j[mask3] - (w - half)
        r[mask3] = N - 1

        _window_cache[key] = (l, r)

    return _window_cache[key]


def _box_filter_1d(tensor, w, dim):
    N = tensor.size(dim)
    device = tensor.device
    l, r = _get_window(N, w, device)

    ps = torch.cat([                                              
        torch.zeros_like(tensor.select(dim, 0).unsqueeze(dim)), #prepend padding
        tensor.cumsum(dim=dim) #cumsum along the 'dim'th slice'
    ], dim=dim)  # shape[..., N+1]

    shape = list(tensor.shape)
    idx_shape = shape.copy()
    idx_shape[dim] = N
    l_idx = l
    r_idx = r + 1

    for _ in range(dim):
        l_idx = l_idx.unsqueeze(0)
        r_idx = r_idx.unsqueeze(0)

    for _ in range(tensor.ndim - dim - 1):
        l_idx = l_idx.unsqueeze(-1)
        r_idx = r_idx.unsqueeze(-1)

    l_idx = l_idx.expand(idx_shape)
    r_idx = r_idx.expand(idx_shape)

    sum_windows = ps.gather(dim, r_idx) - ps.gather(dim, l_idx)
    #counts = (r - l + 1).to(tensor.dtype).view(
    #    *([1]*dim + [N] + [1]*(tensor.ndim-dim-1))
    #)
    #return sum_windows.div(counts)
    return sum_windows.div(4)


def jarosz_pdq_tent(tensor, box_size):
    out = tensor
    for _ in range(2):
        out = _box_filter_1d(out, box_size, dim=2)  # Filter pass along rows
        out = _box_filter_1d(out, box_size, dim=1)  # Filter pass along columns
    return out


def pdq_decimate(tensor, D=64):
    C, H, W = tensor.shape
    device = tensor.device

    idxH = torch.floor(((torch.arange(D, device=device, dtype=torch.float) + 0.5) * H) / D).long()
    idxW = torch.floor(((torch.arange(D, device=device, dtype=torch.float) + 0.5) * W) / D).long()

    return tensor.index_select(1, idxH).index_select(2, idxW)


def jarosz_filter(tensor, out_dim=64, box_size = 4):
    blurred = jarosz_pdq_tent(tensor, box_size)
    return pdq_decimate(blurred, D=out_dim)
