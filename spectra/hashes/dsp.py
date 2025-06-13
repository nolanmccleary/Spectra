import math
import torch
import torch.nn.functional as F



PI = math.pi
_dct_cache = {}

def create_dct_matrix(N, device, dtype): #For 2D Orthnormal DCT
    global _dct_cache
    key = (N, device, dtype)
    if key not in _dct_cache:

        n = torch.arange(N, device=device, dtype=dtype)
        k = n.unsqueeze(0) #[1, N]

        basis = torch.cos(PI * (2 * n + 1).unsqueeze(1) * k / (2 * N)) #[N, 1] * [1, N] -> [N, N]; broadcast across k so we have N dct row vectors of length N
        basis = basis.t().to()

        _dct_cache[key] = basis

    return _dct_cache[key]


def _get_window(N, w, device):
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

    return l, r


def _box_filter_1d(x, w, dim):
    N = x.size(dim)
    device = x.device
    l, r = _get_window(N, w, device)

    ps = torch.cat([
        torch.zeros_like(x.select(dim, 0).unsqueeze(dim)),
        x.cumsum(dim=dim)
    ], dim=dim)  # shape[..., N+1]

    r1 = (r + 1)

    shape = list(x.shape)
    idx_shape = shape.copy()
    idx_shape[dim] = N
    l_idx = l
    r1_idx = r1

    for _ in range(dim):
        l_idx = l_idx.unsqueeze(0)
        r1_idx = r1_idx.unsqueeze(0)

    for _ in range(x.ndim - dim - 1):
        l_idx = l_idx.unsqueeze(-1)
        r1_idx = r1_idx.unsqueeze(-1)

    l_idx = l_idx.expand(idx_shape)
    r1_idx = r1_idx.expand(idx_shape)

    sum_windows = ps.gather(dim, r1_idx) - ps.gather(dim, l_idx)
    counts = (r - l + 1).to(x.dtype).view(
        *([1]*dim + [N] + [1]*(x.ndim-dim-1))
    ).expand(idx_shape)

    return sum_windows.div(counts)


def jarosz_pdq_tent(x):
    C, H, W = x.shape

    full_w_W = math.ceil(W / 128)
    full_w_H = math.ceil(H / 128)
    out = x
    for _ in range(2):
        out = _box_filter_1d(out, full_w_W, dim=2)  # rows
        out = _box_filter_1d(out, full_w_H, dim=1)  # cols
    return out


def pdq_decimate(x, D=64):
    C, H, W = x.shape
    device = x.device

    idxH = torch.floor(((torch.arange(D, device=device, dtype=torch.float) + 0.5) * H) / D).long()
    idxW = torch.floor(((torch.arange(D, device=device, dtype=torch.float) + 0.5) * W) / D).long()

    return x.index_select(1, idxH).index_select(2, idxW)


def jarosz_filter(tensor, out_dim=64):
    blurred = jarosz_pdq_tent(tensor)
    return pdq_decimate(blurred, D=out_dim)
