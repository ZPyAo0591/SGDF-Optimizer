"""Optional fp32 CUDA Triton fused update for standard and Sign-SGDF.

Importing this module requires Triton; main SGDF imports it only for fused=True.
This path is not graph-capturable.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sgdf_kernel(P, G, M, V, N: tl.constexpr,
                 LR: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr,
                 EPS: tl.constexpr, GAMMA: tl.constexpr, WD: tl.constexpr,
                 DECOUPLE: tl.constexpr, MAXIMIZE: tl.constexpr,
                 SIGN: tl.constexpr, INVM: tl.constexpr,
                 SFACTOR: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    p = tl.load(P + idx, mask, other=0)
    g = tl.load(G + idx, mask, other=0)
    m = tl.load(M + idx, mask, other=0)
    v = tl.load(V + idx, mask, other=0)
    if MAXIMIZE:
        g = -g
    if not DECOUPLE:
        g = g + WD * p
    new_m = B1 * m + (1.0 - B1) * g
    r = g - new_m
    new_v = B2 * v + (1.0 - B2) * r * r
    delta = g - INVM * new_m
    var = SFACTOR * new_v
    k = var / (var + delta * delta + EPS)
    if GAMMA == 0.5:
        k = tl.sqrt(k)
    elif GAMMA != 1.0:
        k = tl.exp(GAMMA * tl.log(k))
    direction = INVM * new_m + k * delta
    if SIGN:
        direction = tl.where(direction > 0, 1.0, tl.where(direction < 0, -1.0, 0.0))
    if DECOUPLE:
        p = p * (1.0 - LR * WD)
    p = p - LR * direction
    tl.store(P + idx, p, mask)
    tl.store(M + idx, new_m, mask)
    tl.store(V + idx, new_v, mask)


def fused_sgdf_step(p: torch.Tensor, grad: torch.Tensor,
                    m: torch.Tensor, v: torch.Tensor, *, lr: float,
                    beta1: float, beta2: float, eps: float, gamma: float,
                    wd: float, decouple: bool, maximize: bool, sign: bool,
                    inv_m: float, s_factor: float) -> None:
    if not (p.is_cuda and p.dtype == torch.float32 and p.is_contiguous() and
            grad.is_contiguous() and m.is_contiguous() and v.is_contiguous()):
        raise RuntimeError('Triton fused SGDF only supports contiguous CUDA fp32 tensors')
    _sgdf_kernel[(triton.cdiv(p.numel(), 256),)](
        p, grad, m, v, p.numel(), lr, beta1, beta2, eps, gamma, wd,
        decouple, maximize, sign, inv_m, s_factor, BLOCK=256,
        enable_fp_fusion=False,  # improve agreement with the PyTorch reference
    )
