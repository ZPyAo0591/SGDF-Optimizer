"""Newton-Schulz postprocessing of SGDF filtered gradients.

Distributed Muon chooses either all-gather or a single-all-reduce Gram
polynomial path for a sharded DTensor, never treating a shard as a full matrix.
"""
from __future__ import annotations

import math
import torch

_COEFF = (3.4445, -4.7750, 2.0315)


def _dtype(matrix: torch.Tensor, compute_dtype: str) -> torch.dtype:
    if matrix.dtype == torch.float64:
        # Never silently demote high-precision training to bf16.
        return torch.float64
    if compute_dtype == 'float32':
        return torch.float32
    if compute_dtype == 'bfloat16':
        if matrix.device.type != 'cuda' or not torch.cuda.is_bf16_supported():
            raise RuntimeError('muon_compute_dtype="bfloat16" requires bf16 CUDA support')
        return torch.bfloat16
    if compute_dtype != 'auto':
        raise ValueError('Invalid Muon compute dtype')
    return (torch.bfloat16 if matrix.device.type == 'cuda' and
            torch.cuda.is_bf16_supported() else torch.float32)


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5,
                                compute_dtype: str = 'auto') -> torch.Tensor:
    """Muon quintic Newton-Schulz, on a complete 2-D *local* matrix."""
    if matrix.ndim != 2:
        raise ValueError('Newton-Schulz expects a 2D matrix')
    if steps < 2:
        raise ValueError('steps must be >=2')
    x = matrix.to(dtype=_dtype(matrix, compute_dtype))
    flipped = x.shape[0] > x.shape[1]
    if flipped:
        x = x.T
    acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x = x / (torch.linalg.vector_norm(x.to(acc_dtype)) + 1e-7)
    a, b, c = _COEFF
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if flipped:
        x = x.T
    return x.to(dtype=matrix.dtype)


def _local_postprocess(update: torch.Tensor, ns_steps: int,
                       compute_dtype: str) -> torch.Tensor:
    if update.ndim < 2:
        return update
    mat = update.reshape(update.shape[0], -1)
    rows, cols = mat.shape
    orthogonal = zeropower_via_newtonschulz5(mat, steps=ns_steps,
                                             compute_dtype=compute_dtype)
    return (orthogonal * math.sqrt(max(1.0, rows / cols))).reshape_as(update)


def _gram_layout(update: torch.Tensor, gram_max_dim: int):
    """Return the shard axis and Gram dimension for a supported 1-D mesh."""
    from torch.distributed.tensor import Shard

    mesh, places = update.device_mesh, update.placements
    if (update.ndim != 2 or mesh.ndim != 1 or len(places) != 1 or
            not isinstance(places[0], Shard)):
        return None
    axis = places[0].dim
    if axis not in (0, 1):
        return None
    dim = update.shape[1 - axis]
    return (axis, dim) if dim <= gram_max_dim else None


def _gram_postprocess(update: torch.Tensor, ns_steps: int, gram_max_dim: int,
                      compute_dtype: str) -> torch.Tensor:
    """Single-all-reduce distributed Muon for a row- or column-sharded matrix.

    Newton-Schulz is a matrix polynomial: X_{k+1} = X_k P(G_k) when
    G_k=X_k.T@X_k (or P(G_k)@X_k for G_k=X_k@X_k.T). Hence
    G_{k+1}=P(G_k)G_kP(G_k), and we can accumulate P(G_k) on each rank.
    Only the initial Gram matrix requires an all-reduce. Its trace also
    supplies the global Frobenius norm, avoiding a separate norm reduce.
    """
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    layout = _gram_layout(update, gram_max_dim)
    if layout is None:
        raise RuntimeError('muon_distributed="gram" requires a 2D DTensor on '
                           'a 1-D Shard(0)/Shard(1) mesh with a small enough '
                           'unsharded dimension; try muon_distributed="gather"')
    axis, dim = layout
    local = update.to_local()
    compute = local.to(dtype=_dtype(local, compute_dtype))
    # Tiny Gram matrices benefit from fp64 recurrence: the quintic NS
    # coefficients amplify rounding in the Gram's null-space. Larger
    # matrices retain fp32 matmul throughput on GPUs.
    acc_dtype = (torch.float64 if compute.dtype == torch.float64 or dim <= 64
                 else torch.float32)
    x = compute.to(dtype=acc_dtype)

    # The all-reduced Gram gives both the global norm and the starting
    # polynomial, including when shards have unequal or zero local rows.
    gram = x.T @ x if axis == 0 else x @ x.T
    dist.all_reduce(gram, op=dist.ReduceOp.SUM, group=update.device_mesh.get_group())
    norm = torch.diagonal(gram).sum().clamp_min(0).sqrt() + 1e-7
    gram = gram / (norm * norm)

    a, b, c = _COEFF
    identity = torch.eye(dim, dtype=acc_dtype, device=x.device)
    transform = identity
    for step in range(ns_steps):
        polynomial = a * identity + b * gram + c * (gram @ gram)
        transform = transform @ polynomial
        if step + 1 < ns_steps:
            gram = polynomial @ gram @ polynomial
            # Counter small floating point asymmetry without communication.
            gram = (gram + gram.T) * 0.5

    # One local matrix multiplication applies the accumulated orthogonalizer.
    # Accumulate in fp32 even when auto chooses bf16 for the input. This
    # prevents repeated low-precision rounding in the Gram recurrence.
    output = (x @ transform if axis == 0 else transform @ x) / norm
    rows, cols = update.shape
    output = (output * math.sqrt(max(1.0, rows / cols))).to(dtype=update.dtype)
    return DTensor.from_local(output, device_mesh=update.device_mesh,
                              placements=update.placements,
                              shape=update.shape, stride=update.stride())


def _automatic_mode(update: torch.Tensor, gram_max_dim: int,
                    gather_max_bytes: int) -> str:
    """Use the cheaper communication layout, with an all-gather memory cap."""
    nbytes = update.numel() * update.element_size()
    layout = _gram_layout(update, gram_max_dim)
    if layout is not None:
        _, dim = layout
        # One all-reduced Gram costs ~2*dim**2 elements of network traffic;
        # gathering the full matrix costs roughly its total element count.
        # Favor Gram when it is smaller, or when gathering would exceed cap.
        if 2 * dim * dim <= update.numel() or nbytes > gather_max_bytes:
            return 'gram'
    if nbytes <= gather_max_bytes:
        return 'gather'
    raise RuntimeError(
        'Muon automatic distributed selection cannot safely gather this '
        f'{nbytes}-byte matrix (muon_gather_max_bytes={gather_max_bytes}) '
        'and Gram is unavailable. Increase the limit, muon_gram_max_dim, '
        'or adjust the sharding layout.'
    )

def muon_postprocess(update: torch.Tensor, ns_steps: int = 5,
                     distributed: str = 'auto', gram_max_dim: int = 4096,
                     compute_dtype: str = 'auto',
                     gather_max_bytes: int = 256 * 1024 * 1024) -> torch.Tensor:
    """Orthogonalize a full matrix, or explicitly gather/Gram-reduce DTensor.

    Ordinary torch.Tensor is always treated as full: caller must reject FSDP1
    flat parameter views or other manually sharded raw tensors.
    """
    if update.ndim < 2:
        return update
    try:
        from torch.distributed.tensor import DTensor, Replicate, Shard, Partial
    except ImportError:
        DTensor = None
    if DTensor is None or not isinstance(update, DTensor):
        return _local_postprocess(update, ns_steps, compute_dtype)
    if any(isinstance(p, Partial) for p in update.placements):
        raise RuntimeError('Muon does not support DTensor Partial placements')
    if not any(isinstance(p, Shard) for p in update.placements):
        local = _local_postprocess(update.to_local(), ns_steps, compute_dtype)
        return DTensor.from_local(local, device_mesh=update.device_mesh,
                                  placements=update.placements,
                                  shape=update.shape, stride=update.stride())
    if distributed == 'auto':
        distributed = _automatic_mode(update, gram_max_dim, gather_max_bytes)
    if distributed == 'error':
        raise RuntimeError('Muon on sharded DTensor requires auto, gather or gram mode')
    if distributed == 'gram':
        return _gram_postprocess(update, ns_steps, gram_max_dim, compute_dtype)
    if distributed != 'gather':
        raise ValueError('Muon distributed mode must be auto, error, gather, or gram')
    replicate = (Replicate(),) * update.device_mesh.ndim
    whole = update.redistribute(placements=replicate)
    local_out = _local_postprocess(whole.to_local(), ns_steps, compute_dtype)
    full = DTensor.from_local(local_out, device_mesh=update.device_mesh,
                              placements=replicate)
    return full.redistribute(placements=update.placements)
