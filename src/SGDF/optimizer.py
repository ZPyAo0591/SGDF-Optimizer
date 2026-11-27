"""SGDF optimizer for PyTorch (Algorithm 1, CVPR 2026).

Single torch.optim.Optimizer subclass with optional sign or Muon update directions.
The two moment states always use torch.zeros_like(parameter), as torch.optim.Adam
normally does. Thus fp16/bfloat16 parameters have fp16/bfloat16 states; standard
AMP with fp32 model parameters keeps fp32 states, just as torch.optim.Adam.

K = (s_hat / (s_hat + (grad - m_hat)**2 + eps)) ** gamma.
Epsilon is INSIDE the denominator. An optional Triton fused CUDA path and explicit distributed Muon modes are available.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import torch
from torch.optim import Optimizer

try:
    from torch.distributed.tensor import DTensor, Shard, Partial
except ImportError:  # Older PyTorch versions may not expose DTensor.
    DTensor = None
    Shard = None
    Partial = None


def _is_dtensor(p: torch.Tensor) -> bool:
    return DTensor is not None and isinstance(p, DTensor)


_DEFAULT_FOREACH_BYTES = 16 * 1024 * 1024


def _validate_group(g: dict) -> None:
    try:
        b1, b2 = g['betas']
    except (TypeError, ValueError) as exc:
        raise ValueError('betas must have two values') from exc
    if not (math.isfinite(float(g['lr'])) and g['lr'] >= 0):
        raise ValueError('lr must be finite and >= 0')
    if not (0 <= b1 < 1 and 0 <= b2 < 1):
        raise ValueError('betas must be in [0, 1)')
    if not (math.isfinite(float(g['eps'])) and g['eps'] > 0):
        raise ValueError('eps must be finite and > 0')
    if g['gamma'] is not None and not (math.isfinite(float(g['gamma'])) and g['gamma'] > 0):
        raise ValueError('gamma must be None or finite and > 0')
    if not (math.isfinite(float(g['weight_decay'])) and g['weight_decay'] >= 0):
        raise ValueError('weight_decay must be finite and >= 0')
    if g['foreach'] not in (None, True, False):
        raise ValueError('foreach must be None, True or False')
    if type(g['foreach_max_bytes']) is not int or g['foreach_max_bytes'] < 1024:
        raise ValueError('foreach_max_bytes must be an integer >= 1024')
    if type(g['ns_steps']) is not int or g['ns_steps'] < 2:
        raise ValueError('ns_steps must be an integer >= 2')
    if not isinstance(g['use_sign'], bool) or not isinstance(g['use_muon'], bool):
        raise ValueError('use_sign and use_muon must be bool')
    if g['use_sign'] and g['use_muon']:
        raise ValueError('use_sign and use_muon cannot both be enabled')
    if g['capturable'] not in (True, False):
        raise ValueError('capturable must be bool')
    if g['maximize'] not in (True, False):
        raise ValueError('maximize must be bool')
    if not isinstance(g['weight_decouple'], bool):
        raise ValueError('weight_decouple must be bool')
    if not isinstance(g['fused'], bool):
        raise ValueError('fused must be bool')
    if g['muon_distributed'] not in ('auto', 'error', 'gather', 'gram'):
        raise ValueError('muon_distributed must be auto, error, gather, or gram')
    if type(g['muon_gram_max_dim']) is not int or g['muon_gram_max_dim'] < 1:
        raise ValueError('muon_gram_max_dim must be a positive integer')
    if type(g['muon_gather_max_bytes']) is not int or g['muon_gather_max_bytes'] < 1024:
        raise ValueError('muon_gather_max_bytes must be >= 1024')
    if type(g['muon_ddp_bucket_bytes']) is not int or g['muon_ddp_bucket_bytes'] < 1024:
        raise ValueError('muon_ddp_bucket_bytes must be >= 1024')
    if g['muon_ddp'] not in ('local', 'shard'):
        raise ValueError('muon_ddp must be local or shard')
    if type(g['muon_ddp_min_numel']) is not int or g['muon_ddp_min_numel'] < 1:
        raise ValueError('muon_ddp_min_numel must be a positive integer')
    if g['muon_ddp'] == 'shard' and not g['use_muon']:
        raise ValueError('muon_ddp=shard requires use_muon=True')
    if g['muon_compute_dtype'] not in ('auto', 'float32', 'bfloat16'):
        raise ValueError('muon_compute_dtype must be auto, float32 or bfloat16')
    if g['fused'] and g['use_muon']:
        raise ValueError('fused=True is only supported for standard/Sign-SGDF, not Muon')
    if g['fused'] and g['capturable']:
        raise ValueError('fused=True and capturable=True cannot be combined')


def _step_as_int(step) -> int:
    return int(step.item()) if isinstance(step, torch.Tensor) else int(step)


def _effective_gamma(group: dict) -> float:
    # Paper Sec. 4.2 uses an unscaled gain for Sign-SGDF.
    if group['gamma'] is None:
        return 1.0 if group['use_sign'] else 0.5
    return float(group['gamma'])


def _ensure_step_tensor(state: dict, p: torch.Tensor, capturable: bool) -> torch.Tensor:
    """Use Adam-style scalar tensor steps (CPU normally; device for graphs)."""
    step = state.get('step', 0)
    device = p.device if capturable else torch.device('cpu')
    if isinstance(step, torch.Tensor):
        if step.device == device and step.dtype == torch.float32:
            return step
        step = step.to(device=device, dtype=torch.float32)
    else:
        step = torch.tensor(float(step), dtype=torch.float32, device=device)
    state['step'] = step
    return step


def _gain_power_(gain: torch.Tensor, gamma: float) -> None:
    if gamma == 0.5:
        gain.sqrt_()
    elif gamma != 1.0:
        gain.pow_(gamma)


def _corrections(beta1: float, beta2: float, t):
    """Return 1/(1-beta1**t) and factor in paper's s_hat definition."""
    inv_m = 1.0 / (1.0 - beta1 ** t)
    s_factor = ((1.0 - beta1) * (1.0 - beta1 ** (2 * t)) /
                ((1.0 + beta1) * (1.0 - beta2 ** t)))
    return inv_m, s_factor


class SGDF(Optimizer):
    """PyTorch SGDF, with optional Sign-SGDF or Muon-SGDF update directions.

    Paper Algorithm 1 (elementwise), with t counted per parameter:
      m <- beta1 * m + (1-beta1) * grad
      s <- beta2 * s + (1-beta2) * (grad-m)**2
      m_hat <- m / (1-beta1**t)
      s_hat <- ((1-beta1)*(1-beta1**(2*t)) /
                ((1+beta1)*(1-beta2**t))) * s
      K <- (s_hat / (s_hat + (grad-m_hat)**2 + eps))**gamma
      direction <- m_hat + K*(grad-m_hat)
      p <- p - lr*direction

    Optional mutually exclusive flags transform *only* direction:
      use_sign: direction <- sign(direction)
      use_muon: direction <- Newton-Schulz(direction) for >=2-D weights;
                1-D weights and scalars use normal SGDF.

    Args:
      gamma: None selects 0.5 for standard/Muon SGDF, 1.0 for Sign-SGDF
        (matching the sign extension in Sec. 4.2). Otherwise use the provided value.
      foreach: None (small CUDA batches automatically), True (bounded CUDA/CPU
        tensor-list operations), False (lowest peak temporary allocation).
      foreach_max_bytes: maximum total parameter bytes per foreach batch. Large
        individual tensors use memory-efficient elementwise PyTorch operations.
      capturable: CUDA graph per-tensor path (requires static gradients).
      For DDP, let PyTorch synchronize gradients; no optimizer-side all-reduce.
      Elementwise SGDF works with FSDP local/sharded parameter tensors; Muon
      orthogonalization requires complete matrices and cannot act on shards.
      muon_distributed: 'auto' selects Gram or gather by matrix shape and
        communication size; 'error' rejects shards; 'gather' reconstructs a
        DTensor matrix; 'gram' performs single-reduce polynomial Newton-Schulz
        on a 2-D matrix sharded on a 1-D mesh.
      muon_gram_max_dim: upper bound on the unsharded Gram matrix dimension.
      muon_gather_max_bytes: avoid automatically gathering enormous matrices.
      muon_ddp: 'local' computes Muon on every DDP replica; 'shard' assigns
        expensive complete matrices across ranks and broadcasts directions.
        All ranks must call optimizer.step() in the same order with matching
        parameter groups, synchronized gradients, and equal grad presence.
      muon_ddp_min_numel: avoid broadcasting small Muon matrices.
      muon_ddp_bucket_bytes: maximum coalesced direction broadcast size.
      muon_compute_dtype: 'auto' (bf16 on supported CUDA, fp32 elsewhere),
        'float32' or 'bfloat16' (requires CUDA). fp64 input retains fp64.
      fused: use optional Triton fused CUDA kernel for contiguous fp32
        standard/Sign-SGDF. Explicit opt-in; incompatible configurations error.
      weight_decouple: False = L2 regularization; True = decoupled weight decay.
      maximize: ascend rather than descend the objective.

    Sparse gradients and differentiable optimizer steps are not supported.
    """

    def __init__(
        self, params, lr: float = 0.5, betas=(0.9, 0.999),
        eps: float = 1e-8, gamma: Optional[float] = None, weight_decay: float = 0.0,
        weight_decouple: bool = False, use_sign: bool = False,
        use_muon: bool = False, ns_steps: int = 5,
        foreach: Optional[bool] = None,
        foreach_max_bytes: int = _DEFAULT_FOREACH_BYTES,
        capturable: bool = False, maximize: bool = False,
        fused: bool = False, muon_distributed: str = 'auto',
        muon_gram_max_dim: int = 4096, muon_compute_dtype: str = 'auto',
        muon_gather_max_bytes: int = 256 * 1024 * 1024,
        muon_ddp: str = 'local', muon_ddp_min_numel: int = 65536,
        muon_ddp_bucket_bytes: int = 32 * 1024 * 1024,
    ):
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, gamma=gamma,
                        weight_decay=weight_decay, weight_decouple=weight_decouple,
                        use_sign=use_sign, use_muon=use_muon, ns_steps=ns_steps,
                        foreach=foreach, foreach_max_bytes=foreach_max_bytes,
                        capturable=capturable, maximize=maximize, fused=fused,
                        muon_distributed=muon_distributed,
                        muon_gram_max_dim=muon_gram_max_dim,
                        muon_compute_dtype=muon_compute_dtype,
                        muon_gather_max_bytes=muon_gather_max_bytes,
                        muon_ddp=muon_ddp, muon_ddp_min_numel=muon_ddp_min_numel,
                        muon_ddp_bucket_bytes=muon_ddp_bucket_bytes)
        _validate_group(defaults)
        super().__init__(params, defaults)

    def add_param_group(self, param_group):
        # Optimizer.__init__ calls this method, after self.defaults is initialized.
        if hasattr(self, 'defaults'):
            merged = {**self.defaults, **param_group}
            _validate_group(merged)
        super().add_param_group(param_group)

    def __setstate__(self, state):
        super().__setstate__(state)
        # Allow restoring checkpoints created by earlier SGDF releases.
        for group in self.param_groups:
            for key, default in self.defaults.items():
                group.setdefault(key, default)

    def _init_param(self, p: torch.Tensor, group: dict):
        if p.grad.layout != torch.strided:
            raise RuntimeError('SGDF does not support sparse/non-strided gradients')
        if p.is_complex() or not p.is_floating_point():
            raise TypeError('SGDF expects real floating-point parameters')
        if p.dtype == torch.float16 and group['eps'] < 2 ** -24:
            raise ValueError(
                'SGDF eps underflows to zero with float16 parameters; '
                'use standard AMP with float32 parameters, or eps >= 2**-24'
            )
        if group['use_muon'] and getattr(p, '_is_flat_param', False):
            raise RuntimeError(
                'Muon requires complete weight matrices. FSDP FlatParameter / '
                'sharded flat tensors cannot be orthogonalized independently; '
                'use SGDF(use_muon=False) for FSDP or Muon with DDP.'
            )
        if _is_dtensor(p) and not _is_dtensor(p.grad):
            raise RuntimeError('DTensor parameter requires a DTensor gradient')
        if group['use_muon'] and p.ndim >= 2 and _is_dtensor(p):
            if any(isinstance(place, Partial) for place in p.placements):
                raise RuntimeError('Muon does not support DTensor Partial placements')
            if (group['muon_distributed'] == 'error' and
                    any(isinstance(place, Shard) for place in p.placements)):
                raise RuntimeError(
                    'Muon on sharded DTensor needs muon_distributed="gather" or "gram"; '
                    'local-shard Newton-Schulz is mathematically incorrect.'
                )
        if group['fused']:
            if (p.device.type != 'cuda' or _is_dtensor(p) or p.dtype != torch.float32 or
                    p.grad.dtype != torch.float32 or not p.is_contiguous() or
                    not p.grad.is_contiguous()):
                raise RuntimeError(
                    'fused=True requires contiguous non-DTensor CUDA float32 '
                    'parameters and gradients; use fused=False otherwise'
                )
        if p.grad.device != p.device or p.grad.shape != p.shape:
            raise RuntimeError('Parameter and gradient must share shape and device')
        if group['capturable'] and p.device.type != 'cuda':
            raise ValueError('capturable=True requires CUDA tensors')
        st = self.state[p]
        if not st:
            # Match torch.optim.Adam: moment dtype and device follow parameters.
            st['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            st['exp_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            st['step'] = torch.zeros((), dtype=torch.float32,
                                     device=(p.device if group['capturable'] else 'cpu'))
        else:
            # torch.optim.Optimizer.load_state_dict normally restores state
            # to the current parameter device/dtype, like Adam.
            for key in ('exp_avg', 'exp_var'):
                if key not in st:
                    raise RuntimeError(f'Incomplete SGDF optimizer state: {key} missing')
                if st[key].device != p.device:
                    raise RuntimeError('SGDF state device mismatch; reload state_dict')
        _ensure_step_tensor(st, p, group['capturable'])

    @staticmethod
    def _muon_ddp_owners(group: dict, ps: list[torch.Tensor]) -> dict[int, int]:
        """Assign high-cost 2-D+ Muon updates to single-node DDP ranks."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return {}
        if not group['params']:
            return {}
        world = dist.get_world_size()
        if world <= 1:
            return {}
        if any(_is_dtensor(p) for p in group['params']):
            raise RuntimeError('muon_ddp="shard" requires replicated local tensors; '
                               'for FSDP2/DTensor use muon_distributed="auto"')
        if len({p.device for p in group['params']}) != 1:
            raise RuntimeError('muon_ddp="shard" requires parameters on one device per rank')

        # Catch mismatched gradient availability before entering a per-layer
        # broadcast, which would otherwise risk a multi-rank deadlock.
        backend = str(dist.get_backend()).lower()
        first = ps[0] if ps else group['params'][0]
        dev = first.device if 'nccl' in backend else torch.device('cpu')
        num_params = len(group['params'])
        local_count = torch.tensor([num_params], device=dev, dtype=torch.int64)
        counts = [torch.empty_like(local_count) for _ in range(world)]
        dist.all_gather(counts, local_count)
        if any(int(v.item()) != num_params for v in counts):
            raise RuntimeError('Muon DDP shards require identical parameter groups on all ranks')
        flags = torch.tensor([int(p.grad is not None) for p in group['params']],
                             dtype=torch.uint8, device=dev)
        flags_all = [torch.empty_like(flags) for _ in range(world)]
        dist.all_gather(flags_all, flags)
        if any(not torch.equal(v, flags) for v in flags_all):
            raise RuntimeError('Muon DDP shards require matching gradient presence '
                               'on every rank; synchronize gradients before step')

        # Greedy balancing by approximate Newton-Schulz work, not by layer
        # count. The sorted shapes are identical on all DDP replicas.
        jobs = []
        for i, p in enumerate(group['params']):
            if (p.grad is None or p.ndim < 2 or
                    p.numel() < group['muon_ddp_min_numel']):
                continue
            rows, cols = p.shape[0], p.numel() // p.shape[0]
            lo, hi = min(rows, cols), max(rows, cols)
            jobs.append((hi * lo * lo, i, p))
        jobs.sort(key=lambda item: (-item[0], item[1]))
        loads = [0] * world
        owners = {}
        for cost, _, p in jobs:
            owner = min(range(world), key=lambda idx: loads[idx])
            owners[id(p)] = owner
            loads[owner] += cost
        return owners

    @staticmethod
    def _finish_muon_ddp(
        pending: list[tuple[torch.Tensor, int, Optional[torch.Tensor], float]],
        group: dict,
    ) -> None:
        """Broadcast all shard owners' ready updates in bounded flat buckets.

        Newton-Schulz work has already run concurrently on different ranks.
        Packing updates by owner and dtype avoids one collective per layer.
        """
        import torch.distributed as dist

        if not pending:
            return
        rank = dist.get_rank()
        world = dist.get_world_size()
        max_bytes = group['muon_ddp_bucket_bytes']
        for owner in range(world):
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                items = [entry for entry in pending
                         if entry[1] == owner and entry[0].dtype == dtype]
                if not items:
                    continue
                buckets = []
                bucket, used = [], 0
                for entry in items:
                    size = entry[0].numel() * entry[0].element_size()
                    if bucket and used + size > max_bytes:
                        buckets.append(bucket)
                        bucket, used = [], 0
                    bucket.append(entry)
                    used += size
                if bucket:
                    buckets.append(bucket)
                for batch in buckets:
                    count = sum(p.numel() for p, _, _, _ in batch)
                    if rank == owner:
                        flat = torch.cat([direction.reshape(-1)
                                          for _, _, direction, _ in batch])
                    else:
                        flat = torch.empty(count, dtype=dtype,
                                           device=batch[0][0].device)
                    dist.broadcast(flat, src=owner)
                    offset = 0
                    for p, _, _, lr in batch:
                        num = p.numel()
                        p.add_(flat[offset:offset + num].reshape(p.shape), alpha=-lr)
                        offset += num

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            # Legacy SGDF checkpoint groups may lack newer options.
            for key, default in self.defaults.items():
                group.setdefault(key, default)
            _validate_group(group)
            ps = [p for p in group['params'] if p.grad is not None]
            owners = (self._muon_ddp_owners(group, ps)
                      if group['muon_ddp'] == 'shard' else {})
            if not ps:
                continue
            for p in ps:
                self._init_param(p, group)

            if group['fused']:
                for p in ps:
                    self._step_fused(p, group)
                continue

            # Foreach can consume O(num_params) extra memory. Group by state
            # step and use fixed-byte chunks; large matrices use the loop.
            want_foreach = (group['foreach'] is True or
                            (group['foreach'] is None and ps[0].device.type == 'cuda'))
            if (not want_foreach or group['capturable'] or group['use_muon'] or
                    not all(hasattr(torch, op) for op in (
                        '_foreach_add_', '_foreach_addcmul_', '_foreach_mul_',
                        '_foreach_sub', '_foreach_reciprocal_', '_foreach_mul')) or
                    (group['use_sign'] and not hasattr(torch, '_foreach_sign_'))):
                deferred = []
                for p in ps:
                    self._step_one(p, group, muon_owner=owners.get(id(p)),
                                   muon_deferred=deferred)
                if deferred:
                    self._finish_muon_ddp(deferred, group)
                continue

            buckets = defaultdict(list)
            max_bytes = group['foreach_max_bytes']
            for p in ps:
                st = self.state[p]
                eligible = (
                    not _is_dtensor(p) and not _is_dtensor(p.grad)
                    and p.is_contiguous() and p.grad.is_contiguous()
                    and p.dtype in (torch.float32, torch.float64)
                    and st['exp_avg'].dtype == p.dtype
                    and st['exp_var'].dtype == p.dtype
                    and st['step'].device.type == 'cpu'
                    and p.numel() * p.element_size() <= max_bytes
                )
                if not eligible:
                    self._step_one(p, group)
                    continue
                key = (p.device, p.dtype, _step_as_int(st['step']))
                buckets[key].append(p)

            for bucket in buckets.values():
                batch, batch_bytes = [], 0
                for p in bucket:
                    b = p.numel() * p.element_size()
                    if batch and batch_bytes + b > max_bytes:
                        self._step_foreach(batch, group)
                        batch, batch_bytes = [], 0
                    batch.append(p)
                    batch_bytes += b
                if batch:
                    self._step_foreach(batch, group)
        return loss

    def _step_one(self, p: torch.Tensor, group: dict,
                  muon_owner: Optional[int] = None,
                  muon_deferred: Optional[list] = None) -> None:
        b1, b2 = group['betas']
        lr, eps, gamma = group['lr'], group['eps'], _effective_gamma(group)
        wd, decouple = group['weight_decay'], group['weight_decouple']
        st = self.state[p]
        m, v = st['exp_avg'], st['exp_var']
        grad = p.grad.detach()
        if grad.dtype != m.dtype:
            grad = grad.to(dtype=m.dtype)
        if group['maximize']:
            grad = grad.neg()
        if wd and not decouple:
            # Creates working gradient; NEVER modifies p.grad.
            grad = grad.add(p.detach().to(dtype=m.dtype), alpha=wd)
        elif wd and decouple:
            p.mul_(1 - lr * wd)

        m.mul_(b1).add_(grad, alpha=1 - b1)
        # r: first (grad-m), then reuse SAME buffer as (grad-m_hat).
        r = grad - m
        v.mul_(b2).addcmul_(r, r, value=1 - b2)

        if group['capturable']:
            counter = _ensure_step_tensor(st, p, capturable=True)
            counter.add_(1)
            t = counter
        else:
            counter = _ensure_step_tensor(st, p, capturable=False)
            counter.add_(1)
            t = _step_as_int(counter)
        inv_m, s_factor = _corrections(b1, b2, t)
        r.add_(m, alpha=1.0 - inv_m) if isinstance(inv_m, float) else r.add_(m * (1 - inv_m))

        # EXACT paper formula, written in two full-size temporaries (r, K):
        #   K = (s_hat / (s_hat + r*r + eps))**gamma
        #   s_hat = s_factor * v
        # eps is added to the DENOMINATOR, never outside K.
        # Match torch.optim.Adam's ordinary tensor-dtype arithmetic: no
        # special fp32 promotion for half-precision parameters. As in Adam,
        # eps=1e-8 may round to zero when the denominator is float16.
        k = r.square()
        if isinstance(s_factor, float):
            k.add_(v, alpha=s_factor).add_(eps)
            torch.div(v, k, out=k).mul_(s_factor)
        else:  # CUDA-graph device scalar
            k.add_(v * s_factor).add_(eps)
            torch.div(v, k, out=k).mul_(s_factor)
        _gain_power_(k, gamma)

        if group['use_sign'] or (group['use_muon'] and p.ndim >= 2):
            k.mul_(r)
            if isinstance(inv_m, float):
                k.add_(m, alpha=inv_m)
            else:
                k.add_(m * inv_m)
            if group['use_sign']:
                k.sign_()
                direction = k
            else:
                from .muon import muon_postprocess
                if muon_owner is None:
                    direction = muon_postprocess(
                        k, ns_steps=group['ns_steps'],
                        distributed=group['muon_distributed'],
                        gram_max_dim=group['muon_gram_max_dim'],
                        compute_dtype=group['muon_compute_dtype'],
                        gather_max_bytes=group['muon_gather_max_bytes'],
                    )
                else:
                    import torch.distributed as dist
                    if dist.get_rank() == muon_owner:
                        direction = muon_postprocess(
                            k, ns_steps=group['ns_steps'],
                            distributed='error',
                            compute_dtype=group['muon_compute_dtype'],
                        ).to(dtype=p.dtype).contiguous()
                    else:
                        direction = None
                    if muon_deferred is not None:
                        muon_deferred.append((p, muon_owner, direction, lr))
                        return
                    if direction is None:
                        direction = torch.empty(p.shape, dtype=p.dtype, device=p.device)
                    dist.broadcast(direction, src=muon_owner)
            p.add_(direction.to(dtype=p.dtype), alpha=-lr)
        elif p.dtype == m.dtype and isinstance(inv_m, float):
            # No third full-size update tensor.
            p.add_(m, alpha=-lr * inv_m)
            p.addcmul_(k, r, value=-lr)
        else:
            # CUDA graph capture and dtype-mismatched updates use a temporary direction.
            k.mul_(r)
            if isinstance(inv_m, float):
                k.add_(m, alpha=inv_m)
            else:
                k.add_(m * inv_m)
            p.add_(k.to(dtype=p.dtype), alpha=-lr)

    def _step_foreach(self, ps: list[torch.Tensor], group: dict) -> None:
        b1, b2 = group['betas']
        lr, eps, gamma = group['lr'], group['eps'], _effective_gamma(group)
        wd, decouple = group['weight_decay'], group['weight_decouple']
        ms = [self.state[p]['exp_avg'] for p in ps]
        vs = [self.state[p]['exp_var'] for p in ps]
        gs = [p.grad.detach() for p in ps]
        if group['maximize']:
            gs = torch._foreach_neg(gs)
        if wd:
            if decouple:
                torch._foreach_mul_(ps, 1 - lr * wd)
            else:
                gs = torch._foreach_add(gs, ps, alpha=wd)

        torch._foreach_mul_(ms, b1)
        torch._foreach_add_(ms, gs, alpha=1 - b1)
        r = torch._foreach_sub(gs, ms)
        torch._foreach_mul_(vs, b2)
        torch._foreach_addcmul_(vs, r, r, value=1 - b2)
        t = _step_as_int(self.state[ps[0]]['step']) + 1
        for p in ps:
            self.state[p]['step'].add_(1)
        inv_m, s_factor = _corrections(b1, b2, t)
        torch._foreach_add_(r, ms, alpha=1 - inv_m)
        # Epsilon is added only to the gain denominator.
        gains = torch._foreach_mul(r, r)
        torch._foreach_add_(gains, vs, alpha=s_factor)
        torch._foreach_add_(gains, eps)
        torch._foreach_reciprocal_(gains)
        torch._foreach_mul_(gains, vs)
        torch._foreach_mul_(gains, s_factor)
        if gamma == 0.5:
            torch._foreach_sqrt_(gains)
        elif gamma != 1.0:
            torch._foreach_pow_(gains, gamma)
        if group['use_sign']:
            torch._foreach_mul_(gains, r)
            torch._foreach_add_(gains, ms, alpha=inv_m)
            torch._foreach_sign_(gains)
            torch._foreach_add_(ps, gains, alpha=-lr)
        else:
            # In-place 2-term update: no direction tensors allocated.
            torch._foreach_add_(ps, ms, alpha=-lr * inv_m)
            torch._foreach_addcmul_(ps, gains, r, value=-lr)


    def _step_fused(self, p: torch.Tensor, group: dict) -> None:
        """Optional fused fp32 CUDA path; Triton is imported only on demand.

        Persistent moment buffers remain identical to v0.4.1. The kernel merges
        every elementwise SGDF operation into a single launch per parameter.
        The step counter is CPU-hosted to avoid CUDA synchronization; this
        implementation is intentionally not CUDA-graph-capturable.
        """
        try:
            from ._triton import fused_sgdf_step
        except ImportError as exc:
            raise RuntimeError(
                'fused=True requires a CUDA-enabled PyTorch and Triton. '
                'Install SGDF[triton] or set fused=False.'
            ) from exc
        state = self.state[p]
        counter = _ensure_step_tensor(state, p, capturable=False)
        counter.add_(1)
        t = _step_as_int(counter)
        b1, b2 = group['betas']
        inv_m, s_factor = _corrections(b1, b2, t)
        fused_sgdf_step(
            p, p.grad.detach(), state['exp_avg'], state['exp_var'],
            lr=group['lr'], beta1=b1, beta2=b2, eps=group['eps'],
            gamma=_effective_gamma(group), wd=group['weight_decay'],
            decouple=group['weight_decouple'], maximize=group['maximize'],
            sign=group['use_sign'], inv_m=inv_m, s_factor=s_factor,
        )
