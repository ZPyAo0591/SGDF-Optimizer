# SGDF Usage Guide

SGDF provides three update modes through a single PyTorch optimizer class:
standard **SGDF**, **Sign-SGDF**, and **Muon + SGDF**.

## Installation

Install from the project directory:

```bash
python -m pip install .
```

Or install the prebuilt wheel:

```bash
python -m pip install sgdf-0.5.0-py3-none-any.whl
```

Requires Python 3.9+ and PyTorch 2.0+.

## 1. Standard SGDF

Replace an existing PyTorch optimizer with `SGDF`:

```python
from SGDF import SGDF

model = ...
optimizer = SGDF(model.parameters(), lr=0.1)

for inputs, targets in dataloader:
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
```

The example assumes that `model`, `dataloader`, and `criterion` have already been defined.

## 2. Sign-SGDF

```python
optimizer = SGDF(model.parameters(), lr=1e-3, use_sign=True)
```

`use_sign=True` applies `sign()` to SGDF's filtered gradient direction before the update.
The default gain exponent in this mode is `gamma=1.0`.

## 3. Muon + SGDF

```python
optimizer = SGDF(model.parameters(), lr=0.01, use_muon=True, ns_steps=5)
```

Muon applies Newton–Schulz orthogonalization to matrix-like (`ndim >= 2`)
parameter update directions. One-dimensional parameters and scalars use the
standard SGDF update.

You can choose Muon only for specific parameter groups:

```python
matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
other_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

optimizer = SGDF([
    {"params": matrix_params, "use_muon": True, "lr": 0.01},
    {"params": other_params, "use_muon": False, "lr": 0.001},
])
```

Each trainable parameter must appear in at most one group.
If your model includes embeddings, output heads, or other matrices that should
not use Muon, place them in the standard SGDF group explicitly.

`use_muon=True` and `use_sign=True` cannot be enabled in the same group.

## 4. GPU fused updates (Triton)

Install the optional Triton dependency:

```bash
python -m pip install '.[triton]'
```

Or install Triton separately when using the wheel. Enable the fused update:

```python
optimizer = SGDF(model.parameters(), lr=0.1, fused=True)
```

Fused updates require contiguous CUDA `float32` parameters and gradients.
The path is available for standard SGDF and Sign-SGDF, not Muon.
Do not set `capturable=True` together with `fused=True`.

## 5. AMP / mixed precision

Keep the model parameters in float32, use autocast for forward computation,
and let GradScaler call the optimizer step:

```python
from SGDF import SGDF
import torch

optimizer = SGDF(model.parameters(), lr=0.1)
scaler = torch.cuda.amp.GradScaler()

optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(inputs)
    loss = criterion(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 6. Single-node multi-GPU training

Launch one process per GPU with `torchrun` and initialize `torch.distributed`
with `backend="nccl"`. On CPU, Gloo is available for development.

### DDP (replicated parameters)

Normal DDP already synchronizes gradients; Muon can run on each replica:

```python
optimizer = SGDF(ddp_model.parameters(), lr=0.01, use_muon=True)
```

For larger matrix parameters, distribute Newton–Schulz computation among
DDP ranks on the same node and broadcast coalesced update directions:

```python
optimizer = SGDF(
    ddp_model.parameters(),
    lr=0.01,
    use_muon=True,
    muon_ddp="shard",
    muon_ddp_min_numel=65536,
    muon_ddp_bucket_bytes=32 * 1024 * 1024,
)
```

`muon_ddp="local"` is the default and avoids optimizer-side broadcasts.
`shard` balances large orthogonalizations across ranks by approximate matrix
cost; small matrices are computed locally on every rank. All ranks must have
matching parameters and step in the same order, with synchronized gradients.
For very small models, local computation can be faster than broadcasting.

### FSDP2 / sharded DTensor matrices

FSDP2 preserves parameter shapes as DTensors, unlike FSDP1's flattened
parameter storage. `muon_distributed="auto"` (the default) selects one of:

- `gram`: for a 2-D parameter sharded along one axis on a 1-D DeviceMesh,
  compute one global Gram matrix and run the remaining Newton–Schulz
  iterations locally; full weight all-gather is not required.
- `gather`: materialize the full matrix on every participating rank before
  orthogonalizing, then return the original DTensor placements.

```python
optimizer = SGDF(
    model.parameters(),
    lr=0.01,
    use_muon=True,
    muon_distributed="auto",
    muon_gram_max_dim=4096,
    muon_gather_max_bytes=256 * 1024 * 1024,
)
```

For supported one-dimensional DTensor meshes, you may force either
`muon_distributed="gram"` or `muon_distributed="gather"`. Setting
`muon_distributed="error"` rejects sharded matrices instead.

Only matrix-like parameters use Muon; scalars and vectors use standard SGDF.
Embedding and output-head matrices are best placed in a separate standard
SGDF group when your training recipe excludes them from Muon.

All participating ranks must enter collectives in the same order. FSDP1
FlatParameter shards cannot be safely orthogonalized by this optimizer.
The DDP `muon_ddp="shard"` option is for replicated local parameters and
cannot be combined with DTensor sharding for the same parameter group.

## 7. Weight decay, foreach, and schedulers

```python
optimizer = SGDF(
    model.parameters(),
    lr=0.1,
    weight_decay=0.01,
    weight_decouple=True,
    foreach=True,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

`weight_decouple=True` selects decoupled weight decay; the default is L2
regularization when `weight_decay` is nonzero.

## 8. Save and restore an optimizer checkpoint

```python
# Save
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint.pt")

# Restore after rebuilding the model and optimizer
checkpoint = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

After restoring an LR scheduler, follow the usual PyTorch optimizer/scheduler
load ordering to preserve the learning rates from the checkpoint.

## 9. Main constructor parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `lr` | `0.5` | Learning rate |
| `betas` | `(0.9, 0.999)` | Momentum and variance decay |
| `eps` | `1e-8` | Numerical stabilization |
| `gamma` | `None` | Gain exponent: 0.5 normally; 1.0 for Sign-SGDF |
| `weight_decay` | `0.0` | Weight decay coefficient |
| `weight_decouple` | `False` | Use decoupled decay when enabled |
| `use_sign` | `False` | Sign-SGDF update |
| `use_muon` | `False` | Muon postprocessing |
| `ns_steps` | `5` | Newton–Schulz iterations |
| `foreach` | `None` | Use grouped tensor operations when eligible |
| `foreach_max_bytes` | `16777216` | Maximum bytes per foreach batch |
| `fused` | `False` | Triton fused GPU path |
| `capturable` | `False` | CUDA graph-oriented step path |
| `maximize` | `False` | Ascend rather than descend |
| `muon_distributed` | `"auto"` | `auto`, `error`, `gather`, or `gram` |
| `muon_gram_max_dim` | `4096` | Maximum Gram dimension |
| `muon_gather_max_bytes` | `268435456` | Automatic gather memory cap |
| `muon_ddp` | `"local"` | `local` or `shard` for replicated DDP |
| `muon_ddp_min_numel` | `65536` | Minimum matrix size for DDP sharding |
| `muon_ddp_bucket_bytes` | `33554432` | Coalesced direction broadcast target |
| `muon_compute_dtype` | `"auto"` | `auto`, `float32`, `bfloat16` |

Sparse gradients and differentiable optimizer steps are not supported.
