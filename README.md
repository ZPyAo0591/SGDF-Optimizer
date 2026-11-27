
# SGDF Optimizer

[![PyPI](https://img.shields.io/pypi/v/SGDF)](https://pypi.org/project/SGDF/)
[![Downloads](https://img.shields.io/pypi/dm/SGDF)](https://pypi.org/project/SGDF/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official Python package of SGDF, providing three
update modes in a unified PyTorch optimizer interface:
**SGDF**, **Sign-SGDF**, and **Muon + SGDF**.

For the theoretical foundations, experimental results,
and citations, please refer to our
[CVPR 2026 paper](https://arxiv.org/abs/2603.06120)
and the [main branch](https://github.com/ZPyAo0591/SGDF-Optimizer/tree/main).

## Installation

```bash
pip install SGDF
```

With optional Triton support:

```bash
pip install "SGDF[triton]"
```

Python 3.9+ and PyTorch 2.0+ are required.


## Basic usage

```python
from SGDF import SGDF

optimizer = SGDF(model.parameters(), lr=0.1)                    # SGDF
optimizer = SGDF(model.parameters(), lr=1e-3, use_sign=True)   # Sign-SGDF
optimizer = SGDF(model.parameters(), lr=0.01, use_muon=True)   # Muon + SGDF
```

`use_sign` and `use_muon` are mutually exclusive. SGDF works with standard PyTorch parameter groups, learning-rate schedulers, AMP, and optimizer checkpoints (`state_dict`).

When using Muon, matrix-like parameters are orthogonalized with Newton–Schulz; vectors and scalars receive the standard SGDF update. A common configuration is:

```python
optimizer = SGDF([
    {"params": matrix_params, "use_muon": True, "lr": 0.01},
    {"params": other_params,  "use_muon": False, "lr": 0.001},
])
```

The two parameter lists must not overlap.

For detailed examples of optimizer modes, parameter groups, AMP, distributed Muon,
checkpointing, and options, see [USAGE.md](USAGE.md).

## GPU fused update

Use `fused=True` with contiguous CUDA FP32 parameters and gradients, after installing the optional Triton dependency:

```python
optimizer = SGDF(model.parameters(), lr=0.1, fused=True)
```

The fused path supports standard SGDF and Sign-SGDF. Muon uses the non-fused path.

## Single-node multi-GPU Muon

For ordinary DDP, the optimizer uses synchronized gradients just like other
PyTorch optimizers. If Muon matrices are large, `muon_ddp="shard"` balances
Newton–Schulz computations across local DDP replicas and broadcasts directions
in bounded buckets:

```python
optimizer = SGDF(ddp_model.parameters(), lr=0.01, use_muon=True,
                 muon_ddp="shard")
```

For FSDP2 / DTensor matrix sharding, `muon_distributed="auto"` is the default:

```python
optimizer = SGDF(model.parameters(), lr=0.01, use_muon=True,
                 muon_distributed="auto")
```

Auto mode chooses between reconstructing the whole matrix (`gather`) and
computing a global Gram matrix (`gram`) once, followed by local polynomial
Newton–Schulz iterations. It respects `muon_gram_max_dim` and
`muon_gather_max_bytes`. To select a specific strategy, set
`muon_distributed="gram"` or `"gather"`.

These modes target a single node with multiple GPUs. All ranks must execute
collectives in the same order. FSDP1 flattened parameter shards are unsupported
for Muon orthogonalization.

## Configuration

Main options include `lr`, `betas`, `eps`, `gamma`, `weight_decay`, `weight_decouple`, `maximize`, `use_sign`, `use_muon`, `foreach`, `foreach_max_bytes`, `fused`, `capturable`, `ns_steps`, `muon_distributed`, `muon_gram_max_dim`, `muon_gather_max_bytes`, `muon_compute_dtype`, `muon_ddp`, `muon_ddp_min_numel`, and `muon_ddp_bucket_bytes`.

The default `gamma=None` selects 0.5 for standard/Muon SGDF and 1.0 for Sign-SGDF. Sparse gradients and differentiable optimizer steps are not supported.

License: MIT.
