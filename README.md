# ProFlow: Zero-Shot Physics-Consistent Sampling via Proximal Flow Guidance

Official implementation of **ProFlow**, a proximal guidance framework for zero-shot physics-consistent sampling: inferring physical fields from sparse observations with a *fixed* pretrained flow-based generative prior, without any task-specific retraining.

## Overview

Inferring physical fields from sparse observations while respecting the governing partial differential equations (PDEs) is a fundamental challenge in computational physics. ProFlow addresses this with a rigorous two-step sampling scheme that alternates between:

1. **Terminal optimization** — a proximal minimization step that drives the flow's terminal prediction toward the intersection of the physically consistent set and the observationally consistent set;
2. **Interpolation** — a step that maps the refined state back onto the generative trajectory, keeping the sample consistent with the learned flow probability path.

The procedure admits a Bayesian interpretation as a sequence of local maximum a posteriori (MAP) updates. On Poisson, Helmholtz, Darcy, and viscous Burgers' benchmarks, ProFlow achieves superior physical and observational consistency, as well as more accurate distributional statistics, compared to state-of-the-art diffusion- and flow-based baselines (ECI, DiffusionPDE, D-Flow, PCFM, etc.).

## Installation

```bash
conda create -n proflow python=3.10
conda activate proflow
pip install torch matplotlib tqdm numpy scipy
# optional, for the diffusers-based baselines
pip install diffusers
```

## Usage

### 1. Train the flow prior

Select the task in the corresponding config (e.g. `config/poisson_config.py`, choosing `rectified_type`, `condition_type`, etc.), then:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

Checkpoints are written to `saved/<task>-iter<...>/`.

### 2. Zero-shot physics-consistent sampling

With a pretrained prior, run evaluation with the desired sampler (ProFlow or a baseline):

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py
```

Supported tasks include forward problems (Poisson, Helmholtz, Darcy), joint reconstruction from 50% random observations (Darcy), and sparse-time observation for viscous Burgers'.

### 3. Benchmarks

```bash
python benchmark_runtime.py       # runtime comparison across samplers
python benchmark_sensitivity.py   # sensitivity to K, lambda_obs, lambda_pde, sampling steps
```

Results (RE, PDE error, wall-clock time) are aggregated as mean ± std and exported to CSV.

## Key Hyperparameters

| Parameter | Meaning | Default |
|---|---|---|
| `terminal_iters` (K) | inner proximal optimization steps | 3 (elliptic) / 1 (Burgers) |
| `lambda_obs` | observation-consistency weight / step size | 100 (elliptic) / 14 (Burgers) |
| `lambda_pde` | physics-consistency weight / step size | 1e-3 (elliptic) / 0.1 (Burgers) |
| `sampling_steps` (N) | number of flow integration steps | 100 |

## Citation

If you find this work useful, please cite:

```bibtex
@article{yu2026proflow,
  title   = {ProFlow: Zero-Shot Physics-Consistent Sampling via Proximal Flow Guidance},
  author  = {Yu, Zichao and Li, Ming and Zhang, Wenyi and Zou, Difan and Gao, Weiguo},
  year    = {2026}
}
```
