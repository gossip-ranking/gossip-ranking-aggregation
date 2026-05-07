# Decentralized Ranking Aggregation via Gossip: Convergence and Robustness

Experiments for decentralized rank aggregation with gossip protocols.

This repository studies how classical consensus rules (Borda, Copeland, Footrule)
can be computed in decentralized networks where agents repeatedly average local
information over graph edges. It also includes synthetic and real-data experiments,
contamination robustness studies, and plotting utilities.

## What is included

- Centralized consensus utilities for Borda, Copeland, and Footrule.
- Decentralized gossip implementations (Numba-accelerated for Borda/Copeland).
- Footrule decentralized updates via asynchronous ADMM [Robust Distributed Learning under Resource Constraints: Decentralized Quantile Estimation via (Asynchronous) ADMM](https://arxiv.org/abs/2601.20571).
- Data loaders for PrefLib datasets (Sushi, Sushi-100, T-shirts, Netflix, Debian).
- Synthetic Mallows data generation and contamination experiments.
- Plotting helpers for convergence and method comparison figures.

## Repository layout

- `numba_gossip.py`: fast decentralized Borda/Copeland trial loops.
- `methods.py`: method classes (including decentralized Footrule).
- `synthetic_mallows.py`: synthetic Mallows benchmark driver.
- `dataset.py`: study of decentralized consensus methods on real datasets (Sushi, Debian, etc.).
- `create_contamination_table.py`: contamination robustness experiment on manipulated Mallows distribution.

### Utilities (`utils/`)

- `consensus.py`: centralized consensus methods (Borda count, Copeland, Footrule)
  and local Kemenization refinement.
- `data.py`: Mallows model sampling and contamination utilities. Supports
  deterministic, inverse, and random contamination modes.
- `load_data.py`: PrefLib dataset loaders with deterministic sampling and optional
  corruption (Sushi, Sushi-100, T-shirts, Netflix, Debian).
- `graph.py`: network graph generation (Complete, Watts-Strogatz, 2D Grid, Cycle,
  Geometric random graphs). Ensures connectedness.
- `asyladmm.py`: asynchronous ADMM solver for decentralized Footrule consensus
  on network topologies.
- `helper.py`: conversion utilities (rankings ↔ instance formats, dicts ↔ arrays).
- `generate_csv.py`: export rankings to CSV format for external aggregators like
  `pyflagr`.
- `plot.py`: convergence curve and multi-graph result plotting with error bars.
- `metric.py`: distance metrics (Kendall tau via PrefLib).

## Installation

### 1) Create an environment

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```
pip install -r requirements.txt
```

## Data

Datasets are fetched from [PrefLib](https://preflib.github.io/PrefLib-Jekyll/) URLs at runtime via `utils/load_data.py`.
No manual dataset download is required for the built-in loaders.

Available loaders include:

- `load_sushi`
- `load_debian`

For synthetic Mallows model, please refer to `generate_mallows.py`.


## Notes

- For reproducibility, several scripts set seeds explicitly.
- Long experiments can be CPU intensive; reduce `iterations` and `n_trials` for
  quick sanity checks.
