# Decentralized Ranking Aggregation via Gossip: Convergence and Robustness

Experiments for decentralized rank aggregation with gossip protocols.

This repository studies how classical consensus rules (Borda, Copeland, Footrule)
can be computed in decentralized networks where agents repeatedly average local
information over graph edges. It also includes synthetic and real-data experiments,
contamination robustness studies, and plotting utilities.

## What is included

- Gossip-based consensus algorithms: Decentralized implementations of Borda, Copeland, and Footrule methods that work by averaging local information over network edges.
- Numba-accelerated gossip implementations for fast simulation of Borda/Copeland consensus over thousands of iterations.
- Robustness analysis: Contamination/adversarial attack simulations to study algorithm breakdown points.
- Multiple graph topologies: Complete graphs, Watts-Strogatz, 2D grids, and geometric random graphs.
- Synthetic data generation: Mallows model sampling (single and mixture distributions) for controlled experiments.
- Real dataset support: PrefLib dataset loaders (Sushi, Debian) for empirical evaluation.
- Benchmarking utilities: Comparison with centralized methods via pyflagr integration.
- Visualization: Convergence curves and multi-method comparison plots with error bars.

## Repository layout

### Main Experiment Scripts
- `convergence_experiments.py`: Runs convergence trials for Borda and Copeland gossip algorithms on different graph topologies. Tests on synthetic Mallows mixtures and real datasets (Debian, Sushi). Saves results to pickle files for further analysis.
- `breakdown_experiments.py`: Studies robustness of consensus algorithms under contamination/adversarial attacks. Varies contamination level (epsilon) and tracks consensus error over iterations. Parallel execution via multiprocessing.
- `breakdown_consensus.py`: Core functions for breakdown analysis derived from `numba_gossip.py`. Implements averaging-based consensus error computation for Borda and Copeland methods under contamination scenarios.
- `footrule_experiments.py`: Benchmarks decentralized Footrule method against Borda and Copeland on multiple graph topologies. Supports parallel trial execution with configurable graph types.
- `local_kemenization_experiments.py`: Evaluates centralized aggregation methods (Borda, Copeland, Condorcet, RRA, Kemeny) using pyflagr library on synthetic Mallows data.

### Utilities (`utils/`)
- `consensus.py`: Centralized consensus methods (Borda count, Copeland, Footrule) and local Kemenization refinement.
- `mallows.py`: Mallows model sampling for synthetic preference data. Supports single distributions and mixtures.
- `load_data.py`: PrefLib dataset loaders with deterministic sampling (Sushi, Debian, etc.).
- `graph.py`: Network graph generation (Complete, Watts-Strogatz, 2D Grid, Geometric) with connectedness guarantees.
- `asyladmm.py`: Asynchronous ADMM solver for decentralized Footrule consensus on network topologies.
- `numba_gossip.py`: Numba-accelerated gossip algorithm implementations for Borda/Copeland consensus. Computes score errors and consensus errors over iterations.
- `helper.py`: Conversion utilities (rankings ↔ array formats, dicts ↔ arrays).
- `generate_csv.py`: Export rankings to CSV format for external aggregators like pyflagr.
- `plot.py`: Convergence curve and multi-graph result plotting with error bars.
- `results.py`: Utilities for saving/loading pickle result files.
- `attacks.py`: Contamination/attack utilities for generating corrupted preference datasets.


## Installation

```
pip install -r requirements.txt
```

## Data
Datasets are fetched from [PrefLib](https://preflib.github.io/PrefLib-Jekyll/) URLs at runtime via `utils/load_data.py`.
No manual dataset download is required for the built-in loaders.

Available loaders include:

- `load_sushi`: Sushi preference dataset
- `load_debian`: Debian package voting dataset

For synthetic Mallows model, use `utils/mallows.py`:
- `generate_mallows()`: Single Mallows distribution
- `generate_mallows_mixture()`: Mixture of Mallows distributions

## Running Experiments
### Convergence Analysis

```
python convergence_experiments.py
```
Tests Borda and Copeland convergence on synthetic and real datasets across different graph topologies.

### Robustness / Breakdown Analysis

```
python breakdown_experiments.py --n 1000 3000 --m 7 --phi 0.6 --graph-type "Watts-Strogatz" --n-trials 500
```
Studies algorithm robustness under contamination attacks. Key arguments:
- `--n`: Network size(s)
- `--m`: Number of items to rank
- `--phi`: Mallows concentration parameter
- `--graph-type`: Network topology
- `--checkpoints`: Iteration checkpoints for analysis
- `--workers`: Number of parallel workers

### Decentralized Footrule
```
python footrule_experiments.py
```
Benchmarks the decentralized Footrule method against Borda and Copeland.

### Centralized Method Comparison

```
python local_kemenization_experiments.py
```
Compares Borda, Copeland, Condorcet, RRA, and Kemeny methods on synthetic data.


