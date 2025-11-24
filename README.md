# Three-Channel Correlation Algorithm - Code and Data

This repository contains the code and data for the paper:
**"Two-Path Coefficient of Variation: A Novel Quality Criterion for Three-Channel Self-Noise Estimation"**

## Quick Start

### Generate Figures (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate all figures from existing data
python gen_fig.py
```

Output figures will be saved in `output/{timestamp}/` directory.

### Regenerate Experimental Data (Optional)

If you want to regenerate the experimental data from scratch:

```bash
# Generate all experiments
python gen_data.py

# Generate specific experiment
python gen_data.py --exam E01-algorithm_verification

# List available experiments
python gen_data.py --list
```

**Note**: Data generation can take time (E04: ~10-15 minutes for 100 condition combinations).

## Generated Figures

| Figure | Filename | Description |
|--------|----------|-------------|
| Fig. 1a | E01_figure0_time_domain.png | Time-domain waveforms of three channels |
| Fig. 1b | E01_figure1b_frequency_processing.png | Frequency-domain processing illustration |
| Fig. 3 | figure3_sync_error_time_domain.png | Time-domain comparison under sync errors |
| Fig. 4 | figure4_tpcv_sync_error.png | TPCV vs time synchronization error |
| Fig. 7 | figure7_tpcv_interaction_heatmap.png | TPCV interaction heatmap |
| Fig. 8 | figure8_main_effects.png | Main effects analysis |
| Fig. 11c | figure11c_error_vs_tpcv_scatter.png | Estimation error vs TPCV scatter plot |
| Fig. 11c+ | figure11c_error_vs_tpcv_scatter_with_cave.png | Scatter plot with cave experiment data |

## Directory Structure

```
code_release/
├── README.md                  # This file
├── gen_fig.py                 # Entry point: generate figures
├── gen_data.py                # Entry point: generate experimental data
├── verify_all_data.py         # Data verification utility
├── requirements.txt           # Python dependencies
│
├── code/                      # Core algorithms and visualization
│   ├── core_algorithm.py      # Three-channel correlation algorithm
│   ├── common_sync_algorithm.py  # Synchronization utilities
│   ├── single_experiment.py   # Single experiment runner
│   ├── noise_gradient_analysis.py  # Gradient analysis (E04)
│   ├── experiment_runner.py   # Experiment dispatcher
│   ├── combine_figures.py     # Figure combination utility
│   ├── e01_specialized_visualizer_v2.py  # E01 visualizer
│   ├── e02_specialized_visualizer.py     # E02 visualizer
│   ├── e04_specialized_visualizer.py     # E04 visualizer
│   └── plot_figure1b.py       # Figure 1b generator
│
├── exams/                     # Experiment configurations and data
│   ├── baseline/
│   ├── E01-algorithm_verification/
│   ├── E02-sync_sensitivity/
│   ├── E02-sync_level{1,2,3,4}_v2/
│   └── E04-combined_effects/
│       ├── config.json
│       └── output/latest/results.json
│
└── output/                    # Generated figures (auto-created)
```

## Algorithm Overview

### Three-Channel Correlation Method

The core algorithm estimates self-noise PSD by exploiting:
- Coherent external signal: same across all channels
- Independent self-noise: uncorrelated between channels

Self-noise estimation formula:
```
N_j = P_jj - (P_jk × P_jl) / P_kl
```

### Two-Path Coefficient of Variation (TPCV)

The key innovation: use two mathematically equivalent paths to compute self-noise and measure their agreement.

```
Path 1: N_j^(1) = P_jj - (P_jk × P_jl) / P_kl
Path 2: N_j^(2) = P_jj - (P_jl × P_jk) / P_lk

TPCV = |N^(1) - N^(2)| / (N^(1) + N^(2)) × √2
```

TPCV serves as a quality indicator:
- TPCV < 0.10: Excellent quality
- TPCV < 0.20: Good quality
- TPCV < 0.50: Suspicious
- TPCV ≥ 0.50: Unreliable

## Experiments

### E01: Algorithm Verification
- Ideal conditions with perfect synchronization
- Validates core algorithm correctness

### E02: Synchronization Sensitivity
- Tests algorithm response to timing errors
- 5 delay levels: 0, ±1.25ms, ±2.5ms, ±3.75ms, ±5ms

### E04: Combined Effects
- 10×10 factorial design
- Factors: sync error (0-22.5ms) × signal strength (0.1x-10x)
- Analyzes interaction effects on TPCV

## Requirements

- Python 3.10+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- SciencePlots >= 2.0.0
- Pillow >= 9.0.0

## Citation

If you use this code, please cite:
```
[Paper citation to be added after publication]
```

## License

This code is released for academic research purposes.
