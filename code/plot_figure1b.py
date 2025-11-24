#!/usr/bin/env python3
"""
Plot Three-Channel Frequency Domain Processing Diagram (Figure 1b)

Purpose: Demonstrate complete spectral transformation process from raw ASD to extracted self-noise
Corresponding document: doc/review/20251110晚/R12_investigation.md
Corresponding LaTeX figure: E01_figure1b_frequency_processing.png

Plot contents:
- Raw ASD of three channels (√P_11, √P_22, √P_33)
- Channel 1 self-noise extracted via two paths (N_1^(1), N_1^(2))
- Theoretical self-noise level (black dashed line)
"""

import sys
import json
import argparse
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import scienceplots  # For IEEE publication-quality figures
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))
from core_algorithm import ThreeChannelCorrelation

# Note: Font settings not applied here to avoid overriding scienceplots IEEE style
# IEEE style requires serif font (Times New Roman), will be set in main() and plot functions


def load_baseline_config():
    """Load baseline experiment configuration"""
    config_path = Path(__file__).parent.parent / "exams" / "baseline" / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_signals_and_compute_spectra(config):
    """
    Generate signals and compute spectra

    Returns:
        dict: Dictionary containing frequency array and various ASD data
    """
    # Create algorithm instance
    algo = ThreeChannelCorrelation(config)

    # Generate three-channel signals
    ch1, ch2, ch3 = algo.generate_three_channel_signals()

    # Compute raw PSD of three channels, then take square root for ASD
    f, P_11 = algo.compute_psd(ch1)
    _, P_22 = algo.compute_psd(ch2)
    _, P_33 = algo.compute_psd(ch3)

    # Compute cross power spectral density (for two-path self-noise extraction)
    _, P_12 = algo.compute_cross_psd(ch1, ch2)
    _, P_13 = algo.compute_cross_psd(ch1, ch3)
    _, P_23 = algo.compute_cross_psd(ch2, ch3)
    _, P_32 = algo.compute_cross_psd(ch3, ch2)  # P_32 = conj(P_23)

    # Compute self-noise via two paths
    # Path 1: N_1^(1) = P_11 - (P_12 × P_13) / P_23
    N_1_path1 = P_11 - (P_12 * P_13) / P_23

    # Path 2: N_1^(2) = P_11 - (P_13 × P_12) / P_32
    N_1_path2 = P_11 - (P_13 * P_12) / P_32

    # Take real part and absolute value (physically self-noise PSD must be positive real)
    N_1_path1 = np.real(N_1_path1)
    N_1_path2 = np.real(N_1_path2)

    # Theoretical self-noise level (read from configuration)
    noise_asd_theory = config["signal_parameters"]["noise_asd_ng_sqrthz"]

    # Take square root of PSD to get ASD (ng/√Hz)
    return {
        'frequencies': f,
        'P_11': np.sqrt(P_11),      # ASD: ng/√Hz
        'P_22': np.sqrt(P_22),      # ASD: ng/√Hz
        'P_33': np.sqrt(P_33),      # ASD: ng/√Hz
        'N_1_path1': np.sqrt(np.abs(N_1_path1)),  # ASD: ng/√Hz (abs处理负值)
        'N_1_path2': np.sqrt(np.abs(N_1_path2)),  # ASD: ng/√Hz (abs处理负值)
        'theory': noise_asd_theory  # ASD: ng/√Hz
    }


def save_json_data(spectra, config, output_path):
    """
    Save plot data and configuration to JSON file

    Args:
        spectra: Dictionary containing all spectral data
        config: Experiment configuration dictionary
        output_path: JSON file output path
    """
    from datetime import datetime

    # Prepare JSON data structure
    json_data = {
        # Metadata
        "metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "script_version": "1.0.0",
            "script_name": "plot_figure1b.py",
            "description": "Three-channel frequency domain processing data",
            "figure_name": "E01_figure1b_frequency_processing",
            "data_source": "Synthetic three-channel signals with baseline configuration"
        },

        # Configuration parameters (key to experiment reproducibility)
        "configuration": {
            "sampling_parameters": {
                "sampling_rate_hz": config["signal_parameters"]["sampling_rate_hz"],
                "signal_duration_s": config["signal_parameters"]["signal_length_seconds"],
                "total_samples": config["signal_parameters"]["sampling_rate_hz"] * config["signal_parameters"]["signal_length_seconds"]
            },
            "signal_parameters": {
                "target_frequency_hz": config["signal_parameters"]["target_frequency_hz"],
                "signal_amplitude_ng_sqrthz": config["signal_parameters"]["signal_asd_ng_sqrthz"],
                "noise_amplitude_ng_sqrthz": config["signal_parameters"]["noise_asd_ng_sqrthz"]
            },
            "welch_parameters": {
                "nperseg": config["welch_parameters"]["nperseg"],
                "overlap_ratio": config["welch_parameters"]["overlap_ratio"],
                "window_type": config["welch_parameters"]["window"],
                "detrend": config["welch_parameters"].get("detrend", "constant")
            },
            "random_seed": config.get("computation_settings", {}).get("random_seed", "not_specified")
        },

        # Plot data (complete)
        "plot_data": {
            "frequencies": {
                "values": spectra['frequencies'].tolist(),
                "unit": "Hz",
                "description": "Frequency array from Welch PSD estimation",
                "length": len(spectra['frequencies'])
            },
            "channel_1_asd": {
                "values": spectra['P_11'].tolist(),
                "unit": "ng/sqrt(Hz)",
                "label": "sqrt(P_11) (Ch1)",
                "description": "Channel 1 amplitude spectral density"
            },
            "channel_2_asd": {
                "values": spectra['P_22'].tolist(),
                "unit": "ng/sqrt(Hz)",
                "label": "sqrt(P_22) (Ch2)",
                "description": "Channel 2 amplitude spectral density"
            },
            "channel_3_asd": {
                "values": spectra['P_33'].tolist(),
                "unit": "ng/sqrt(Hz)",
                "label": "sqrt(P_33) (Ch3)",
                "description": "Channel 3 amplitude spectral density"
            },
            "noise_path1_asd": {
                "values": spectra['N_1_path1'].tolist(),
                "unit": "ng/sqrt(Hz)",
                "label": "sqrt(N_1^(1)) (Path 1)",
                "description": "Channel 1 self-noise extracted via Path 1: N_1 = P_11 - (P_12 * P_13) / P_23"
            },
            "noise_path2_asd": {
                "values": spectra['N_1_path2'].tolist(),
                "unit": "ng/sqrt(Hz)",
                "label": "sqrt(N_1^(2)) (Path 2)",
                "description": "Channel 1 self-noise extracted via Path 2: N_1 = P_11 - (P_13 * P_12) / P_32"
            },
            "theoretical_noise": {
                "value": spectra['theory'],
                "unit": "ng/sqrt(Hz)",
                "label": f"Theory ({spectra['theory']:.2e} ng/sqrt(Hz))",
                "description": "Theoretical noise level from configuration"
            }
        },

        # Plot settings
        "plot_settings": {
            "figure_size_inches": [7, 4.2],
            "style": ["science", "ieee"],
            "font_settings": {
                "base_font_size": 8,
                "axes_title_size": 9,
                "axes_label_size": 8,
                "tick_label_size": 7,
                "legend_font_size": 7
            },
            "line_styles": {
                "channel_1": {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.2, "label": "IEEE blue solid"},
                "channel_2": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 1.2, "label": "IEEE orange dashed"},
                "channel_3": {"color": "#2ca02c", "linestyle": "-.", "linewidth": 1.2, "label": "IEEE green dash-dot"},
                "noise_path1": {"color": "#d62728", "marker": "o", "markersize": 3.5, "label": "IEEE red circles"},
                "noise_path2": {"color": "#9467bd", "marker": "s", "markersize": 3.5, "label": "IEEE purple squares"},
                "theory": {"color": "k", "linestyle": "--", "linewidth": 0.8, "alpha": 0.7, "label": "black dashed"}
            },
            "axes": {
                "xlabel": "Frequency (Hz)",
                "ylabel": "Amplitude Spectral Density (ng/sqrt(Hz))",
                "title": "Three-Channel Frequency Domain Processing",
                "scale": "log-log"
            },
            "legend": {
                "location": "lower right",
                "ncol": 2,
                "frameon": True,
                "framealpha": 0.95
            },
            "grid": {
                "major": {"alpha": 0.3, "linewidth": 0.5, "linestyle": "-"},
                "minor": {"alpha": 0.15, "linewidth": 0.3, "linestyle": ":"}
            },
            "dpi": 600
        },

        # Statistical information
        "statistics": {
            "frequency_range_hz": {
                "min": float(spectra['frequencies'][1]),  # Skip DC
                "max": float(spectra['frequencies'][-1])
            },
            "number_of_frequency_points": len(spectra['frequencies']),
            "two_path_deviation": {
                "description": "Relative deviation between Path 1 and Path 2",
                "mean": float(np.mean(np.abs(spectra['N_1_path1'] - spectra['N_1_path2']) /
                                     (spectra['N_1_path1'] + spectra['N_1_path2'] + 1e-20) * 2)),
                "max": float(np.max(np.abs(spectra['N_1_path1'] - spectra['N_1_path2']) /
                                    (spectra['N_1_path1'] + spectra['N_1_path2'] + 1e-20) * 2))
            }
        }
    }

    # Save JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] JSON data saved to: {output_path}")


def plot_frequency_processing(spectra, output_path):
    """
    Plot frequency domain processing diagram

    Args:
        spectra: Dictionary containing all spectral data
        output_path: Output figure path
    """
    # Apply IEEE style (using context manager for consistency)
    with plt.style.context(['science', 'ieee']):
        # Force serif font (IEEE standard requirement)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

        # IEEE standard font size settings (global consistency)
        plt.rcParams.update({
            'font.size': 8,           # 基础字号
            'axes.titlesize': 9,      # 子图标题
            'axes.labelsize': 8,      # 坐标轴标签
            'xtick.labelsize': 7,     # X轴刻度标签
            'ytick.labelsize': 7,     # Y轴刻度标签
            'legend.fontsize': 7,     # 图例字号
            'figure.titlesize': 10    # Main title
        })

        # Create figure (IEEE double-column width: 7 inches, height adapted to golden ratio)
        fig, ax = plt.subplots(figsize=(7, 4.2))

        f = spectra['frequencies']

        # IEEE line width standards
        line_width_main = 1.2      # Main curve line width (increased: 1.0→1.2)
        line_width_ref = 0.8       # Reference line width
        marker_size = 3.5          # Marker size
        marker_edge_width = 0.7    # Marker edge width

        # Plot raw ASD of three channels (using IEEE recommended colors and line styles)
        ax.loglog(f, spectra['P_11'], '-', color='#1f77b4', linewidth=line_width_main,
                  label=r'$\sqrt{P_{11}}$ (Ch1)', zorder=3)      # IEEE blue solid line
        ax.loglog(f, spectra['P_22'], '--', color='#ff7f0e', linewidth=line_width_main,
                  label=r'$\sqrt{P_{22}}$ (Ch2)', zorder=3)      # IEEE orange dashed line
        ax.loglog(f, spectra['P_33'], '-.', color='#2ca02c', linewidth=line_width_main,
                  label=r'$\sqrt{P_{33}}$ (Ch3)', zorder=3)      # IEEE green dash-dot line

        # Marker spacing (uniform distribution in log space, avoid dense low-freq and sparse high-freq)
        # Select approximately 30 points in log space
        valid_f_mask = f > 0  # Skip zero frequency
        valid_indices = np.where(valid_f_mask)[0]
        if len(valid_indices) > 30:
            # Uniformly select 30 indices in log space
            log_indices = np.logspace(np.log10(valid_indices[0]), np.log10(valid_indices[-1]), 30)
            marker_indices = np.unique(log_indices.astype(int))
            # Ensure all indices are within valid range
            marker_indices = marker_indices[marker_indices < len(f)]
        else:
            marker_indices = valid_indices

        # Plot self-noise extracted via two paths (hollow markers, IEEE standard colors)
        ax.loglog(f[marker_indices], spectra['N_1_path1'][marker_indices],
                  'o', markersize=marker_size, markerfacecolor='none',
                  markeredgecolor='#d62728', markeredgewidth=marker_edge_width,
                  label=r'$\sqrt{N_1^{(1)}}$ (Path 1)', zorder=4)  # IEEE red circles
        ax.loglog(f[marker_indices], spectra['N_1_path2'][marker_indices],
                  's', markersize=marker_size, markerfacecolor='none',
                  markeredgecolor='#9467bd', markeredgewidth=marker_edge_width,
                  label=r'$\sqrt{N_1^{(2)}}$ (Path 2)', zorder=4)  # IEEE purple squares

        # Plot theoretical value (black dashed line, simplified label)
        ax.axhline(y=spectra['theory'], color='k', linestyle='--',
                   linewidth=line_width_ref, alpha=0.7,
                   label=f"Theory ({spectra['theory']:.2e} ng/$\\sqrt{{Hz}}$)",
                   zorder=2)

        # Set axis labels (IEEE standard format)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'Amplitude Spectral Density (ng/$\sqrt{\mathrm{Hz}}$)')
        # ax.set_title('Three-Channel Frequency Domain Processing', pad=10)  # Commented: removed figure title

        # IEEE standard grid lines (thin lines, low transparency)
        ax.grid(True, which='major', alpha=0.3, linewidth=0.5, linestyle='-', zorder=1)
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.3, linestyle=':', zorder=1)

        # Legend optimization (IEEE standard: 2-column layout, position avoids data obscuring)
        # Use ncol=2 to reduce vertical space, place in lower right to avoid obscuring 10-100Hz data
        ax.legend(loc='lower right', ncol=2, frameon=True, framealpha=0.95,
                  edgecolor='gray', fancybox=False, shadow=False,
                  columnspacing=1.0, handlelength=2.0)

        # Set axis range
        ax.set_xlim([1, 1000])  # Display 1Hz to 1000Hz range

        # Optimize layout (IEEE standard spacing)
        plt.tight_layout(pad=0.3)

        # Save figure (IEEE standard: 600 DPI for publication)
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
        print(f"[OK] Frequency processing figure saved to: {output_path}")

        # Close figure
        plt.close()


def parse_args():
    """Parse CLI arguments for custom output locations."""
    parser = argparse.ArgumentParser(description="Generate E01 Figure1b assets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store PNG/JSON outputs (default: manuscript/latex)"
    )
    parser.add_argument(
        "--output-basename",
        type=str,
        default="E01_figure1b_frequency_processing",
        help="Base filename without extension for generated assets"
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    # Apply IEEE style (scienceplots library)
    plt.style.use(['science', 'ieee'])

    # Force serif font (IEEE standard requirement, must be set after style.use to override default)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['axes.unicode_minus'] = False  # Correctly display minus sign

    plt.rcParams.update({
        'font.size': 8,           # Base font size
        'axes.titlesize': 9,      # Subplot title
        'axes.labelsize': 8,      # Axis labels
        'xtick.labelsize': 7,     # X-axis tick labels
        'ytick.labelsize': 7,     # Y-axis tick labels
        'legend.fontsize': 7      # Legend font size
    })

    print("=" * 60)
    print("Three-Channel Frequency Domain Processing Diagram Generator (Figure 1b)")
    print("=" * 60)
    print("[Style] Using scienceplots IEEE style")

    # 1. Load configuration
    print("\n[1/5] Loading baseline experiment configuration...")
    config_path = Path(__file__).parent.parent / "exams" / "baseline" / "config.json"

    # 2. Create temporary algorithm instance for signal generation
    print("[2/5] Generating three-channel signals and computing spectra...")

    # Create ThreeChannelCorrelation instance requires config file path
    # So we create a temporary config file or directly use existing one
    algo = ThreeChannelCorrelation(str(config_path))

    # Generate three-channel signals
    ch1, ch2, ch3 = algo.generate_three_channel_signals()
    print(f"  - Sampling rate: {algo.fs} Hz")
    print(f"  - Signal length: {algo.T} seconds")
    print(f"  - Target frequency: {algo.f0} Hz")

    # Compute all required spectra
    print("[3/5] Computing PSD and CPSD...")
    f, P_11 = algo.compute_psd(ch1)
    _, P_22 = algo.compute_psd(ch2)
    _, P_33 = algo.compute_psd(ch3)

    _, P_12 = algo.compute_cross_psd(ch1, ch2)
    _, P_13 = algo.compute_cross_psd(ch1, ch3)
    _, P_23 = algo.compute_cross_psd(ch2, ch3)
    _, P_32 = algo.compute_cross_psd(ch3, ch2)

    # Compute self-noise via two paths
    N_1_path1 = np.real(P_11 - (P_12 * P_13) / P_23)
    N_1_path2 = np.real(P_11 - (P_13 * P_12) / P_32)

    # Theoretical self-noise
    noise_asd_theory = algo.noise_asd_ng

    # Assemble data
    spectra = {
        'frequencies': f,
        'P_11': np.sqrt(P_11),  # ASD: ng/√Hz
        'P_22': np.sqrt(P_22),  # ASD: ng/√Hz
        'P_33': np.sqrt(P_33),  # ASD: ng/√Hz
        'N_1_path1': np.sqrt(np.abs(N_1_path1)),  # ASD: ng/√Hz
        'N_1_path2': np.sqrt(np.abs(N_1_path2)),  # ASD: ng/√Hz
        'theory': noise_asd_theory  # ASD: ng/√Hz
    }

    # 4. Plot and save figure and JSON data
    print("[4/5] Plotting frequency domain processing diagram...")
    default_output_dir = Path(__file__).parent.parent.parent / "manuscript" / "latex"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.output_basename or "E01_figure1b_frequency_processing"
    output_path_png = output_dir / f"{base_name}.png"
    output_path_json = output_dir / f"{base_name}.json"

    plot_frequency_processing(spectra, output_path_png)

    # 5. Save JSON data
    print("[5/5] Saving JSON data...")
    # Load configuration to pass to JSON save function
    config = json.loads(config_path.read_text(encoding='utf-8'))
    save_json_data(spectra, config, output_path_json)

    # Output statistical information
    print("\n" + "=" * 60)
    print("Statistical Information:")
    print("=" * 60)
    print(f"Frequency range: {f[1]:.4f} - {f[-1]:.2f} Hz")
    print(f"Number of frequency points: {len(f)}")

    # Calculate values at 10Hz
    freq_idx = np.argmin(np.abs(f - algo.f0))
    print(f"\nValues at {f[freq_idx]:.2f} Hz:")
    print(f"  P_11: {P_11[freq_idx]:.4e} ng/√Hz")
    print(f"  P_22: {P_22[freq_idx]:.4e} ng/√Hz")
    print(f"  P_33: {P_33[freq_idx]:.4e} ng/√Hz")
    print(f"  N_1^(1): {N_1_path1[freq_idx]:.4e} ng/√Hz")
    print(f"  N_1^(2): {N_1_path2[freq_idx]:.4e} ng/√Hz")
    print(f"  Theory: {noise_asd_theory:.4e} ng/√Hz")

    # Calculate relative deviation of two paths
    rel_diff = np.abs(N_1_path1 - N_1_path2) / (N_1_path1 + N_1_path2 + 1e-20) * 2
    print(f"\nTwo-path average relative deviation: {np.mean(rel_diff):.4f}")
    print(f"Two-path maximum relative deviation: {np.max(rel_diff):.4f}")

    print("\n" + "=" * 60)
    print("[OK] Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

