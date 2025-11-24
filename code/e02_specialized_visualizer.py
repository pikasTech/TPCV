#!/usr/bin/env python3
"""
E02-sync_sensitivity Specialized Visualizer
Generates summary figures for time synchronization sensitivity analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime


class E02SpecializedVisualizer:
    """Specialized visualizer for E02 synchronization sensitivity experiment"""

    def __init__(self, base_dir=None):
        """
        Initialize E02 specialized visualizer

        Args:
            base_dir: Project base directory (auto-detect by default)
        """
        if base_dir is None:
            # Auto-detect base_dir from current file location
            current_file = Path(__file__).resolve()
            # Current file is in code/ directory, go up two levels to base_dir
            base_dir = current_file.parent.parent

        self.base_dir = Path(base_dir)
        self.exams_dir = self.base_dir / "exams"
        self.setup_matplotlib_style()

    def save_figure_json(self, json_path, data_dict):
        """
        Save figure data to JSON file

        Args:
            json_path: Path to save JSON file
            data_dict: Dictionary containing plot data, parameters, and metadata
        """
        # Add timestamp to metadata
        if 'metadata' not in data_dict:
            data_dict['metadata'] = {}
        data_dict['metadata']['generation_timestamp'] = datetime.now().isoformat()

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        data_dict = convert_numpy(data_dict)

        # Save to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

        print(f"  Saved JSON: {json_path}")

    def setup_matplotlib_style(self):
        """Configure matplotlib for publication-quality figures with scienceplots IEEE style"""
        try:
            import scienceplots
            plt.style.use(['science', 'ieee'])
        except ImportError:
            print("Warning: scienceplots not installed. Using fallback style.")
            print("Install with: pip install scienceplots")

        # Force serif font (IEEE requirement, must be set after style.use to override defaults)
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

        # Chinese font configuration (override IEEE default settings)
        import matplotlib.font_manager as fm
        if hasattr(fm, '_fmcache'):
            fm._fmcache.clear()

        # Set Chinese font (SimHei) for Chinese character display support
        matplotlib.rcParams['font.sans-serif'] = [
            'SimHei',           # Prefer SimHei (Chinese)
            'DejaVu Sans',
            'Arial',
            'Helvetica'
        ]
        matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

        # IEEE style font size configuration
        matplotlib.rcParams['font.size'] = 8
        matplotlib.rcParams['axes.labelsize'] = 9
        matplotlib.rcParams['axes.titlesize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['legend.fontsize'] = 7
        matplotlib.rcParams['figure.titlesize'] = 11

        # Other IEEE style configuration
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.rcParams['savefig.dpi'] = 300

        # LaTeX rendering configuration (use matplotlib built-in LaTeX, no system TeX dependency)
        matplotlib.rcParams['text.usetex'] = False  # Use matplotlib built-in LaTeX
        matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern font
        matplotlib.rcParams['mathtext.default'] = 'regular'

    def load_experiment_results(self, experiment_name):
        """
        Load results from an experiment

        Args:
            experiment_name: Name of the experiment (e.g., 'E02-sync_sensitivity')

        Returns:
            dict: Results data
        """
        exam_dir = self.exams_dir / experiment_name
        output_dir = exam_dir / "output"

        # Find the latest timestamp
        timestamps = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
        if not timestamps:
            raise FileNotFoundError(f"No output found for {experiment_name}")

        latest_timestamp = timestamps[-1]
        results_file = output_dir / latest_timestamp / "results.json"

        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def calculate_tpcv(self, results_data):
        """
        Calculate two-path coefficient of variation for all permutations

        Args:
            results_data: Results dictionary from experiment

        Returns:
            float: Maximum TPCV value across all physical channels
        """
        tpcvs = []

        for channel_key in ['physical_channel_1', 'physical_channel_2', 'physical_channel_3']:
            if channel_key not in results_data['results']:
                continue

            channel_data = results_data['results'][channel_key]

            # Extract noise values for permutations
            noise_values = []
            for perm in ['123', '132', '213', '231', '312', '321']:
                if perm in channel_data:
                    noise_values.append(channel_data[perm])

            if len(noise_values) >= 2:
                mean_val = np.mean(noise_values)
                std_val = np.std(noise_values, ddof=1)
                tpcv = (std_val / mean_val) if mean_val != 0 else 0
                tpcvs.append(tpcv)

        return max(tpcvs) if tpcvs else 0

    def load_all_e02_data(self):
        """
        Load data from all E02 experiments

        Returns:
            dict: Dictionary mapping delay (±microseconds) to TPCV values
        """
        # Note: Delays represent HALF of maximum channel-to-channel span (±representation)
        # This change (R16, 2025-11-21) converts from span to ±notation for academic convention
        # Level 0: Ch1=0, Ch2=0,   Ch3=0   => ±0    = 0 μs    (span=0 samples)
        # Level 1: Ch1=0, Ch2=-2,  Ch3=+3  => ±2.5  = 1250 μs (span=5 samples = 2.5ms)
        # Level 2: Ch1=0, Ch2=-5,  Ch3=+5  => ±5    = 2500 μs (span=10 samples = 5ms)
        # Level 3: Ch1=0, Ch2=-7,  Ch3=+8  => ±7.5  = 3750 μs (span=15 samples = 7.5ms)
        # Level 4: Ch1=0, Ch2=-10, Ch3=+10 => ±10   = 5000 μs (span=20 samples = 10ms)
        experiments = {
            'E02-sync_sensitivity': 0,      # 0 μs (±0, span=0)
            'E02-sync_level1_v2': 1250,     # 1250 μs (±1.25ms, span=2.5ms, 5 samples @ 2000Hz)
            'E02-sync_level2_v2': 2500,     # 2500 μs (±2.5ms, span=5ms, 10 samples @ 2000Hz)
            'E02-sync_level3_v2': 3750,     # 3750 μs (±3.75ms, span=7.5ms, 15 samples @ 2000Hz)
            'E02-sync_level4_v2': 5000,     # 5000 μs (±5ms, span=10ms, 20 samples @ 2000Hz)
        }

        data = {}
        for exp_name, delay_us_pm in experiments.items():  # Renamed: delay_us → delay_us_pm (±)
            try:
                results = self.load_experiment_results(exp_name)
                tpcv = self.calculate_tpcv(results)
                data[delay_us_pm] = {
                    'tpcv': tpcv,
                    'experiment': exp_name,
                    'results': results
                }
            except Exception as e:
                print(f"Warning: Could not load {exp_name}: {e}")

        return data

    def generate_f02_tpcv_vs_sync_error(self, output_dir):
        """
        Generate F02: TPCV vs Time Synchronization Error figure

        Args:
            output_dir: Directory to save the figure
        """
        # Load all E02 data
        data = self.load_all_e02_data()

        # Sort by delay
        delays = sorted(data.keys())
        tpcvs = [data[d]['tpcv'] for d in delays]

        # Apply IEEE style
        with plt.style.context(['science', 'ieee']):
            # Create figure (use same figsize as figure3)
            fig, ax = plt.subplots(figsize=(6, 4))

            # Plot TPCV vs sync error
            ax.plot(delays, tpcvs, 'o-', linewidth=1.2, markersize=6,
                    color='#2E86AB', label='Measured TPCV')

            # Add TPCV=0.20 threshold line
            ax.axhline(y=0.20, color='red', linestyle='--', linewidth=1.2,
                       label='TPCV=0.20 threshold')

            # Configure axes (use IEEE style labels)
            ax.set_xlabel(r'Time Synchronization Error ($\mu$s)')
            ax.set_ylabel('Two-Path Coefficient of Variation (TPCV)')
            # ax.set_title('TPCV vs Time Synchronization Error')  # Commented: Remove figure title

            # Use logarithmic scale for x-axis if there's a wide range (and no zero values)
            if min(delays) > 0 and max(delays) / min(delays) > 10:
                ax.set_xscale('log')
                # Add minor ticks for log scale
                from matplotlib.ticker import LogLocator, NullFormatter
                ax.xaxis.set_minor_locator(LogLocator(subs='all'))
                ax.xaxis.set_minor_formatter(NullFormatter())

            # Grid
            ax.grid(True, alpha=0.3)

            # Legend - place inside the figure
            ax.legend(loc='best', framealpha=0.8, edgecolor='gray', fontsize=8)

            # Tight layout
            plt.tight_layout()

            # Save figure
            output_path = Path(output_dir) / "figure5_tpcv_sync_error.png"
            plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()

        # Save JSON data
        json_data = {
            'metadata': {
                'experiment_name': 'E02-sync_sensitivity',
                'figure_id': 'figure5',
                'figure_type': 'TPCV vs Time Synchronization Error',
                'description': 'Two-Path Coefficient of Variation vs Time Synchronization Error across 5 delay levels (±representation)',
                'representation_note': 'R16 (2025-11-21): Changed from span to ±notation for academic convention'
            },
            'experiment_parameters': {
                'sampling_rate_hz': 2000,
                'signal_length_seconds': 600,
                'target_frequency_hz': 10,
                'welch_nperseg': 4096,
                'welch_overlap_ratio': 0.875,
                'welch_window': 'hann',
                'random_seed': 42,
                'delay_representation': 'half_span'
            },
            'plot_data': {
                'x_axis': {
                    'label': 'Time Synchronization Error (μs)',
                    'values': delays,
                    'unit': 'microseconds',
                    'representation': '±half_span',
                    'note': 'Values represent ±half of the maximum channel span'
                },
                'y_axis': {
                    'label': 'Two-Path Coefficient of Variation (TPCV)',
                    'values': tpcvs,
                    'unit': 'dimensionless'
                },
                'threshold_line': {
                    'value': 0.20,
                    'description': 'TPCV threshold for quality assessment'
                }
            },
            'data_sources': {
                'experiments': [data[d]['experiment'] for d in delays]
            }
        }

        json_path = Path(output_dir) / "figure5_tpcv_sync_error.json"
        self.save_figure_json(json_path, json_data)

        print(f"Generated F02: {output_path}")
        return output_path

    def generate_f01a_time_domain_comparison(self, output_dir):
        """
        Generate F01a: Time Domain Comparison figure (3x2 layout)

        V2 approach: First generate 6 individual subplots, then combine using combine_figures

        Uses real time-domain waveform data to generate comparison plots

        Args:
            output_dir: Directory to save the figure
        """
        # Load data for level 0 (ideal) and level 4 (maximum delay)
        data = self.load_all_e02_data()

        if 0 not in data or 5000 not in data:
            raise ValueError(f"Missing required data. Available delays: {list(data.keys())}")

        ideal_data = data[0]['results']  # E02-sync_sensitivity
        max_delay_data = data[5000]['results']  # E02-sync_level4_v2 (±5ms, span=10ms)

        # Try to load waveform data
        ideal_waveform = ideal_data.get('waveform_data')
        max_delay_waveform = max_delay_data.get('waveform_data')

        if ideal_waveform is None or max_delay_waveform is None:
            print("WARNING: Missing waveform data, cannot generate real time-domain plots")
            print("Please run: python cli.py run --exam E02-sync_sensitivity")
            print("            python cli.py run --exam E02-sync_level4_v2")
            raise ValueError("Missing waveform data for E02 experiments")

        # Have waveform data, generate real time-domain plots - V2 approach: generate individual subplots
        print("Detected waveform data, generating real time-domain comparison plots (V2 approach: individual subplots)...")

        # Create individual directory to save individual subplots
        individual_dir = Path(output_dir) / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)

        # Extract waveform data
        ideal_ch1 = np.array(ideal_waveform['channel_1'])
        ideal_ch2 = np.array(ideal_waveform['channel_2'])
        ideal_ch3 = np.array(ideal_waveform['channel_3'])
        ideal_time = np.array(ideal_waveform['time'])

        max_delay_ch1 = np.array(max_delay_waveform['channel_1'])
        max_delay_ch2 = np.array(max_delay_waveform['channel_2'])
        max_delay_ch3 = np.array(max_delay_waveform['channel_3'])
        max_delay_time = np.array(max_delay_waveform['time'])

        # Display only first 0.1 seconds (according to experiment plan)
        display_duration = 0.1  # seconds
        fs = ideal_waveform['sampling_rate']
        display_samples = int(display_duration * fs)

        # Apply IEEE style
        with plt.style.context(['science', 'ieee']):
            panel_files = []

            # Panel a: Ideal Sync - Channel Overlay
            fig_a, ax_a = plt.subplots(figsize=(6, 4))
            ax_a.plot(ideal_time[:display_samples], ideal_ch1[:display_samples],
                      label='Ch1', alpha=0.7, linewidth=1.2)
            ax_a.plot(ideal_time[:display_samples], ideal_ch2[:display_samples],
                      label='Ch2', alpha=0.7, linewidth=1.2)
            ax_a.plot(ideal_time[:display_samples], ideal_ch3[:display_samples],
                      label='Ch3', alpha=0.7, linewidth=1.2)
            ax_a.set_xlabel(r'Time (s)')
            ax_a.set_ylabel(r'Amplitude (ng)')
            ax_a.legend(loc='best', framealpha=0.8, edgecolor='gray', fontsize=8)
            ax_a.grid(True, alpha=0.3)
            plt.tight_layout()
            file_a = individual_dir / "figure2_a.png"
            fig_a.savefig(file_a, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig_a)
            panel_files.append(file_a)
            print(f"  Generated panel a: {file_a.name}")

            # Panel b: Max Delay - Channel Overlay
            fig_b, ax_b = plt.subplots(figsize=(6, 4))
            ax_b.plot(max_delay_time[:display_samples], max_delay_ch1[:display_samples],
                      label='Ch1', alpha=0.7, linewidth=1.2)
            ax_b.plot(max_delay_time[:display_samples], max_delay_ch2[:display_samples],
                      label='Ch2', alpha=0.7, linewidth=1.2)
            ax_b.plot(max_delay_time[:display_samples], max_delay_ch3[:display_samples],
                      label='Ch3', alpha=0.7, linewidth=1.2)
            ax_b.set_xlabel(r'Time (s)')
            ax_b.set_ylabel(r'Amplitude (ng)')
            ax_b.legend(loc='best', framealpha=0.8, edgecolor='gray', fontsize=8)
            ax_b.grid(True, alpha=0.3)
            plt.tight_layout()
            file_b = individual_dir / "figure2_b.png"
            fig_b.savefig(file_b, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig_b)
            panel_files.append(file_b)
            print(f"  Generated panel b: {file_b.name}")

            # Panel c: Ideal Sync - Relative Delay
            fig_c, ax_c = plt.subplots(figsize=(6, 4))
            ax_c.plot(ideal_time[:display_samples],
                      ideal_ch2[:display_samples] - ideal_ch1[:display_samples],
                      label='Ch2 - Ch1', alpha=0.7, linewidth=1.2)
            ax_c.plot(ideal_time[:display_samples],
                      ideal_ch3[:display_samples] - ideal_ch1[:display_samples],
                      label='Ch3 - Ch1', alpha=0.7, linewidth=1.2)
            ax_c.set_xlabel(r'Time (s)')
            ax_c.set_ylabel(r'Difference (ng)')
            ax_c.legend(loc='best', framealpha=0.8, edgecolor='gray', fontsize=8)
            ax_c.grid(True, alpha=0.3)
            plt.tight_layout()
            file_c = individual_dir / "figure2_c.png"
            fig_c.savefig(file_c, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig_c)
            panel_files.append(file_c)
            print(f"  Generated panel c: {file_c.name}")

            # Panel d: Max Delay - Relative Delay
            fig_d, ax_d = plt.subplots(figsize=(6, 4))
            ax_d.plot(max_delay_time[:display_samples],
                      max_delay_ch2[:display_samples] - max_delay_ch1[:display_samples],
                      label='Ch2 - Ch1', alpha=0.7, linewidth=1.2)
            ax_d.plot(max_delay_time[:display_samples],
                      max_delay_ch3[:display_samples] - max_delay_ch1[:display_samples],
                      label='Ch3 - Ch1', alpha=0.7, linewidth=1.2)
            ax_d.set_xlabel(r'Time (s)')
            ax_d.set_ylabel(r'Difference (ng)')
            ax_d.legend(loc='best', framealpha=0.8, edgecolor='gray', fontsize=8)
            ax_d.grid(True, alpha=0.3)
            plt.tight_layout()
            file_d = individual_dir / "figure2_d.png"
            fig_d.savefig(file_d, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig_d)
            panel_files.append(file_d)
            print(f"  Generated panel d: {file_d.name}")

            # Panel e: Ideal Sync - Cross-correlation
            from scipy.signal import correlate

            corr_ideal_12 = correlate(ideal_ch1[:display_samples], ideal_ch2[:display_samples], mode='same')
            corr_ideal_13 = correlate(ideal_ch1[:display_samples], ideal_ch3[:display_samples], mode='same')
            lag_axis = (np.arange(len(corr_ideal_12)) - len(corr_ideal_12)//2) / fs * 1000  # ms

            fig_e, ax_e = plt.subplots(figsize=(6, 4))
            ax_e.plot(lag_axis, corr_ideal_12, label='Ch1-Ch2', alpha=0.7, linewidth=1.2)
            ax_e.plot(lag_axis, corr_ideal_13, label='Ch1-Ch3', alpha=0.7, linewidth=1.2)
            ax_e.set_xlabel(r'Lag (ms)')
            ax_e.set_ylabel(r'Cross-correlation')
            ax_e.legend(loc='best', framealpha=0.8, edgecolor='gray', fontsize=8)
            ax_e.grid(True, alpha=0.3)
            ax_e.set_xlim([-20, 20])  # Display only ±20ms
            plt.tight_layout()
            file_e = individual_dir / "figure2_e.png"
            fig_e.savefig(file_e, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig_e)
            panel_files.append(file_e)
            print(f"  Generated panel e: {file_e.name}")

            # Panel f: Max Delay - Cross-correlation
            corr_max_12 = correlate(max_delay_ch1[:display_samples], max_delay_ch2[:display_samples], mode='same')
            corr_max_13 = correlate(max_delay_ch1[:display_samples], max_delay_ch3[:display_samples], mode='same')

            fig_f, ax_f = plt.subplots(figsize=(6, 4))
            ax_f.plot(lag_axis, corr_max_12, label='Ch1-Ch2', alpha=0.7, linewidth=1.2)
            ax_f.plot(lag_axis, corr_max_13, label='Ch1-Ch3', alpha=0.7, linewidth=1.2)
            ax_f.set_xlabel(r'Lag (ms)')
            ax_f.set_ylabel(r'Cross-correlation')
            ax_f.legend(loc='best', framealpha=0.8, edgecolor='gray', fontsize=8)
            ax_f.grid(True, alpha=0.3)
            ax_f.set_xlim([-20, 20])  # Display only ±20ms
            plt.tight_layout()
            file_f = individual_dir / "figure2_f.png"
            fig_f.savefig(file_f, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig_f)
            panel_files.append(file_f)
            print(f"  Generated panel f: {file_f.name}")

        # Use combine_figures for combination
        from combine_figures import combine_figures

        output_path = Path(output_dir) / "figure2_sync_error_time_domain.png"
        combine_figures(
            input_files=panel_files,
            output_file=output_path,
            layout=(3, 2),
            labels=['a', 'b', 'c', 'd', 'e', 'f'],
            label_position='bottom_center',
            label_fontsize=18,
            label_fontweight='bold',
            dpi=300
        )

        print(f"Generated F01a (V2 combination): {output_path}")

        # Save JSON data (real waveform version)
        json_data = {
            'metadata': {
                'experiment_name': 'E02-sync_sensitivity',
                'figure_id': 'figure2',
                'figure_type': 'Time Domain Comparison (V2 - Individual Panels)',
                'description': 'Time domain analysis showing channel overlay, relative delay, and cross-correlation for ideal vs max delay synchronization. Generated as 6 individual panels then combined.',
                'generation_method': 'V2 - Individual panels + combine_figures'
            },
            'experiment_parameters': {
                'sampling_rate_hz': fs,
                'signal_length_seconds': 600,
                'target_frequency_hz': 10,
                'welch_nperseg': 4096,
                'welch_overlap_ratio': 0.875,
                'welch_window': 'hann',
                'random_seed': 42,
                'display_duration_seconds': display_duration
            },
            'plot_data': {
                'subplot_layout': '3x2',
                'individual_panels': [str(f) for f in panel_files],
                'row1_overlay': {
                    'description': 'Three-channel overlay comparison',
                    'panel_a': 'Ideal sync channel overlay',
                    'panel_b': 'Max delay channel overlay'
                },
                'row2_relative_delay': {
                    'description': 'Relative delay (Ch2-Ch1 and Ch3-Ch1)',
                    'panel_c': 'Ideal sync relative delay',
                    'panel_d': 'Max delay relative delay'
                },
                'row3_cross_correlation': {
                    'description': 'Cross-correlation between channels',
                    'panel_e': 'Ideal sync cross-correlation',
                    'panel_f': 'Max delay cross-correlation',
                    'lag_range_ms': [-20, 20]
                }
            },
            'data_sources': {
                'ideal_experiment': 'E02-sync_sensitivity (0 μs delay)',
                'max_delay_experiment': 'E02-sync_level4_v2 (10000 μs delay)'
            }
        }

        json_path = Path(output_dir) / "figure2_sync_error_time_domain.json"
        self.save_figure_json(json_path, json_data)

        return output_path

    def generate_all_figures(self, output_dir=None):
        """
        Generate all E02 summary figures

        Args:
            output_dir: Output directory (default: latest E02-sync_sensitivity output)
        """
        if output_dir is None:
            # Use latest E02-sync_sensitivity output directory
            exam_dir = self.exams_dir / "E02-sync_sensitivity" / "output"
            timestamps = sorted([d.name for d in exam_dir.iterdir() if d.is_dir()])
            if not timestamps:
                raise FileNotFoundError("No output directory found for E02-sync_sensitivity")
            output_dir = exam_dir / timestamps[-1] / "figures"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating E02 summary figures in: {output_dir}")

        # Generate figures
        f01a_path = self.generate_f01a_time_domain_comparison(output_dir)
        f02_path = self.generate_f02_tpcv_vs_sync_error(output_dir)

        return {
            'f01a': f01a_path,
            'f02': f02_path
        }


def main():
    """Main entry point for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate E02 synchronization sensitivity summary figures'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for figures (default: latest E02-sync_sensitivity/output/*/figures)'
    )

    args = parser.parse_args()

    # Create visualizer and generate figures
    visualizer = E02SpecializedVisualizer()
    results = visualizer.generate_all_figures(output_dir=args.output_dir)

    print("\n=== Summary ===")
    for fig_name, fig_path in results.items():
        print(f"{fig_name}: {fig_path}")
