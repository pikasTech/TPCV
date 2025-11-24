#!/usr/bin/env python3
"""
E01 Algorithm Verification - Specialized Visualizer V2
======================================================
Refactored version: Generate individual subplots, then combine using combine_figures.py

Main improvements:
1. Each subplot generated as a separate image (no subplots, no titles)
2. Use combine_figures.py for unified combination
3. Labels (A), (B), (C) unified at bottom center
4. Figure captions added by LaTeX, not in image
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
import scienceplots
from combine_figures import combine_figures


class E01SpecializedVisualizerV2:
    """V2 version: Generate individual plots + combination"""

    def __init__(self, exam_name="E01-algorithm_verification", timestamp=None, base_dir=None):
        """Initialize visualizer"""
        self.exam_name = exam_name
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.exam_dir = self.base_dir / "exams" / exam_name

        # Find timestamp
        if timestamp is None:
            output_dir = self.exam_dir / "output"
            timestamps = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
            if not timestamps:
                raise ValueError(f"No output directories found in {output_dir}")
            self.timestamp = timestamps[-1]
        else:
            self.timestamp = timestamp

        self.output_dir = self.exam_dir / "output" / self.timestamp
        self.figures_dir = self.output_dir / "figures"

        # Create individual plots subdirectory
        self.individual_dir = self.figures_dir / "individual"
        self.individual_dir.mkdir(parents=True, exist_ok=True)

        # Load results and config
        self.load_data()
        self.setup_matplotlib_style()

    def load_data(self):
        """Load experiment results and configuration"""
        results_file = self.output_dir / "results.json"
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

        config_file = self.exam_dir / "config.json"
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        print(f"Loaded data from {self.output_dir}")

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

    def regenerate_signals(self, duration_seconds=0.1):
        """Regenerate signals, returning both coherent noise and self-noise components"""
        import sys
        code_dir = Path(__file__).parent
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))

        from core_algorithm import ThreeChannelCorrelation

        config_file = str(self.exam_dir / "config.json")
        core = ThreeChannelCorrelation(config_file)

        # Get configuration parameters
        fs = core.fs
        n_samples = int(duration_seconds * fs)

        # Generate coherent signal and self-noise
        # Use same method as core_algorithm
        test_noise = np.random.normal(0, 1.0, n_samples)

        # Adjust Welch parameters for small sample size
        nperseg = min(core.nperseg, n_samples)
        noverlap = int(nperseg * 0.5)  # Use 50% overlap

        from scipy import signal as sp_signal
        f_test, psd_test = sp_signal.welch(
            test_noise, fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )
        test_freq_idx = np.argmin(np.abs(f_test - core.f0))
        measured_asd = np.sqrt(psd_test[test_freq_idx])

        sigma_signal = core.signal_asd_ng / measured_asd
        sigma_noise = core.noise_asd_ng / measured_asd

        # Generate coherent signal (shared across all channels)
        coherent_signal = np.random.normal(0, sigma_signal, n_samples)

        # Generate self-noise for each channel
        noise1 = np.random.normal(0, sigma_noise * core.channel_noise_factors[0], n_samples)
        noise2 = np.random.normal(0, sigma_noise * core.channel_noise_factors[1], n_samples)
        noise3 = np.random.normal(0, sigma_noise * core.channel_noise_factors[2], n_samples)

        # Compose total signal
        ch1 = coherent_signal + noise1
        ch2 = coherent_signal + noise2
        ch3 = coherent_signal + noise3

        time = np.arange(n_samples) / fs

        print(f"Regenerated {duration_seconds}s signals with {n_samples} samples at {fs} Hz")

        return {
            'total': [ch1, ch2, ch3],
            'coherent': coherent_signal,
            'self_noise': [noise1, noise2, noise3],
            'time': time,
            'fs': fs
        }

    def generate_single_panel(
        self,
        data_dict,
        output_file,
        figsize=(7, 2.5),
        xlabel=None,
        ylabel=None
    ):
        """
        Generate single panel (no title, no label)

        Parameters
        ----------
        data_dict : dict
            Contains plot data and configuration
        output_file : Path
            Output file path
        figsize : tuple
            Figure size
        xlabel, ylabel : str
            Axis labels
        """
        with plt.style.context(['science', 'ieee']):
            fig, ax = plt.subplots(figsize=figsize)

            # Plot based on data type
            plot_type = data_dict.get('type', 'line')

            if plot_type == 'line':
                ax.plot(
                    data_dict['x'],
                    data_dict['y'],
                    color=data_dict.get('color', '#1f77b4'),
                    linewidth=data_dict.get('linewidth', 1.2),
                    alpha=data_dict.get('alpha', 0.7)
                )
                # Add RMS annotation (if present)
                if 'rms' in data_dict:
                    ax.text(
                        0.02, 0.95,
                        data_dict['rms_label'],
                        transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(
                            boxstyle='round',
                            facecolor='white',
                            alpha=0.8,
                            edgecolor='gray'
                        ),
                        fontsize=8
                    )

            elif plot_type == 'errorbar':
                ax.errorbar(
                    data_dict['x'],
                    data_dict['y'],
                    yerr=data_dict.get('yerr'),
                    fmt='o-',
                    capsize=4,
                    color=data_dict.get('color', '#1f77b4'),
                    linewidth=1.5,
                    markersize=6,
                    markerfacecolor='white',
                    markeredgewidth=1.5
                )
                if 'xticks' in data_dict:
                    ax.set_xticks(data_dict['xticks']['positions'])
                    ax.set_xticklabels(data_dict['xticks']['labels'])

            elif plot_type == 'heatmap':
                im = ax.imshow(
                    data_dict['data'],
                    cmap=data_dict.get('cmap', 'YlOrRd'),
                    aspect='auto'
                )
                ax.set_xticks(np.arange(len(data_dict['xticks'])))
                ax.set_yticks(np.arange(len(data_dict['yticks'])))
                ax.set_xticklabels(data_dict['xticks'])
                ax.set_yticklabels(data_dict['yticks'])

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(data_dict.get('cbar_label', ''))

                # Add value annotations
                if data_dict.get('annotate', False):
                    for i in range(data_dict['data'].shape[0]):
                        for j in range(data_dict['data'].shape[1]):
                            ax.text(
                                j, i,
                                f"{data_dict['data'][i, j]:.1f}",
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=7
                            )

            elif plot_type == 'bar':
                bars = ax.bar(
                    data_dict['x'],
                    data_dict['y'],
                    color=data_dict.get('color', '#2ca02c'),
                    alpha=data_dict.get('alpha', 0.7),
                    edgecolor='black',
                    linewidth=1.0
                )
                if 'xticks' in data_dict:
                    ax.set_xticks(data_dict['xticks']['positions'])
                    ax.set_xticklabels(data_dict['xticks']['labels'])

                # Add horizontal line (if present)
                if 'hline' in data_dict:
                    ax.axhline(
                        y=data_dict['hline']['y'],
                        color=data_dict['hline'].get('color', 'red'),
                        linestyle=data_dict['hline'].get('linestyle', '--'),
                        linewidth=data_dict['hline'].get('linewidth', 1.5),
                        label=data_dict['hline'].get('label')
                    )
                    if data_dict['hline'].get('label'):
                        ax.legend(loc='best', frameon=True)

            # Set axis labels
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)

            # Add grid
            if data_dict.get('grid', True):
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save
            fig.savefig(
                output_file,
                format='png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            print(f"[OK] Saved panel: {output_file.name}")
            plt.close(fig)

        return output_file

    def generate_f00_individual_panels(self):
        """
        Generate 3 individual panels for F00 (time domain waveforms)
        Each subplot shows three lines: coherent noise, self-noise, total noise
        Display only first 0.1 seconds

        Returns
        -------
        list
            List of 3 panel file paths
        """
        print("\n" + "="*60)
        print("Generating F00: Individual Time Domain Panels")
        print("="*60)

        # Regenerate signals (including components), use 0.1 seconds
        signals = self.regenerate_signals(duration_seconds=0.1)
        time = signals['time']
        fs = signals['fs']

        # Display only first 0.1 seconds
        display_duration = 0.1
        display_samples = int(display_duration * fs)

        # Define configuration for 3 channels
        channels = [
            {
                'total': signals['total'][0][:display_samples],
                'coherent': signals['coherent'][:display_samples],
                'self_noise': signals['self_noise'][0][:display_samples],
                'label': 'Channel 1',
                'filename': 'figure0_time_domain_A.png'
            },
            {
                'total': signals['total'][1][:display_samples],
                'coherent': signals['coherent'][:display_samples],
                'self_noise': signals['self_noise'][1][:display_samples],
                'label': 'Channel 2',
                'filename': 'figure0_time_domain_B.png'
            },
            {
                'total': signals['total'][2][:display_samples],
                'coherent': signals['coherent'][:display_samples],
                'self_noise': signals['self_noise'][2][:display_samples],
                'label': 'Channel 3',
                'filename': 'figure0_time_domain_C.png'
            }
        ]

        panel_files = []

        # First calculate y-axis range for all channels to ensure consistency
        all_y_values = []
        for ch_config in channels:
            all_y_values.extend(ch_config['coherent'])
            all_y_values.extend(ch_config['self_noise'])
            all_y_values.extend(ch_config['total'])
        y_min, y_max = np.min(all_y_values), np.max(all_y_values)
        y_margin = (y_max - y_min) * 0.05

        # Apply IEEE style (completely mimicking E02)
        with plt.style.context(['science', 'ieee']):
            # Generate each independent subplot
            for i, ch_config in enumerate(channels):
                output_file = self.individual_dir / ch_config['filename']

                # Create figure
                fig, ax = plt.subplots(figsize=(6, 4))

                # Plot three waveforms
                ax.plot(time, ch_config['coherent'],
                       label='Background noise',
                       alpha=0.7,
                       linewidth=1,
                       color='blue',
                       linestyle='--')
                ax.plot(time, ch_config['self_noise'],
                       label='Self noise',
                       alpha=0.7,
                       linewidth=1,
                       color='red',
                       linestyle=':')
                ax.plot(time, ch_config['total'],
                       label='Total noise',
                       alpha=0.7,
                       linewidth=1,
                       color='black',
                       linestyle='-')

                # Set uniform y-axis range
                ax.set_ylim(y_min - y_margin, y_max + y_margin)

                # Set axis labels
                ax.set_xlabel(r'Time (s)')
                ax.set_ylabel(r'Amplitude (ng)')

                # Add legend
                ax.legend(loc='best', framealpha=0.8, edgecolor='gray', fontsize=8)

                # Add grid
                ax.grid(True, alpha=0.3)

                # Save
                plt.tight_layout()
                fig.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig)

                print(f"[OK] Saved panel: {output_file.name}")
                panel_files.append(output_file)

        return panel_files

    def generate_f00_combined(self):
        """
        Combine 3 panels of F00 into one composite figure

        Returns
        -------
        Path
            Composite figure file path
        """
        print("\n" + "="*60)
        print("Combining F00 panels...")
        print("="*60)

        # Generate individual panels
        panel_files = self.generate_f00_individual_panels()

        # Combine (horizontal layout)
        output_file = self.figures_dir / "figure0_time_domain_waveforms.png"

        combine_figures(
            input_files=panel_files,
            output_file=output_file,
            layout=(1, 3),
            labels=['a', 'b', 'c'],
            label_position='bottom_center',
            spacing=0.02
        )

        return output_file

    def generate_f01_individual_panels(self):
        """
        Generate 4 individual panels for F01 (algorithm validation)

        Returns
        -------
        list
            List of 4 panel file paths
        """
        print("\n" + "="*60)
        print("Generating F01: Individual Algorithm Validation Panels")
        print("="*60)

        # Extract result data
        results_data = self.results['results']
        permutations = ['123', '132', '213', '231', '312', '321']
        physical_channels = ['physical_channel_1', 'physical_channel_2', 'physical_channel_3']

        # Extract self-noise values
        noise_values = {perm: [] for perm in permutations}
        for perm in permutations:
            for ch in physical_channels:
                if perm in results_data[ch]:
                    noise_values[perm].append(results_data[ch][perm])

        # Calculate statistics
        perm_means = {perm: np.mean(vals) for perm, vals in noise_values.items()}
        perm_stds = {perm: np.std(vals) for perm, vals in noise_values.items()}
        perm_cvs = {perm: (perm_stds[perm] / perm_means[perm] * 100) if perm_means[perm] > 0 else 0
                   for perm in permutations}

        # Calculate relative error
        all_values = [val for vals in noise_values.values() for val in vals]
        global_mean = np.mean(all_values)
        relative_errors = {perm: (perm_means[perm] - global_mean) / global_mean * 100
                          for perm in permutations}

        # Create heatmap data
        heatmap_data = np.zeros((3, 6))
        for i, ch in enumerate(physical_channels):
            for j, perm in enumerate(permutations):
                if perm in results_data[ch]:
                    heatmap_data[i, j] = results_data[ch][perm]

        # Panel A: Permutation sensitivity
        x_pos = np.arange(len(permutations))
        means = [perm_means[p] for p in permutations]
        stds = [perm_stds[p] for p in permutations]

        panel_a_dict = {
            'type': 'errorbar',
            'x': x_pos,
            'y': means,
            'yerr': stds,
            'color': '#1f77b4',
            'xticks': {'positions': x_pos, 'labels': permutations},
            'grid': True
        }

        # Panel B: Heatmap
        panel_b_dict = {
            'type': 'heatmap',
            'data': heatmap_data,
            'cmap': 'YlOrRd',
            'xticks': permutations,
            'yticks': ['Ch1', 'Ch2', 'Ch3'],
            'cbar_label': r'Self-Noise (ng/$\sqrt{\mathrm{Hz}}$)',
            'annotate': True
        }

        # Panel C: CV distribution
        cv_values = [perm_cvs[p] for p in permutations]
        mean_cv = np.mean(cv_values)

        panel_c_dict = {
            'type': 'bar',
            'x': x_pos,
            'y': cv_values,
            'color': '#2ca02c',
            'alpha': 0.7,
            'xticks': {'positions': x_pos, 'labels': permutations},
            'hline': {
                'y': mean_cv,
                'color': 'red',
                'linestyle': '--',
                'linewidth': 1.5,
                'label': f'Mean CV = {mean_cv:.2f}%'
            },
            'grid': True
        }

        # Panel D: Relative error
        rel_err_values = [relative_errors[p] for p in permutations]
        colors = ['#d62728' if err > 0 else '#1f77b4' for err in rel_err_values]

        panel_d_dict = {
            'type': 'bar',
            'x': x_pos,
            'y': rel_err_values,
            'color': colors,
            'alpha': 0.7,
            'xticks': {'positions': x_pos, 'labels': permutations},
            'grid': True
        }

        # Generate 4 panels
        panels = [
            {
                'data': panel_a_dict,
                'filename': 'figure1_validation_A.png',
                'xlabel': 'Channel Permutation',
                'ylabel': r'Self-Noise (ng/$\sqrt{\mathrm{Hz}}$)',
                'figsize': (3.5, 2.5)
            },
            {
                'data': panel_b_dict,
                'filename': 'figure1_validation_B.png',
                'xlabel': 'Permutation',
                'ylabel': 'Physical Channel',
                'figsize': (3.5, 2.5)
            },
            {
                'data': panel_c_dict,
                'filename': 'figure1_validation_C.png',
                'xlabel': 'Channel Permutation',
                'ylabel': 'Coefficient of Variation (%)',
                'figsize': (3.5, 2.5)
            },
            {
                'data': panel_d_dict,
                'filename': 'figure1_validation_D.png',
                'xlabel': 'Channel Permutation',
                'ylabel': 'Relative Error (%)',
                'figsize': (3.5, 2.5)
            }
        ]

        panel_files = []
        for panel in panels:
            output_file = self.individual_dir / panel['filename']
            self.generate_single_panel(
                panel['data'],
                output_file,
                figsize=panel['figsize'],
                xlabel=panel['xlabel'],
                ylabel=panel['ylabel']
            )
            panel_files.append(output_file)

        return panel_files

    def generate_f01_combined(self):
        """
        Combine 4 panels of F01 into one 2x2 composite figure

        Returns
        -------
        Path
            Composite figure file path
        """
        print("\n" + "="*60)
        print("Combining F01 panels...")
        print("="*60)

        # Generate individual panels
        panel_files = self.generate_f01_individual_panels()

        # Combine
        output_file = self.figures_dir / "figure1_algorithm_validation.png"

        combine_figures(
            input_files=panel_files,
            output_file=output_file,
            layout=(2, 2),
            labels=['a', 'b', 'c', 'd'],
            label_position='bottom_center',
            spacing=0.05
        )

        return output_file

    def generate_all_figures(self):
        """Generate all figures (individual plots + composite figures)"""
        print("\n" + "="*60)
        print(f"E01 Specialized Visualization V2")
        print(f"Experiment: {self.exam_name}")
        print(f"Timestamp: {self.timestamp}")
        print("="*60)

        # Ensure directories exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.individual_dir.mkdir(parents=True, exist_ok=True)

        # Generate composite figures (internally generates individual plots first)
        f00_file = self.generate_f00_combined()
        f01_file = self.generate_f01_combined()

        print("\n" + "="*60)
        print("All figures generated successfully!")
        print("="*60)
        print(f"\nF00 Combined: {f00_file}")
        print(f"F01 Combined: {f01_file}")
        print(f"\nIndividual panels: {self.individual_dir}")

        return [f00_file, f01_file]


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate specialized figures for E01 (V2: individual + combined)'
    )
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Experiment timestamp (uses latest if not provided)')

    args = parser.parse_args()

    visualizer = E01SpecializedVisualizerV2(timestamp=args.timestamp)
    visualizer.generate_all_figures()


if __name__ == '__main__':
    main()
