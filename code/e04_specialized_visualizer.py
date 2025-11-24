"""
E04 Specialized Visualizer - Combined Effects Interaction Analysis

Generates publication-quality figures for E04 experiment:
- F05: figure11_tpcv_interaction_heatmap.pdf - Time sync and noise interaction heatmap
- F06: figure12_main_effects.pdf - Main effects analysis (2 subplots)
- F07: figure13_interaction_plot.pdf - Interaction effect multi-line plot

Based on E04 config requirements and 10x10 factorial design.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
import sys
from datetime import datetime

# Import scienceplots for IEEE style
try:
    import scienceplots
    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False
    print("Warning: scienceplots not installed. Using default matplotlib style.")
    print("Install with: pip install scienceplots")


def extract_experimental_parameters(results_path):
    """
    Extract experimental parameters from results.json for JSON metadata

    Returns dict with:
    - sampling_rate_hz
    - signal_length_seconds
    - target_frequency_hz
    - signal_asd_ng_sqrthz (baseline)
    - welch_nperseg, overlap_ratio, window
    - factorial_design parameters
    - sync error levels
    - noise multipliers
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})

    # Read config file path from metadata
    # Handle both absolute and relative paths, and Windows/Unix path separators
    config_file_str = metadata.get('config_file', '')
    config = {}

    if config_file_str:
        # Try as-is first
        config_path = Path(config_file_str)
        if not config_path.exists():
            # Try relative to results_path (go up to project root: exams/)
            # results_path structure: exams/E04-.../output/timestamp/results.json
            results_dir = Path(results_path).parent.parent.parent.parent  # up to project root (noise_exam/)
            config_path = results_dir / config_file_str.replace('\\', '/')

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # Last attempt: try from current working directory
            try:
                config_path = Path.cwd() / config_file_str.replace('\\', '/')
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
            except:
                pass

    return {
        # Signal parameters
        'sampling_rate_hz': config.get('signal_parameters', {}).get('sampling_rate_hz', 2000),
        'signal_length_seconds': config.get('signal_parameters', {}).get('signal_length_seconds', 600),
        'target_frequency_hz': config.get('signal_parameters', {}).get('target_frequency_hz', 10),
        'baseline_noise_ng_sqrthz': config.get('signal_parameters', {}).get('signal_asd_ng_sqrthz', 444),

        # Welch parameters
        'welch_nperseg': config.get('welch_parameters', {}).get('nperseg', 4096),
        'welch_overlap_ratio': config.get('welch_parameters', {}).get('overlap_ratio', 0.875),
        'welch_window': config.get('welch_parameters', {}).get('window', 'hann'),
        'frequency_resolution_hz': config.get('welch_parameters', {}).get('frequency_resolution_hz', 0.488),

        # Factorial design
        'factorial_design': config.get('factorial_design', {}),

        # Sync conditions
        'sync_conditions': config.get('experimental_conditions', {}).get('synchronization_error_config', {}).get('sync_error_conditions', []),

        # Noise levels
        'signal_multipliers': metadata.get('signal_multipliers', []),
        'fixed_noise_level_v_sqrthz': metadata.get('fixed_noise_level_v_sqrthz', 4.354e-06),

        # Quality thresholds
        'tpcv_threshold': config.get('performance_evaluation', {}).get('tpcv_threshold', 0.2),
        'tpcv_excellent': config.get('performance_evaluation', {}).get('tpcv_excellent', 0.1),
        'tpcv_good': config.get('performance_evaluation', {}).get('tpcv_good', 0.15),

        # Metadata
        'experiment_id': metadata.get('experiment_id', 'E04-combined_effects'),
        'timestamp': metadata.get('timestamp', ''),
        'timestamp_iso': metadata.get('timestamp_iso', ''),
    }


def load_e04_data(results_path):
    """Load and structure E04 results data"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract gradient_analysis (list of 100 conditions)
    conditions = data['gradient_analysis']

    # Extract unique sync levels and noise levels
    sync_levels = []
    noise_levels = []
    tpcv_matrix = {}
    error_matrix = {}  # New: relative error matrix

    for cond in conditions:
        sync_label = cond['sync_condition']
        noise_label = cond['signal_label']

        # Extract TPCV value
        tpcv = cond['performance_metrics']['coefficient_of_variation']

        # Extract relative error: |estimation_accuracy - 1.0|
        estimation_accuracy = cond['performance_metrics'].get('estimation_accuracy', 1.0)
        relative_error = abs(estimation_accuracy - 1.0)

        # Store in matrix
        if sync_label not in sync_levels:
            sync_levels.append(sync_label)
        if noise_label not in noise_levels:
            noise_levels.append(noise_label)

        tpcv_matrix[(sync_label, noise_label)] = tpcv
        error_matrix[(sync_label, noise_label)] = relative_error

    # Sort levels
    sync_levels.sort(key=lambda x: int(x.split()[0]))  # "0 samples", "5 samples", etc.
    noise_levels.sort(key=lambda x: float(x.split('x')[0]))  # "0.1x baseline", etc.

    # Create CV matrix
    tpcv_array = np.zeros((len(noise_levels), len(sync_levels)))
    error_array = np.zeros((len(noise_levels), len(sync_levels)))
    for i, noise in enumerate(noise_levels):
        for j, sync in enumerate(sync_levels):
            tpcv_array[i, j] = tpcv_matrix.get((sync, noise), np.nan)
            error_array[i, j] = error_matrix.get((sync, noise), np.nan)

    return {
        'sync_levels': sync_levels,
        'noise_levels': noise_levels,
        'tpcv_matrix': tpcv_array,
        'error_matrix': error_array,  # New: relative error matrix
        'conditions': conditions
    }


def extract_sync_time_us(sync_label):
    """Extract time in microseconds from sync label like '5 samples'"""
    samples = int(sync_label.split()[0])
    # E04 uses 2000 Hz sampling rate, so 1 sample = 0.5 ms = 500 us
    return samples * 500


def extract_noise_multiplier(noise_label):
    """Extract multiplier from noise label like '1.2x baseline'"""
    return float(noise_label.split('x')[0])


def generate_f05_interaction_heatmap(data, output_path, params=None):
    """
    F05: figure11_tpcv_interaction_heatmap.pdf
    2D heatmap showing CV values and relative errors across sync error and noise levels
    Now generates two separate panels and combines them using combine_figures

    Args:
        data: Processed E04 data from load_e04_data()
        output_path: Path to save PNG figure
        params: Experimental parameters dict (optional, for JSON metadata)
    """
    from combine_figures import combine_figures

    # Apply IEEE style with Chinese font support
    if SCIENCEPLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])

    # Force serif font (IEEE requirement, must be set after style.use to override defaults)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

    # Configure Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Apply font size standard
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
    })

    sync_levels = data['sync_levels']
    noise_levels = data['noise_levels']
    tpcv_matrix = data['tpcv_matrix']
    error_matrix = data['error_matrix']

    # Convert labels to numeric values for axes
    sync_times_us = [extract_sync_time_us(s) for s in sync_levels]
    noise_multipliers = [extract_noise_multiplier(n) for n in noise_levels]

    # Filter data: only keep sync_times_us <= 15000 portion
    sync_limit_us = 15000
    sync_indices = np.array([i for i, t in enumerate(sync_times_us) if t <= sync_limit_us])
    sync_times_us_filtered = [sync_times_us[i] for i in sync_indices]
    tpcv_matrix_filtered = tpcv_matrix[:, sync_indices]
    error_matrix_filtered = error_matrix[:, sync_indices]

    # Calculate cell centers for proper text positioning
    n_sync = len(sync_times_us_filtered)
    n_noise = len(noise_multipliers)
    sync_step = (sync_times_us_filtered[-1] - sync_times_us_filtered[0]) / n_sync if n_sync > 1 else 1
    noise_step = (noise_multipliers[-1] - noise_multipliers[0]) / n_noise

    # Create meshgrid for contours
    X, Y = np.meshgrid(sync_times_us_filtered, noise_multipliers)

    # Panel A: TPCV Heatmap
    from matplotlib.colors import LinearSegmentedColormap
    # Unified blue to red colormap (Blue to Red)
    colors = ['#0d47a1', '#1976d2', '#42a5f5', '#81c784', '#ffd54f', '#ff9800', '#f44336']
    n_bins = 100
    cmap_cv = LinearSegmentedColormap.from_list('cv_quality', colors, N=n_bins)

    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 6))

    # Plot CV heatmap - modified vmax to 0.3, x-axis limited to 15000μs
    im1 = ax1.imshow(tpcv_matrix_filtered, cmap=cmap_cv, aspect='auto',
                     vmin=0.0, vmax=0.3, origin='lower',
                     extent=[sync_times_us_filtered[0], sync_times_us_filtered[-1],
                            noise_multipliers[0], noise_multipliers[-1]])

    # Add colorbar for TPCV
    cbar1 = plt.colorbar(im1, ax=ax1, label='TPCV Value', shrink=0.75, aspect=20)
    cbar1.ax.tick_params(labelsize=10)
    cbar1.set_label('TPCV Value', fontsize=11)
    cbar1.ax.axhline(y=0.10, color='white', linestyle='--', linewidth=0.8, alpha=0.8)
    cbar1.ax.axhline(y=0.15, color='white', linestyle='--', linewidth=0.8, alpha=0.8)
    cbar1.ax.axhline(y=0.20, color='white', linestyle='-', linewidth=1.2, alpha=0.9)

    # Annotate grid points with CV values
    for i in range(n_noise):
        for j in range(n_sync):
            tpcv_val = tpcv_matrix_filtered[i, j]
            if not np.isnan(tpcv_val):
                x_center = sync_times_us_filtered[0] + (j + 0.5) * sync_step
                y_center = noise_multipliers[0] + (i + 0.5) * noise_step
                color = 'white' if tpcv_val > 0.25 else 'black'
                ax1.text(x_center, y_center, f'{tpcv_val:.2f}',
                        ha='center', va='center', fontsize=5.5, color=color, weight='normal')

    # Add contour line at CV = 0.20
    CS1 = ax1.contour(X, Y, tpcv_matrix_filtered, levels=[0.20], colors='black', linewidths=2.0)
    ax1.clabel(CS1, inline=True, fontsize=10, fmt='TPCV=0.20')

    # Labels (no title)
    ax1.set_xlabel(r'Time Synchronization Error ($\mu$s)', fontsize=12)
    ax1.set_ylabel('Background Noise Level (× baseline)', fontsize=12)

    # Save panel A
    panel_a_path = output_path.parent / 'figure11_panel_a.png'
    plt.savefig(panel_a_path, format='png', dpi=600, bbox_inches='tight')
    plt.close(fig1)
    print(f"Generated panel A: {panel_a_path}")

    # Panel B: Relative Error Heatmap
    # Unified use of same blue to red colormap as Panel A
    cmap_error = cmap_cv  # Use same colormap as TPCV

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))

    # Plot relative error heatmap - modified vmax to 0.3, x-axis limited to 15000μs
    im2 = ax2.imshow(error_matrix_filtered, cmap=cmap_error, aspect='auto',
                     vmin=0.0, vmax=1.0, origin='lower',
                     extent=[sync_times_us_filtered[0], sync_times_us_filtered[-1],
                            noise_multipliers[0], noise_multipliers[-1]])

    # Add colorbar for relative error
    cbar2 = plt.colorbar(im2, ax=ax2, label='Relative Error', shrink=0.75, aspect=20)
    cbar2.ax.tick_params(labelsize=10)
    cbar2.set_label('Relative Error', fontsize=11)
    cbar2.ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.8)
    cbar2.ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.9)

    # Annotate grid points with relative error values
    for i in range(n_noise):
        for j in range(n_sync):
            error_val = error_matrix_filtered[i, j]
            if not np.isnan(error_val):
                x_center = sync_times_us_filtered[0] + (j + 0.5) * sync_step
                y_center = noise_multipliers[0] + (i + 0.5) * noise_step
                color = 'white' if error_val > 1.0 else 'black'
                ax2.text(x_center, y_center, f'{error_val:.2f}',
                        ha='center', va='center', fontsize=5.5, color=color, weight='normal')

    # Add contour line at error = 0.5
    CS2 = ax2.contour(X, Y, error_matrix_filtered, levels=[0.5], colors='black', linewidths=2.0)
    ax2.clabel(CS2, inline=True, fontsize=10, fmt='Error=0.5')

    # Labels (no title)
    ax2.set_xlabel(r'Time Synchronization Error ($\mu$s)', fontsize=12)
    ax2.set_ylabel('Background Noise Level (× baseline)', fontsize=12)

    # Save panel B
    panel_b_path = output_path.parent / 'figure11_panel_b.png'
    plt.savefig(panel_b_path, format='png', dpi=600, bbox_inches='tight')
    plt.close(fig2)
    print(f"Generated panel B: {panel_b_path}")

    # Combine panels
    combine_figures(
        input_files=[panel_a_path, panel_b_path],
        output_file=output_path,
        layout=(1, 2),
        labels=['a', 'b'],
        label_position='bottom_center',
        spacing=0.04,
        dpi=600
    )
    print(f"Generated combined figure: {output_path}")

    # Save JSON data
    json_path = output_path.with_suffix('.json')
    json_data = {
        'metadata': {
            'figure_name': 'figure11_tpcv_interaction_heatmap',
            'experiment_id': params.get('experiment_id', 'E04-combined_effects') if params else 'E04-combined_effects',
            'generated_timestamp': datetime.now().isoformat(),
            'source_timestamp': params.get('timestamp', '') if params else '',
            'description': 'TPCV and Relative Error interaction heatmap showing dual-factor effects',
        },
        'experimental_parameters': params if params else {},
        'plot_data': {
            'sync_levels': {
                'labels': sync_levels,
                'time_us': sync_times_us,
                'description': 'Time synchronization error levels'
            },
            'noise_levels': {
                'labels': noise_levels,
                'multipliers': noise_multipliers,
                'description': 'Background noise level multipliers'
            },
            'tpcv_matrix': {
                'data': tpcv_matrix.tolist(),
                'shape': list(tpcv_matrix.shape),
                'vmin': 0.0,
                'vmax': 0.2,
                'description': 'Coefficient of Variation values'
            },
            'error_matrix': {
                'data': error_matrix.tolist(),
                'shape': list(error_matrix.shape),
                'vmin': 0.0,
                'vmax': 0.3,
                'description': 'Relative error values |estimation_accuracy - 1.0|'
            },
            'quality_thresholds': {
                'tpcv_excellent': 0.10,
                'tpcv_good': 0.15,
                'tpcv_threshold': 0.20,
                'error_acceptable': 0.05,
                'error_threshold': 0.10
            }
        },
        'contour_lines': {
            'tpcv_threshold': 0.20,
            'error_threshold': 0.5,
            'description': 'Contour lines marking quality boundaries'
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Generated JSON: {json_path}")


def generate_f06_main_effects(data, output_path, params=None):
    """
    F06: figure12_main_effects.pdf
    Two subplots showing main effects of sync error and noise level
    Now generates two separate panels and combines them using combine_figures
    """
    from combine_figures import combine_figures

    # Apply IEEE style with Chinese font support
    if SCIENCEPLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])

    # Force serif font (IEEE requirement, must be set after style.use to override defaults)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

    # Configure Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Apply IEEE font size standard
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
    })

    sync_levels = data['sync_levels']
    noise_levels = data['noise_levels']
    tpcv_matrix = data['tpcv_matrix']

    sync_times_us = [extract_sync_time_us(s) for s in sync_levels]
    noise_multipliers = [extract_noise_multiplier(n) for n in noise_levels]

    # Panel A: Main effect of synchronization error (averaged over all noise levels)
    sync_main_effect = np.mean(tpcv_matrix, axis=0)
    sync_std = np.std(tpcv_matrix, axis=0)

    fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.5))

    ax1.plot(sync_times_us, sync_main_effect, 'o-', linewidth=1.2, markersize=3,
            color='#1976d2', label='Mean TPCV')
    ax1.fill_between(sync_times_us,
                     sync_main_effect - sync_std,
                     sync_main_effect + sync_std,
                     alpha=0.2, color='#1976d2', label='±1 SD')
    ax1.axhline(y=0.20, color='red', linestyle='--', linewidth=1, label='Threshold (0.20)')
    ax1.axhline(y=0.10, color='green', linestyle='--', linewidth=0.8, label='Excellent (0.10)')

    ax1.set_xlabel(r'Time Synchronization Error ($\mu$s)')
    ax1.set_ylabel('TPCV Value')
    ax1.legend(loc='upper left', frameon=True, fontsize=6.5, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # Save panel A
    panel_a_path = output_path.parent / 'figure12_panel_a.png'
    plt.savefig(panel_a_path, format='png', dpi=600, bbox_inches='tight')
    plt.close(fig1)
    print(f"Generated panel A: {panel_a_path}")

    # Panel B: Main effect of noise level (averaged over all sync errors)
    noise_main_effect = np.mean(tpcv_matrix, axis=1)
    noise_std = np.std(tpcv_matrix, axis=1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(3.5, 2.5))

    ax2.plot(noise_multipliers, noise_main_effect, 's-', linewidth=1.2, markersize=3,
            color='#f57c00', label='Mean TPCV')
    ax2.fill_between(noise_multipliers,
                     noise_main_effect - noise_std,
                     noise_main_effect + noise_std,
                     alpha=0.2, color='#f57c00', label='±1 SD')
    ax2.axhline(y=0.20, color='red', linestyle='--', linewidth=1, label='Threshold (0.20)')
    ax2.axhline(y=0.10, color='green', linestyle='--', linewidth=0.8, label='Excellent (0.10)')

    ax2.set_xlabel('Background Noise Level (× baseline)')
    ax2.set_ylabel('TPCV Value')
    ax2.legend(loc='upper left', frameon=True, fontsize=6.5, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linewidth=0.5)

    # Save panel B
    panel_b_path = output_path.parent / 'figure12_panel_b.png'
    plt.savefig(panel_b_path, format='png', dpi=600, bbox_inches='tight')
    plt.close(fig2)
    print(f"Generated panel B: {panel_b_path}")

    # Combine panels
    combine_figures(
        input_files=[panel_a_path, panel_b_path],
        output_file=output_path,
        layout=(1, 2),
        labels=['a', 'b'],
        label_position='bottom_center',
        spacing=0.04,
        dpi=600
    )
    print(f"Generated combined figure: {output_path}")

    # Save JSON data
    json_path = output_path.with_suffix('.json')
    json_data = {
        'metadata': {
            'figure_name': 'figure12_main_effects',
            'experiment_id': params.get('experiment_id', 'E04-combined_effects') if params else 'E04-combined_effects',
            'generated_timestamp': datetime.now().isoformat(),
            'source_timestamp': params.get('timestamp', '') if params else '',
            'description': 'Main effects analysis of sync error and noise level on CV',
        },
        'experimental_parameters': params if params else {},
        'plot_data': {
            'panel_a_sync_effect': {
                'x_axis': {
                    'values': sync_times_us,
                    'unit': 'microseconds',
                    'description': 'Time synchronization error'
                },
                'y_axis': {
                    'mean': sync_main_effect.tolist(),
                    'std': sync_std.tolist(),
                    'unit': 'CV value',
                    'description': 'Mean TPCV averaged over all noise levels'
                }
            },
            'panel_b_noise_effect': {
                'x_axis': {
                    'values': noise_multipliers,
                    'unit': 'multiplier',
                    'description': 'Background noise level (× baseline)'
                },
                'y_axis': {
                    'mean': noise_main_effect.tolist(),
                    'std': noise_std.tolist(),
                    'unit': 'CV value',
                    'description': 'Mean TPCV averaged over all sync errors'
                }
            },
            'quality_thresholds': {
                'tpcv_excellent': 0.10,
                'tpcv_threshold': 0.20
            }
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Generated JSON: {json_path}")


def generate_f07_interaction_plot(data, output_path, params=None):
    """
    F07: figure13_interaction_plot.pdf
    Multi-line plot showing interaction effects at 4 representative noise levels
    """
    # Apply IEEE style with Chinese font support
    if SCIENCEPLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])

    # Force serif font (IEEE requirement, must be set after style.use to override defaults)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

    # Configure Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Apply IEEE font size standard
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 6.5,  # Reduced for multi-line legend
        'figure.constrained_layout.use': True
    })

    sync_levels = data['sync_levels']
    noise_levels = data['noise_levels']
    tpcv_matrix = data['tpcv_matrix']

    sync_times_us = [extract_sync_time_us(s) for s in sync_levels]
    noise_multipliers = [extract_noise_multiplier(n) for n in noise_levels]

    # Select 4 representative noise levels
    # Low, medium-low, medium-high, high
    indices = [0, 3, 6, 9]  # 0.1x, 3.4x, 6.7x, 10.0x
    colors = ['#1976d2', '#66bb6a', '#ffa726', '#ef5350']

    # IEEE single column width
    fig, ax = plt.subplots(figsize=(3.5, 2.625))

    for idx, color in zip(indices, colors):
        noise_label = noise_levels[idx]
        tpcv_values = tpcv_matrix[idx, :]
        ax.plot(sync_times_us, tpcv_values, 'o-', linewidth=1.0, markersize=2.5,
               color=color, label=f'{noise_label}')

    # Add threshold line
    ax.axhline(y=0.20, color='red', linestyle='--', linewidth=1,
              label='Threshold (0.20)', zorder=10)
    ax.axhline(y=0.10, color='green', linestyle='--', linewidth=0.8,
              label='Excellent (0.10)', zorder=10)

    ax.set_xlabel(r'Time Synchronization Error ($\mu$s)')
    ax.set_ylabel('TPCV Value')
    # ax.set_title('Interaction Effect - TPCV vs Sync Error')  # Commented: Remove figure title

    # Optimize legend: place outside plot area to avoid overlap
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
             ncol=1, frameon=True, fontsize=6.5, framealpha=0.95,
             title='Noise Level', title_fontsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Add annotation for interaction pattern - repositioned to avoid legend
    ax.text(0.02, 0.98, 'Synergistic\nDegradation',
           transform=ax.transAxes, fontsize=6.5,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5, edgecolor='gray', linewidth=0.5),
           verticalalignment='top', horizontalalignment='left')

    # Use tight bbox to include legend
    plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

    # Save JSON data
    json_path = output_path.with_suffix('.json')

    # Build line data for each selected noise level
    lines_data = []
    for idx, color in zip(indices, colors):
        noise_label = noise_levels[idx]
        tpcv_values = tpcv_matrix[idx, :]
        lines_data.append({
            'noise_level_index': int(idx),
            'noise_level_label': noise_label,
            'noise_multiplier': noise_multipliers[idx],
            'line_color': color,
            'data_points': {
                'x': sync_times_us,
                'y': tpcv_values.tolist()
            }
        })

    json_data = {
        'metadata': {
            'figure_name': 'figure13_interaction_plot',
            'experiment_id': params.get('experiment_id', 'E04-combined_effects') if params else 'E04-combined_effects',
            'generated_timestamp': datetime.now().isoformat(),
            'source_timestamp': params.get('timestamp', '') if params else '',
            'description': 'Interaction effect showing TPCV vs sync error at 4 representative noise levels',
        },
        'experimental_parameters': params if params else {},
        'plot_data': {
            'x_axis': {
                'values': sync_times_us,
                'unit': 'microseconds',
                'description': 'Time synchronization error'
            },
            'lines': lines_data,
            'quality_thresholds': {
                'tpcv_excellent': 0.10,
                'tpcv_threshold': 0.20
            },
            'annotation': 'Synergistic Degradation - higher noise + higher sync error produces disproportionate CV increase'
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Generated JSON: {json_path}")


def generate_f05b_tpcv_vs_error_scatter(data, output_path, params=None):
    """
    F05b: TPCV vs Relative Error - Main Effect Analysis
    Shows the main effect of CV on measurement relative error using all 100 data points
    Using main effect plot style (similar to F06, single curve with error band)
    """
    # Apply IEEE style with Chinese font support
    if SCIENCEPLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])

    # Force serif font (IEEE requirement, must be set after style.use to override defaults)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

    # Configure Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Apply IEEE font size standard
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 6.5,  # Reduced for better layout
        'figure.constrained_layout.use': True
    })

    tpcv_matrix = data['tpcv_matrix']
    error_matrix = data['error_matrix']

    # Flatten all data (100 points)
    tpcv_all = tpcv_matrix.flatten()
    error_all = error_matrix.flatten()

    # Remove NaN values
    valid_mask = ~(np.isnan(tpcv_all) | np.isnan(error_all))
    tpcv_valid = tpcv_all[valid_mask]
    error_valid = error_all[valid_mask]

    # Bin CV values into intervals for main effect analysis
    # Use 25 bins for balanced resolution
    n_bins = 25
    tpcv_min, tpcv_max = tpcv_valid.min(), tpcv_valid.max()
    tpcv_bins = np.linspace(tpcv_min, tpcv_max, n_bins + 1)
    tpcv_bin_centers = (tpcv_bins[:-1] + tpcv_bins[1:]) / 2

    # Calculate mean and std of Error for each CV bin
    error_means = []
    error_stds = []

    for i in range(n_bins):
        # Find points in this CV bin
        mask = (tpcv_valid >= tpcv_bins[i]) & (tpcv_valid < tpcv_bins[i+1])
        if i == n_bins - 1:  # Last bin includes upper boundary
            mask = (tpcv_valid >= tpcv_bins[i]) & (tpcv_valid <= tpcv_bins[i+1])

        error_in_bin = error_valid[mask]

        if len(error_in_bin) > 0:
            error_means.append(np.mean(error_in_bin))
            error_stds.append(np.std(error_in_bin))
        else:
            error_means.append(np.nan)
            error_stds.append(np.nan)

    error_means = np.array(error_means)
    error_stds = np.array(error_stds)

    # Remove NaN bins
    valid_bins = ~np.isnan(error_means)
    tpcv_bin_centers = tpcv_bin_centers[valid_bins]
    error_means = error_means[valid_bins]
    error_stds = error_stds[valid_bins]

    # Create figure - increased size to make legend less obstructive
    fig, ax = plt.subplots(figsize=(5, 3.75))

    # Plot main effect curve with error band (similar to F06 style)
    ax.plot(tpcv_bin_centers, error_means, 'o-', linewidth=1.2, markersize=3,
            color='#1976d2', label='Mean Relative Error')
    ax.fill_between(tpcv_bin_centers,
                     error_means - error_stds,
                     error_means + error_stds,
                     alpha=0.2, color='#1976d2', label='±1 SD')

    # Add threshold lines (without labels to reduce legend size)
    ax.axvline(x=0.10, color='green', linestyle='--', linewidth=0.8, alpha=0.8)
    ax.axvline(x=0.20, color='red', linestyle='--', linewidth=1, alpha=0.8)
    ax.axhline(y=0.05, color='blue', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(y=0.10, color='orange', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel('TPCV (Two-Path Coefficient of Variation)')
    ax.set_ylabel('Relative Error')
    # ax.set_title('Main Effect of TPCV on Relative Error')  # Commented: Remove figure title

    # Simplified legend: only data series
    ax.legend(loc='upper left', frameon=True, fontsize=6,
             framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Add threshold line annotations directly on plot
    ax.text(0.10, ax.get_ylim()[1] * 0.95, 'TPCV=0.10', fontsize=5.5,
           color='green', ha='left', va='top', rotation=90, alpha=0.9)
    ax.text(0.20, ax.get_ylim()[1] * 0.95, 'TPCV=0.20', fontsize=5.5,
           color='red', ha='left', va='top', rotation=90, alpha=0.9)

    # Add horizontal threshold annotations
    ax.text(ax.get_xlim()[1] * 0.98, 0.05, 'Err=0.05', fontsize=5.5,
           color='blue', ha='right', va='bottom', alpha=0.9)
    ax.text(ax.get_xlim()[1] * 0.98, 0.10, 'Err=0.10', fontsize=5.5,
           color='orange', ha='right', va='bottom', alpha=0.9)

    # Add annotation with data statistics - repositioned to avoid legend
    text_str = f'n={len(tpcv_valid)} | bins={n_bins}\n'
    text_str += f'TPCV: {tpcv_min:.3f}-{tpcv_max:.3f}'

    ax.text(0.98, 0.02, text_str,
           transform=ax.transAxes, fontsize=5.5,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='gray', linewidth=0.5),
           verticalalignment='bottom', horizontalalignment='right',
           family='monospace')

    # Use constrained_layout for better spacing
    plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

    # Save JSON data
    json_path = output_path.with_suffix('.json')
    json_data = {
        'metadata': {
            'figure_name': 'figure11b_tpcv_vs_error_scatter',
            'experiment_id': params.get('experiment_id', 'E04-combined_effects') if params else 'E04-combined_effects',
            'generated_timestamp': datetime.now().isoformat(),
            'source_timestamp': params.get('timestamp', '') if params else '',
            'description': 'Main effect of TPCV on relative error (binned analysis with error bands)',
        },
        'experimental_parameters': params if params else {},
        'plot_data': {
            'raw_data': {
                'cv_all_points': tpcv_valid.tolist(),
                'error_all_points': error_valid.tolist(),
                'total_points': len(tpcv_valid)
            },
            'binned_analysis': {
                'n_bins': n_bins,
                'tpcv_bin_centers': tpcv_bin_centers.tolist(),
                'error_means': error_means.tolist(),
                'error_stds': error_stds.tolist(),
                'cv_range': [float(tpcv_min), float(tpcv_max)],
                'error_range': [float(error_valid.min()), float(error_valid.max())]
            },
            'quality_thresholds': {
                'tpcv_excellent': 0.10,
                'tpcv_threshold': 0.20,
                'error_acceptable': 0.05,
                'error_threshold': 0.10
            }
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Generated JSON: {json_path}")


def generate_f05c_error_vs_tpcv_scatter(data, output_path, params=None):
    """
    F05c: Error vs CV - Two-panel plot (full range + zoomed view)
    Panel A: Full range showing all data
    Panel B: Zoomed view for Relative Error < 1.5
    """
    from combine_figures import combine_figures

    # Apply IEEE style with Chinese font support
    if SCIENCEPLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])

    # Force serif font (IEEE requirement, must be set after style.use to override defaults)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

    # Configure Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Apply IEEE font size standard
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 6.5,
        'figure.constrained_layout.use': True
    })

    tpcv_matrix = data['tpcv_matrix']
    error_matrix = data['error_matrix']

    # Flatten all data (100 points)
    tpcv_all = tpcv_matrix.flatten()
    error_all = error_matrix.flatten()

    # Remove NaN values
    valid_mask = ~(np.isnan(tpcv_all) | np.isnan(error_all))
    tpcv_valid = tpcv_all[valid_mask]
    error_valid = error_all[valid_mask]

    # Helper function to compute binned statistics
    def compute_bins(error_data, tpcv_data, n_bins=25):
        """Compute binned mean and std for Error vs TPCV"""
        error_min, error_max = error_data.min(), error_data.max()
        error_bins = np.linspace(error_min, error_max, n_bins + 1)
        error_bin_centers = (error_bins[:-1] + error_bins[1:]) / 2

        tpcv_means = []
        tpcv_stds = []

        for i in range(n_bins):
            mask = (error_data >= error_bins[i]) & (error_data < error_bins[i+1])
            if i == n_bins - 1:
                mask = (error_data >= error_bins[i]) & (error_data <= error_bins[i+1])

            tpcv_in_bin = tpcv_data[mask]

            if len(tpcv_in_bin) > 0:
                tpcv_means.append(np.mean(tpcv_in_bin))
                tpcv_stds.append(np.std(tpcv_in_bin))
            else:
                tpcv_means.append(np.nan)
                tpcv_stds.append(np.nan)

        tpcv_means = np.array(tpcv_means)
        tpcv_stds = np.array(tpcv_stds)

        # Remove NaN bins
        valid_bins = ~np.isnan(tpcv_means)
        error_bin_centers = error_bin_centers[valid_bins]
        tpcv_means = tpcv_means[valid_bins]
        tpcv_stds = tpcv_stds[valid_bins]

        return error_bin_centers, tpcv_means, tpcv_stds, error_min, error_max

    # Helper function to plot a single panel
    def plot_panel(ax, error_centers, tpcv_means, tpcv_stds, error_min, error_max,
                   tpcv_data, error_data, n_bins, xlim=None):
        """Plot a single Error vs TPCV panel"""
        # Plot main effect curve
        ax.plot(error_centers, tpcv_means, 'o-', linewidth=1.2, markersize=3,
                color='#d32f2f', label='Mean TPCV')
        ax.fill_between(error_centers,
                        tpcv_means - tpcv_stds,
                        tpcv_means + tpcv_stds,
                        alpha=0.2, color='#d32f2f', label='±1 SD')

        # Threshold lines
        ax.axhline(y=0.10, color='green', linestyle='--', linewidth=0.8, alpha=0.8)
        ax.axhline(y=0.20, color='red', linestyle='--', linewidth=1, alpha=0.8)

        # Find intersections
        error_at_cv010 = None
        error_at_cv020 = None
        for i in range(len(tpcv_means) - 1):
            if error_at_cv010 is None and (tpcv_means[i] <= 0.10 and tpcv_means[i+1] >= 0.10):
                t = (0.10 - tpcv_means[i]) / (tpcv_means[i+1] - tpcv_means[i])
                error_at_cv010 = error_centers[i] + t * (error_centers[i+1] - error_centers[i])
            if error_at_cv020 is None and (tpcv_means[i] <= 0.20 and tpcv_means[i+1] >= 0.20):
                t = (0.20 - tpcv_means[i]) / (tpcv_means[i+1] - tpcv_means[i])
                error_at_cv020 = error_centers[i] + t * (error_centers[i+1] - error_centers[i])

        # Mark intersections
        if error_at_cv010 is not None:
            ax.plot(error_at_cv010, 0.10, 'o', color='green', markersize=6, zorder=10)
        if error_at_cv020 is not None:
            ax.plot(error_at_cv020, 0.20, 'o', color='red', markersize=6, zorder=10)

        # Styling
        ax.set_xlabel('Relative Error')
        ax.set_ylabel('TPCV (Two-Path Coefficient of Variation)')
        ax.set_xlim(left=0)
        if xlim is not None:
            ax.set_xlim(right=xlim)

        ax.legend(loc='upper left', frameon=True, fontsize=6, framealpha=0.95)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Threshold annotations
        xmax = xlim if xlim is not None else ax.get_xlim()[1]
        ax.text(xmax * 0.98, 0.10, 'TPCV=0.10', fontsize=5.5,
               color='green', ha='right', va='bottom', alpha=0.9)
        ax.text(xmax * 0.98, 0.20, 'TPCV=0.20', fontsize=5.5,
               color='red', ha='right', va='bottom', alpha=0.9)

        return error_at_cv010, error_at_cv020

    # Panel A: Full range
    n_bins_full = 15
    error_centers_full, tpcv_means_full, tpcv_stds_full, error_min_full, error_max_full = \
        compute_bins(error_valid, tpcv_valid, n_bins_full)

    fig1, ax1 = plt.subplots(figsize=(5, 3.75))
    error_at_cv010_full, error_at_cv020_full = plot_panel(
        ax1, error_centers_full, tpcv_means_full, tpcv_stds_full,
        error_min_full, error_max_full, tpcv_valid, error_valid, n_bins_full
    )

    # Save Panel A
    panel_a_path = output_path.parent / 'figure11c_panel_a.png'
    plt.savefig(panel_a_path, format='png', dpi=600, bbox_inches='tight')
    plt.close(fig1)
    print(f"Generated panel A: {panel_a_path}")

    # Panel B: Zoomed (Error < 2.0)
    n_bins_zoom = 10  # Increased to ensure data point near origin
    zoom_mask = error_valid < 2.0
    error_zoom = error_valid[zoom_mask]
    tpcv_zoom = tpcv_valid[zoom_mask]

    error_centers_zoom, tpcv_means_zoom, tpcv_stds_zoom, error_min_zoom, error_max_zoom = \
        compute_bins(error_zoom, tpcv_zoom, n_bins_zoom)

    fig2, ax2 = plt.subplots(figsize=(5, 3.75))
    error_at_cv010_zoom, error_at_cv020_zoom = plot_panel(
        ax2, error_centers_zoom, tpcv_means_zoom, tpcv_stds_zoom,
        error_min_zoom, error_max_zoom, tpcv_zoom, error_zoom, n_bins_zoom, xlim=2.0
    )

    # Save Panel B
    panel_b_path = output_path.parent / 'figure11c_panel_b.png'
    plt.savefig(panel_b_path, format='png', dpi=600, bbox_inches='tight')
    plt.close(fig2)
    print(f"Generated panel B: {panel_b_path}")

    # Combine panels
    combine_figures(
        input_files=[panel_a_path, panel_b_path],
        output_file=output_path,
        layout=(1, 2),
        labels=['a', 'b'],
        label_position='bottom_center',
        spacing=0.04,
        dpi=600
    )
    print(f"Generated combined figure: {output_path}")

    # Save JSON data
    json_path = output_path.with_suffix('.json')
    json_data = {
        'metadata': {
            'figure_name': 'figure11c_error_vs_tpcv_scatter',
            'experiment_id': params.get('experiment_id', 'E04-combined_effects') if params else 'E04-combined_effects',
            'generated_timestamp': datetime.now().isoformat(),
            'source_timestamp': params.get('timestamp', '') if params else '',
            'description': 'Two-panel plot: full range and zoomed view (Error<2.0) of Error vs TPCV',
        },
        'experimental_parameters': params if params else {},
        'plot_data': {
            'panel_a_full_range': {
                'n_bins': n_bins_full,
                'error_bin_centers': error_centers_full.tolist(),
                'tpcv_means': tpcv_means_full.tolist(),
                'tpcv_stds': tpcv_stds_full.tolist(),
                'error_range': [float(error_min_full), float(error_max_full)],
                'n_points': len(tpcv_valid),
                'intersections': {
                    'error_at_cv010': float(error_at_cv010_full) if error_at_cv010_full is not None else None,
                    'error_at_cv020': float(error_at_cv020_full) if error_at_cv020_full is not None else None,
                }
            },
            'panel_b_zoomed': {
                'n_bins': n_bins_zoom,
                'error_bin_centers': error_centers_zoom.tolist(),
                'tpcv_means': tpcv_means_zoom.tolist(),
                'tpcv_stds': tpcv_stds_zoom.tolist(),
                'error_range': [float(error_min_zoom), float(error_max_zoom)],
                'error_limit': 2.0,
                'n_points': len(tpcv_zoom),
                'intersections': {
                    'error_at_cv010': float(error_at_cv010_zoom) if error_at_cv010_zoom is not None else None,
                    'error_at_cv020': float(error_at_cv020_zoom) if error_at_cv020_zoom is not None else None,
                }
            },
            'quality_thresholds': {
                'tpcv_excellent': 0.10,
                'tpcv_threshold': 0.20
            }
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Generated JSON: {json_path}")


def generate_f05c_error_vs_tpcv_scatter_with_cave(data, output_path, params=None):
    """
    F05c with Cave Test Data: Single panel zoomed view with cave experiment results
    Shows multiple TPCV metrics (mean/median) and relative error baselines (mean/median/min/max)

    Cave test data (from main.tex):
    - Pre-compensation: TPCV 29.6%, 61.0%, 26.4%, noise 6.89, 5.22, 5.91 ng/√Hz
    - Post-compensation: TPCV 1.7%, 11.9%, 1.2%, noise 2.96, 2.58, 4.65 ng/√Hz
    """

    # Apply IEEE style with Chinese font support
    if SCIENCEPLOTS_AVAILABLE:
        plt.style.use(['science', 'ieee'])

    # Force serif font (IEEE requirement, must be set after style.use to override defaults)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

    # Configure Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Apply IEEE font size standard
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 6,
        'figure.constrained_layout.use': True
    })

    tpcv_matrix = data['tpcv_matrix']
    error_matrix = data['error_matrix']

    # Flatten all data (100 points)
    tpcv_all = tpcv_matrix.flatten()
    error_all = error_matrix.flatten()

    # Remove NaN values
    valid_mask = ~(np.isnan(tpcv_all) | np.isnan(error_all))
    tpcv_valid = tpcv_all[valid_mask]
    error_valid = error_all[valid_mask]

    # Cave test data calculation
    # Pre-compensation data (3 channels)
    cave_pre_noise = np.array([6.89, 5.22, 5.91])  # ng/√Hz @10Hz
    cave_pre_tpcv = np.array([29.6, 61.0, 26.4])   # %

    # Post-compensation data (3 channels)
    cave_post_noise = np.array([2.96, 2.58, 4.65])  # ng/√Hz @10Hz
    cave_post_tpcv = np.array([1.7, 11.9, 1.2])     # %

    # Calculate TPCV statistics (convert to decimal)
    cave_pre_tpcv_mean = np.mean(cave_pre_tpcv) / 100
    cave_pre_tpcv_median = np.median(cave_pre_tpcv) / 100
    cave_post_tpcv_mean = np.mean(cave_post_tpcv) / 100
    cave_post_tpcv_median = np.median(cave_post_tpcv) / 100

    # Calculate noise statistics
    cave_pre_noise_mean = np.mean(cave_pre_noise)
    cave_pre_noise_median = np.median(cave_pre_noise)
    cave_pre_noise_min = np.min(cave_pre_noise)
    cave_pre_noise_max = np.max(cave_pre_noise)

    cave_post_noise_mean = np.mean(cave_post_noise)
    cave_post_noise_median = np.median(cave_post_noise)
    cave_post_noise_min = np.min(cave_post_noise)
    cave_post_noise_max = np.max(cave_post_noise)

    # Define baseline options for relative error calculation
    baseline_options = {
        'mean': cave_post_noise_mean,
        'median': cave_post_noise_median,
        'min': cave_post_noise_min,
        'max': cave_post_noise_max
    }

    # Calculate relative errors for each baseline
    def calc_rel_error(noise_values, baseline):
        return np.median(np.abs(noise_values - baseline) / baseline)

    cave_data_points = []

    # Generate all combinations: 2 TPCV metrics × 4 baselines × 2 compensation states
    for tpcv_type in ['mean', 'median']:
        for baseline_type, baseline in baseline_options.items():
            # Pre-compensation
            tpcv_pre = cave_pre_tpcv_mean if tpcv_type == 'mean' else cave_pre_tpcv_median
            error_pre = calc_rel_error(cave_pre_noise, baseline)
            cave_data_points.append({
                'tpcv': tpcv_pre,
                'error': error_pre,
                'tpcv_type': tpcv_type,
                'baseline_type': baseline_type,
                'state': 'pre'
            })

            # Post-compensation
            tpcv_post = cave_post_tpcv_mean if tpcv_type == 'mean' else cave_post_tpcv_median
            error_post = calc_rel_error(cave_post_noise, baseline)
            cave_data_points.append({
                'tpcv': tpcv_post,
                'error': error_post,
                'tpcv_type': tpcv_type,
                'baseline_type': baseline_type,
                'state': 'post'
            })

    # Helper function to compute binned statistics
    def compute_bins(error_data, tpcv_data, n_bins=25):
        """Compute binned mean and std for Error vs TPCV"""
        error_min, error_max = error_data.min(), error_data.max()
        error_bins = np.linspace(error_min, error_max, n_bins + 1)
        error_bin_centers = (error_bins[:-1] + error_bins[1:]) / 2

        tpcv_means = []
        tpcv_stds = []

        for i in range(n_bins):
            mask = (error_data >= error_bins[i]) & (error_data < error_bins[i+1])
            if i == n_bins - 1:
                mask = (error_data >= error_bins[i]) & (error_data <= error_bins[i+1])

            tpcv_in_bin = tpcv_data[mask]

            if len(tpcv_in_bin) > 0:
                tpcv_means.append(np.mean(tpcv_in_bin))
                tpcv_stds.append(np.std(tpcv_in_bin))
            else:
                tpcv_means.append(np.nan)
                tpcv_stds.append(np.nan)

        tpcv_means = np.array(tpcv_means)
        tpcv_stds = np.array(tpcv_stds)

        # Remove NaN bins
        valid_bins = ~np.isnan(tpcv_means)
        error_bin_centers = error_bin_centers[valid_bins]
        tpcv_means = tpcv_means[valid_bins]
        tpcv_stds = tpcv_stds[valid_bins]

        return error_bin_centers, tpcv_means, tpcv_stds, error_min, error_max

    # Generate single panel (zoomed view)
    # Must match panel B parameters in generate_f05c_error_vs_tpcv_scatter
    n_bins_zoom = 10
    zoom_mask = error_valid < 2.0
    error_zoom = error_valid[zoom_mask]
    tpcv_zoom = tpcv_valid[zoom_mask]

    error_centers_zoom, tpcv_means_zoom, tpcv_stds_zoom, error_min_zoom, error_max_zoom = \
        compute_bins(error_zoom, tpcv_zoom, n_bins_zoom)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot simulation data curve
    ax.plot(error_centers_zoom, tpcv_means_zoom, 'o-', linewidth=1.5, markersize=4,
            color='#d32f2f', label='Simulation Mean', zorder=5)
    ax.fill_between(error_centers_zoom,
                    tpcv_means_zoom - tpcv_stds_zoom,
                    tpcv_means_zoom + tpcv_stds_zoom,
                    alpha=0.2, color='#d32f2f', label='Simulation ±1 SD')

    # Threshold lines
    ax.axhline(y=0.10, color='green', linestyle='--', linewidth=0.8, alpha=0.8)
    ax.axhline(y=0.20, color='red', linestyle='--', linewidth=1, alpha=0.8)

    # Find intersections for reference
    error_at_cv010 = None
    error_at_cv020 = None
    for i in range(len(tpcv_means_zoom) - 1):
        if error_at_cv010 is None and (tpcv_means_zoom[i] <= 0.10 and tpcv_means_zoom[i+1] >= 0.10):
            t = (0.10 - tpcv_means_zoom[i]) / (tpcv_means_zoom[i+1] - tpcv_means_zoom[i])
            error_at_cv010 = error_centers_zoom[i] + t * (error_centers_zoom[i+1] - error_centers_zoom[i])
        if error_at_cv020 is None and (tpcv_means_zoom[i] <= 0.20 and tpcv_means_zoom[i+1] >= 0.20):
            t = (0.20 - tpcv_means_zoom[i]) / (tpcv_means_zoom[i+1] - tpcv_means_zoom[i])
            error_at_cv020 = error_centers_zoom[i] + t * (error_centers_zoom[i+1] - error_centers_zoom[i])

    # Mark threshold intersections
    if error_at_cv010 is not None:
        ax.plot(error_at_cv010, 0.10, 'o', color='green', markersize=6, zorder=10)
    if error_at_cv020 is not None:
        ax.plot(error_at_cv020, 0.20, 'o', color='red', markersize=6, zorder=10)

    # Plot cave test data points (best configuration: median-median)
    # Filter for best configuration: TPCV-median, baseline-median
    best_pre = [p for p in cave_data_points if p['state'] == 'pre' and
                p['tpcv_type'] == 'median' and p['baseline_type'] == 'min'][0]
    best_post = [p for p in cave_data_points if p['state'] == 'post' and
                 p['tpcv_type'] == 'median' and p['baseline_type'] == 'min'][0]

    # Plot pre-compensation point (triangle, blue)
    ax.plot(best_pre['error'], best_pre['tpcv'], '^',
           color='#1565C0', markersize=12, markeredgecolor='black',
           markeredgewidth=1.5, zorder=20, label='Cave Pre-comp.')

    # Plot post-compensation point (star, gold)
    ax.plot(best_post['error'], best_post['tpcv'], '*',
           color='#FFD700', markersize=16, markeredgecolor='black',
           markeredgewidth=1, zorder=20, label='Cave Post-comp.')

    # Add thick dashed line connecting the two points
    ax.plot([best_pre['error'], best_post['error']],
            [best_pre['tpcv'], best_post['tpcv']],
            '--', color='#666666', linewidth=2.5, zorder=15, alpha=0.8)

    # Styling
    ax.set_xlabel('Relative Error', fontsize=9)
    ax.set_ylabel('TPCV (Two-Path Coefficient of Variation)', fontsize=9)
    ax.set_xlim(left=0, right=2.0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Threshold annotations
    ax.text(1.95, 0.10, 'TPCV=0.10', fontsize=6, color='green', ha='right', va='bottom', alpha=0.9)
    ax.text(1.95, 0.20, 'TPCV=0.20', fontsize=6, color='red', ha='right', va='bottom', alpha=0.9)

    # Legend (improved spacing)
    ax.legend(loc='upper left', frameon=True, fontsize=7, framealpha=0.9,
              handlelength=1.5, handletextpad=0.5, labelspacing=1.0, borderpad=0.8)

    # Save figure
    plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated figure with cave data: {output_path}")

    # Analyze best configuration
    # Find which configuration is closest to simulation curve
    analysis_results = []
    for pt in cave_data_points:
        # Find closest simulation bin
        closest_idx = np.argmin(np.abs(error_centers_zoom - pt['error']))
        sim_tpcv = tpcv_means_zoom[closest_idx]
        distance = np.sqrt((pt['error'] - error_centers_zoom[closest_idx])**2 +
                          (pt['tpcv'] - sim_tpcv)**2)
        analysis_results.append({
            'state': pt['state'],
            'tpcv_type': pt['tpcv_type'],
            'baseline_type': pt['baseline_type'],
            'error': pt['error'],
            'tpcv': pt['tpcv'],
            'closest_sim_error': error_centers_zoom[closest_idx],
            'closest_sim_tpcv': sim_tpcv,
            'distance': distance
        })

    # Print analysis
    print("\n" + "="*60)
    print("Analysis: Cave Test vs Simulation Comparison")
    print("="*60)

    # Sort by distance
    analysis_results.sort(key=lambda x: x['distance'])

    print("\nAll configurations (sorted by distance to simulation curve):")
    print("-"*60)
    for i, r in enumerate(analysis_results):
        print(f"{i+1:2d}. {r['state']:4s} | TPCV-{r['tpcv_type']:6s} | base-{r['baseline_type']:6s} | "
              f"Error={r['error']:.3f}, TPCV={r['tpcv']:.3f} | Dist={r['distance']:.4f}")

    # Find best for pre and post
    best_pre = min([r for r in analysis_results if r['state'] == 'pre'], key=lambda x: x['distance'])
    best_post = min([r for r in analysis_results if r['state'] == 'post'], key=lambda x: x['distance'])

    print("\n" + "="*60)
    print("BEST CONFIGURATIONS:")
    print("="*60)
    print(f"\nPre-compensation best match:")
    print(f"  TPCV type: {best_pre['tpcv_type']}, Baseline: {best_pre['baseline_type']}")
    print(f"  Cave: Error={best_pre['error']:.3f}, TPCV={best_pre['tpcv']:.3f}")
    print(f"  Sim:  Error={best_pre['closest_sim_error']:.3f}, TPCV={best_pre['closest_sim_tpcv']:.3f}")
    print(f"  Distance: {best_pre['distance']:.4f}")

    print(f"\nPost-compensation best match:")
    print(f"  TPCV type: {best_post['tpcv_type']}, Baseline: {best_post['baseline_type']}")
    print(f"  Cave: Error={best_post['error']:.3f}, TPCV={best_post['tpcv']:.3f}")
    print(f"  Sim:  Error={best_post['closest_sim_error']:.3f}, TPCV={best_post['closest_sim_tpcv']:.3f}")
    print(f"  Distance: {best_post['distance']:.4f}")
    print("="*60)

    # Save JSON data
    json_path = output_path.with_suffix('.json')
    json_data = {
        'metadata': {
            'figure_name': 'figure11c_error_vs_tpcv_scatter_with_cave',
            'experiment_id': params.get('experiment_id', 'E04-combined_effects') if params else 'E04-combined_effects',
            'generated_timestamp': datetime.now().isoformat(),
            'source_timestamp': params.get('timestamp', '') if params else '',
            'description': 'Single panel zoomed view (Error<1.5) with multiple cave experiment configurations',
        },
        'experimental_parameters': params if params else {},
        'cave_test_data': {
            'raw_data': {
                'pre_compensation': {
                    'tpcv_values_percent': cave_pre_tpcv.tolist(),
                    'noise_values_ng': cave_pre_noise.tolist(),
                },
                'post_compensation': {
                    'tpcv_values_percent': cave_post_tpcv.tolist(),
                    'noise_values_ng': cave_post_noise.tolist(),
                }
            },
            'statistics': {
                'pre_compensation': {
                    'tpcv_mean': float(cave_pre_tpcv_mean),
                    'tpcv_median': float(cave_pre_tpcv_median),
                    'noise_mean': float(cave_pre_noise_mean),
                    'noise_median': float(cave_pre_noise_median),
                    'noise_min': float(cave_pre_noise_min),
                    'noise_max': float(cave_pre_noise_max),
                },
                'post_compensation': {
                    'tpcv_mean': float(cave_post_tpcv_mean),
                    'tpcv_median': float(cave_post_tpcv_median),
                    'noise_mean': float(cave_post_noise_mean),
                    'noise_median': float(cave_post_noise_median),
                    'noise_min': float(cave_post_noise_min),
                    'noise_max': float(cave_post_noise_max),
                }
            },
            'all_configurations': [
                {
                    'state': r['state'],
                    'tpcv_type': r['tpcv_type'],
                    'baseline_type': r['baseline_type'],
                    'error': r['error'],
                    'tpcv': r['tpcv'],
                    'distance_to_sim': r['distance']
                }
                for r in analysis_results
            ],
            'best_configurations': {
                'pre_compensation': {
                    'tpcv_type': best_pre['tpcv_type'],
                    'baseline_type': best_pre['baseline_type'],
                    'error': best_pre['error'],
                    'tpcv': best_pre['tpcv'],
                    'distance': best_pre['distance']
                },
                'post_compensation': {
                    'tpcv_type': best_post['tpcv_type'],
                    'baseline_type': best_post['baseline_type'],
                    'error': best_post['error'],
                    'tpcv': best_post['tpcv'],
                    'distance': best_post['distance']
                }
            }
        },
        'simulation_data': {
            'n_bins': n_bins_zoom,
            'error_bin_centers': error_centers_zoom.tolist(),
            'tpcv_means': tpcv_means_zoom.tolist(),
            'tpcv_stds': tpcv_stds_zoom.tolist(),
            'error_range': [float(error_min_zoom), float(error_max_zoom)],
            'error_limit': 1.5,
            'n_points': len(tpcv_zoom),
            'intersections': {
                'error_at_cv010': float(error_at_cv010) if error_at_cv010 is not None else None,
                'error_at_cv020': float(error_at_cv020) if error_at_cv020 is not None else None,
            }
        },
        'quality_thresholds': {
            'tpcv_excellent': 0.10,
            'tpcv_threshold': 0.20
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Generated JSON: {json_path}")


def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python e04_specialized_visualizer.py <results_path>")
        print("Example: python e04_specialized_visualizer.py exams/E04-combined_effects/output/20251108_021828/results.json")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    # Output directory (same as results)
    output_dir = results_path.parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    print(f"\n=== E04 Specialized Visualizer ===")
    print(f"Loading data from: {results_path}")

    # Load data
    data = load_e04_data(results_path)
    print(f"Loaded {len(data['sync_levels'])} sync levels × {len(data['noise_levels'])} noise levels")

    # Extract experimental parameters for JSON metadata
    params = extract_experimental_parameters(results_path)
    print(f"Extracted experimental parameters from config")

    # Generate figures
    print("\nGenerating figures (with JSON data export)...")

    generate_f05_interaction_heatmap(data, output_dir / 'figure11_tpcv_interaction_heatmap.png', params)
    generate_f05b_tpcv_vs_error_scatter(data, output_dir / 'figure11b_tpcv_vs_error_scatter.png', params)
    generate_f05c_error_vs_tpcv_scatter(data, output_dir / 'figure11c_error_vs_tpcv_scatter.png', params)
    generate_f06_main_effects(data, output_dir / 'figure12_main_effects.png', params)
    generate_f07_interaction_plot(data, output_dir / 'figure13_interaction_plot.png', params)

    print(f"\n=== All E04 figures and JSON files generated successfully ===")
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - 5 PNG figures (600 DPI)")
    print(f"  - 5 JSON data files (with experimental parameters and plot data)")


if __name__ == '__main__':
    main()
