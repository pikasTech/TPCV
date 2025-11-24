#!/usr/bin/env python3
"""
Unified Figure Combination Tool V2 - External Label Version
==========================================================

Features:
1. Combine multiple individual figures into a composite figure
2. Add (a), (b), (c) labels **outside bottom** of each subfigure (no occlusion)
3. Sufficient spacing between subfigures to avoid label overlap
4. No figure title added (paper figure titles go below the figure)

Usage:
------
from combine_figures_v2 import combine_figures

# Method 1: Specify file list and layout
combine_figures(
    input_files=['fig_A.png', 'fig_B.png', 'fig_C.png'],
    output_file='combined.png',
    layout=(1, 3),  # 1 row, 3 columns
    labels=['a', 'b', 'c']
)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Tuple, Optional, Union
import matplotlib
import scienceplots


def setup_matplotlib_style():
    """Set matplotlib style (IEEE format)"""
    # Apply scienceplots IEEE style
    plt.style.use(['science', 'ieee'])

    # Force serif font (IEEE requirement, Times New Roman)
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

    # Configure Chinese font support
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # IEEE style overrides
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['savefig.dpi'] = 300


def combine_figures(
    input_files: List[Union[str, Path]],
    output_file: Union[str, Path],
    layout: Optional[Tuple[int, int]] = None,
    labels: Optional[List[str]] = None,
    label_position: str = 'bottom_center',
    label_fontsize: int = 18,
    label_fontweight: str = "bold",
    label_color: str = 'black',
    label_bbox: bool = False,
    spacing: float = 0.02,
    dpi: int = 300
) -> Path:
    """
    Combine multiple individual figures into a composite figure with labels outside bottom

    Parameters
    ----------
    input_files : List[Union[str, Path]]
        List of input figure files
    output_file : Union[str, Path]
        Output file path
    layout : Optional[Tuple[int, int]], optional
        Layout (rows, columns), auto-calculated if None, default is None
    labels : Optional[List[str]], optional
        Label list, auto-generated as ['a', 'b', 'c', ...] if None, default is None
    label_position : str, optional
        Label position (parameter retained, currently only supports bottom_center)
    label_fontsize : int, optional
        Label font size, default is 12
    label_fontweight : str, optional
        Label font weight, default is 'normal'
    label_color : str, optional
        Label color, default is 'black'
    label_bbox : bool, optional
        Whether to add white background box to labels, default is False (no box to avoid occlusion)
    spacing : float, optional
        Spacing ratio between subfigures, default is 0.02 (2%)
    dpi : int, optional
        Output DPI, default is 300

    Returns
    -------
    Path
        Output file path
    """
    # Convert to Path objects
    input_files = [Path(f) for f in input_files]
    output_file = Path(output_file)

    # Check input files
    for f in input_files:
        if not f.exists():
            raise FileNotFoundError(f"Input file not found: {f}")

    n_images = len(input_files)

    # Auto-calculate layout
    if layout is None:
        if n_images <= 3:
            layout = (1, n_images)
        elif n_images == 4:
            layout = (2, 2)
        elif n_images <= 6:
            layout = (2, 3)
        elif n_images <= 9:
            layout = (3, 3)
        else:
            nrows = int(np.ceil(np.sqrt(n_images)))
            ncols = int(np.ceil(n_images / nrows))
            layout = (nrows, ncols)

    nrows, ncols = layout

    # Check if layout is sufficient
    if nrows * ncols < n_images:
        raise ValueError(f"Layout {layout} is too small for {n_images} images")

    # Auto-generate labels (lowercase letters)
    if labels is None:
        labels = []
        for i in range(n_images):
            if i < 26:
                labels.append(chr(97 + i))  # a-z (lowercase)
            else:
                q, r = divmod(i, 26)
                labels.append(chr(97 + q - 1) + chr(97 + r))  # aa, ab, ...

    # Check label count
    if len(labels) < n_images:
        raise ValueError(f"Not enough labels ({len(labels)}) for {n_images} images")

    # Set matplotlib style
    setup_matplotlib_style()

    # Read all images
    images = [mpimg.imread(str(f)) for f in input_files]

    # Calculate image dimensions (assume all images have same size, use first image dimensions)
    img_height, img_width = images[0].shape[:2]

    # Calculate figure size (inches)
    original_dpi = 300
    subplot_width_inch = img_width / original_dpi
    subplot_height_inch = img_height / original_dpi

    # Reserve extra space for external labels
    label_height_inch = label_fontsize / 72.0 * 2.5  # 2.5 times font size as label area height

    # Calculate subplot spacing (inches)
    hspace_inch = spacing * subplot_width_inch
    vspace_inch = spacing * subplot_height_inch + label_height_inch * 0.3

    # Calculate total figure size
    fig_width = subplot_width_inch * ncols + hspace_inch * (ncols - 1)
    fig_height = (subplot_height_inch + label_height_inch) * nrows + vspace_inch * (nrows - 1)

    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Use GridSpec for precise layout control
    # Each row contains two grid rows: subfigure and label
    row_heights = []
    for _ in range(nrows):
        row_heights.append(subplot_height_inch)  # Subfigure height
        row_heights.append(label_height_inch)   # Label area height

    # Calculate hspace and wspace (relative ratios)
    avg_height = (subplot_height_inch + label_height_inch) / 2
    avg_width = subplot_width_inch

    gs = gridspec.GridSpec(
        nrows=nrows * 2,  # Each row split into 2 grids: subfigure + label
        ncols=ncols,
        figure=fig,
        height_ratios=row_heights,
        hspace=vspace_inch / avg_height,
        wspace=hspace_inch / avg_width
    )

    # Draw each subfigure and its label
    for idx, (img, label) in enumerate(zip(images, labels)):
        grid_row = (idx // ncols) * 2  # Each subfigure occupies two grid rows (image + label)
        grid_col = idx % ncols

        # Create subfigure (occupies grid_row)
        ax_img = fig.add_subplot(gs[grid_row, grid_col])
        ax_img.imshow(img)
        ax_img.axis('off')

        # Create label area (occupies grid_row+1)
        ax_label = fig.add_subplot(gs[grid_row + 1, grid_col])
        ax_label.axis('off')

        # Add text in label area (centered)
        label_text = f'({label})'

        bbox_props = dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            alpha=0.8,
            edgecolor='none'
        ) if label_bbox else None

        ax_label.text(
            0.5, 0.5, label_text,
            transform=ax_label.transAxes,
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            fontfamily='serif',  # Times New Roman
            color=label_color,
            ha='center', va='center',
            bbox=bbox_props
        )

    # Save figure (don't use tight_layout as we've precisely controlled the layout)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_file,
        format='png',
        dpi=dpi,
        bbox_inches=None,  # Don't crop, use precisely calculated size
        facecolor='white',
        edgecolor='none',
        pad_inches=0.05  # Small margin
    )
    plt.close(fig)

    print(f"[OK] Combined figure saved: {output_file}")
    print(f"     Layout: {nrows}x{ncols}, Labels: external bottom, Spacing: {spacing:.3f}")
    return output_file


def main():
    """Test function"""
    import tempfile

    # Create test images
    test_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {test_dir}")

    # Generate 3 test images
    for i, label in enumerate(['a', 'b', 'c']):
        fig, ax = plt.subplots(figsize=(4, 3))
        # Add content to image bottom to test label occlusion
        ax.text(0.5, 0.5, f'Test Image {label.upper()}',
               ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(0.5, 0.05, 'Bottom content (should not be occluded)',
               ha='center', va='bottom', fontsize=10, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        test_file = test_dir / f'test_{label}.png'
        fig.savefig(test_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Created: {test_file}")

    # Test combination (1x3 layout)
    print("\n=== Test 1: 1x3 layout ===")
    input_files = [test_dir / f'test_{label}.png' for label in ['a', 'b', 'c']]
    output_file = test_dir / 'combined_1x3.png'

    combine_figures(
        input_files=input_files,
        output_file=output_file,
        layout=(1, 3),
        labels=['a', 'b', 'c'],
        spacing=0.03
    )

    # Test combination (2x2 layout)
    print("\n=== Test 2: 2x2 layout ===")
    # Create 4th image
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, 'Test Image D',
           ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.05, 'Bottom content (should not be occluded)',
           ha='center', va='bottom', fontsize=10, color='red')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    test_file_d = test_dir / 'test_d.png'
    fig.savefig(test_file_d, dpi=300, bbox_inches='tight')
    plt.close(fig)

    input_files_2x2 = [test_dir / f'test_{label}.png' for label in ['a', 'b', 'c', 'd']]
    output_file_2x2 = test_dir / 'combined_2x2.png'

    combine_figures(
        input_files=input_files_2x2,
        output_file=output_file_2x2,
        layout=(2, 2),
        labels=['a', 'b', 'c', 'd'],
        spacing=0.04
    )

    print(f"\n[SUCCESS] Test completed successfully!")
    print(f"\nOutputs:")
    print(f"  1x3: {output_file}")
    print(f"  2x2: {output_file_2x2}")
    print(f"\nKey features:")
    print(f"  - Labels (a), (b), (c), (d) are placed OUTSIDE subfigures at the bottom")
    print(f"  - Adequate spacing between subfigures (no overlap)")
    print(f"  - Times New Roman font")
    print(f"  - Bottom content in original images is NOT occluded by labels")


if __name__ == '__main__':
    main()
