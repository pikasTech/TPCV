#!/usr/bin/env python3
"""
统一的图片拼接工具 V2 - 外部标签版本
==========================================

功能：
1. 将多个单独的图片拼接成一个复合图
2. 在每个子图的**外部底部**添加 (a), (b), (c) 等标注（不遮挡原图）
3. 子图之间有足够间隙，避免标签重叠
4. 不添加图题（论文的图题在图的下方）

使用方法：
---------
from combine_figures_v2 import combine_figures

# 方法1：指定图片列表和布局
combine_figures(
    input_files=['fig_A.png', 'fig_B.png', 'fig_C.png'],
    output_file='combined.png',
    layout=(1, 3),  # 1行3列
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
    """设置matplotlib样式（IEEE风格）"""
    # Apply scienceplots IEEE style
    plt.style.use(['science', 'ieee'])

    # 强制serif字体（IEEE要求，Times New Roman）
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
    将多个单独的图片拼接成一个复合图，标签放在子图外部底部

    Parameters
    ----------
    input_files : List[Union[str, Path]]
        输入图片文件列表
    output_file : Union[str, Path]
        输出文件路径
    layout : Optional[Tuple[int, int]], optional
        布局 (行数, 列数)，如果为None则自动计算，默认为None
    labels : Optional[List[str]], optional
        标注列表，如果为None则自动生成 ['a', 'b', 'c', ...]，默认为None
    label_position : str, optional
        标注位置（保留参数，当前仅支持bottom_center）
    label_fontsize : int, optional
        标注字体大小，默认为12
    label_fontweight : str, optional
        标注字体粗细，默认为'normal'
    label_color : str, optional
        标注颜色，默认为'black'
    label_bbox : bool, optional
        是否为标注添加白色背景框，默认为False（不添加，避免遮挡）
    spacing : float, optional
        子图之间的间距比例，默认为0.02（2%）
    dpi : int, optional
        输出DPI，默认为300

    Returns
    -------
    Path
        输出文件路径
    """
    # 转换为Path对象
    input_files = [Path(f) for f in input_files]
    output_file = Path(output_file)

    # 检查输入文件
    for f in input_files:
        if not f.exists():
            raise FileNotFoundError(f"Input file not found: {f}")

    n_images = len(input_files)

    # 自动计算布局
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

    # 检查布局是否足够
    if nrows * ncols < n_images:
        raise ValueError(f"Layout {layout} is too small for {n_images} images")

    # 自动生成标注（小写字母）
    if labels is None:
        labels = []
        for i in range(n_images):
            if i < 26:
                labels.append(chr(97 + i))  # a-z (小写)
            else:
                q, r = divmod(i, 26)
                labels.append(chr(97 + q - 1) + chr(97 + r))  # aa, ab, ...

    # 检查标注数量
    if len(labels) < n_images:
        raise ValueError(f"Not enough labels ({len(labels)}) for {n_images} images")

    # 设置matplotlib样式
    setup_matplotlib_style()

    # 读取所有图片
    images = [mpimg.imread(str(f)) for f in input_files]

    # 计算图片尺寸（假设所有图片尺寸相同，使用第一张图片的尺寸）
    img_height, img_width = images[0].shape[:2]

    # 计算figure尺寸（英寸）
    original_dpi = 300
    subplot_width_inch = img_width / original_dpi
    subplot_height_inch = img_height / original_dpi

    # 为外部标签预留额外空间
    label_height_inch = label_fontsize / 72.0 * 2.5  # 字体大小的2.5倍作为标签区域高度

    # 子图间隙计算（英寸）
    hspace_inch = spacing * subplot_width_inch
    vspace_inch = spacing * subplot_height_inch + label_height_inch * 0.3

    # 计算总figure尺寸
    fig_width = subplot_width_inch * ncols + hspace_inch * (ncols - 1)
    fig_height = (subplot_height_inch + label_height_inch) * nrows + vspace_inch * (nrows - 1)

    # 创建figure
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 使用GridSpec精确控制布局
    # 每行包含子图和标签两个grid行
    row_heights = []
    for _ in range(nrows):
        row_heights.append(subplot_height_inch)  # 子图高度
        row_heights.append(label_height_inch)   # 标签区域高度

    # 计算hspace和wspace（相对比例）
    avg_height = (subplot_height_inch + label_height_inch) / 2
    avg_width = subplot_width_inch

    gs = gridspec.GridSpec(
        nrows=nrows * 2,  # 每行拆分为2个grid：子图 + 标签
        ncols=ncols,
        figure=fig,
        height_ratios=row_heights,
        hspace=vspace_inch / avg_height,
        wspace=hspace_inch / avg_width
    )

    # 绘制每个子图及其标签
    for idx, (img, label) in enumerate(zip(images, labels)):
        grid_row = (idx // ncols) * 2  # 每个子图占据两行grid（图 + 标签）
        grid_col = idx % ncols

        # 创建子图（占据第grid_row行）
        ax_img = fig.add_subplot(gs[grid_row, grid_col])
        ax_img.imshow(img)
        ax_img.axis('off')

        # 创建标签区域（占据第grid_row+1行）
        ax_label = fig.add_subplot(gs[grid_row + 1, grid_col])
        ax_label.axis('off')

        # 在标签区域添加文本（中心位置）
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

    # 保存图片（不使用tight_layout，因为我们已经精确控制了布局）
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_file,
        format='png',
        dpi=dpi,
        bbox_inches=None,  # 不裁剪，使用精确计算的尺寸
        facecolor='white',
        edgecolor='none',
        pad_inches=0.05  # 小的边距
    )
    plt.close(fig)

    print(f"[OK] Combined figure saved: {output_file}")
    print(f"     Layout: {nrows}x{ncols}, Labels: external bottom, Spacing: {spacing:.3f}")
    return output_file


def main():
    """测试函数"""
    import tempfile

    # 创建测试图片
    test_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {test_dir}")

    # 生成3张测试图片
    for i, label in enumerate(['a', 'b', 'c']):
        fig, ax = plt.subplots(figsize=(4, 3))
        # 添加一些内容到图片底部，测试标签是否遮挡
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

    # 测试拼接（1x3布局）
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

    # 测试拼接（2x2布局）
    print("\n=== Test 2: 2x2 layout ===")
    # 创建第4张图片
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
