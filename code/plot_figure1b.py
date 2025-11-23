#!/usr/bin/env python3
"""
绘制三通道频域处理过程示意图 (Figure 1b)

用途：展示从原始ASD到提取自噪声的完整频谱变换过程
对应文档：doc/review/20251110晚/R12_investigation.md
对应LaTeX图：E01_figure1b_frequency_processing.png

绘图内容：
- 三个通道的原始ASD (√P_11, √P_22, √P_33)
- 通过两条路径提取的通道1自噪声 (N_1^(1), N_1^(2))
- 理论自噪声水平（黑色虚线）
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

# 添加code目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from core_algorithm import ThreeChannelCorrelation

# 注意：不在此处设置字体，避免覆盖scienceplots的IEEE风格
# IEEE风格要求使用serif字体（Times New Roman），将在main()和plot函数中设置


def load_baseline_config():
    """加载baseline实验配置"""
    config_path = Path(__file__).parent.parent / "exams" / "baseline" / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_signals_and_compute_spectra(config):
    """
    生成信号并计算频谱

    Returns:
        dict: 包含频率数组和各种ASD的字典
    """
    # 创建算法实例
    algo = ThreeChannelCorrelation(config)

    # 生成三通道信号
    ch1, ch2, ch3 = algo.generate_three_channel_signals()

    # 计算三个通道的原始PSD，然后开平方为ASD
    f, P_11 = algo.compute_psd(ch1)
    _, P_22 = algo.compute_psd(ch2)
    _, P_33 = algo.compute_psd(ch3)

    # 计算交叉功率谱密度 (用于两路径自噪声提取)
    _, P_12 = algo.compute_cross_psd(ch1, ch2)
    _, P_13 = algo.compute_cross_psd(ch1, ch3)
    _, P_23 = algo.compute_cross_psd(ch2, ch3)
    _, P_32 = algo.compute_cross_psd(ch3, ch2)  # P_32 = conj(P_23)

    # 计算两条路径的自噪声
    # 路径1: N_1^(1) = P_11 - (P_12 × P_13) / P_23
    N_1_path1 = P_11 - (P_12 * P_13) / P_23

    # 路径2: N_1^(2) = P_11 - (P_13 × P_12) / P_32
    N_1_path2 = P_11 - (P_13 * P_12) / P_32

    # 取实部和绝对值（物理上自噪声PSD必须为正实数）
    N_1_path1 = np.real(N_1_path1)
    N_1_path2 = np.real(N_1_path2)

    # 理论自噪声水平（从配置读取）
    noise_asd_theory = config["signal_parameters"]["noise_asd_ng_sqrthz"]

    # 将PSD开平方为ASD (ng/√Hz)
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
    保存绘图数据和配置到JSON文件

    Args:
        spectra: 包含所有频谱数据的字典
        config: 实验配置字典
        output_path: JSON文件输出路径
    """
    from datetime import datetime

    # 准备JSON数据结构
    json_data = {
        # 元数据
        "metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "script_version": "1.0.0",
            "script_name": "plot_figure1b.py",
            "description": "Three-channel frequency domain processing data",
            "figure_name": "E01_figure1b_frequency_processing",
            "data_source": "Synthetic three-channel signals with baseline configuration"
        },

        # 配置参数（实验可复现性的关键）
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

        # 绘图数据（完整）
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

        # 图表设置
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

        # 统计信息
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

    # 保存JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] JSON data saved to: {output_path}")


def plot_frequency_processing(spectra, output_path):
    """
    绘制频域处理过程图

    Args:
        spectra: 包含所有频谱数据的字典
        output_path: 输出图片路径
    """
    # 应用IEEE风格（使用context manager保持一致性）
    with plt.style.context(['science', 'ieee']):
        # 强制使用serif字体（IEEE标准要求）
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
        
        # IEEE标准字号设置（全局一致性）
        plt.rcParams.update({
            'font.size': 8,           # 基础字号
            'axes.titlesize': 9,      # 子图标题
            'axes.labelsize': 8,      # 坐标轴标签
            'xtick.labelsize': 7,     # X轴刻度标签
            'ytick.labelsize': 7,     # Y轴刻度标签
            'legend.fontsize': 7,     # 图例字号
            'figure.titlesize': 10    # 主标题
        })

        # 创建图形（IEEE双栏宽度：7英寸，高度适配黄金比例）
        fig, ax = plt.subplots(figsize=(7, 4.2))

        f = spectra['frequencies']

        # IEEE线条粗细标准
        line_width_main = 1.2      # 主曲线线宽（增粗：1.0→1.2）
        line_width_ref = 0.8       # 参考线线宽
        marker_size = 3.5          # 标记大小
        marker_edge_width = 0.7    # 标记边缘宽度

        # 绘制三个通道的原始ASD（使用IEEE推荐配色和线型）
        ax.loglog(f, spectra['P_11'], '-', color='#1f77b4', linewidth=line_width_main,
                  label=r'$\sqrt{P_{11}}$ (Ch1)', zorder=3)      # IEEE蓝色实线
        ax.loglog(f, spectra['P_22'], '--', color='#ff7f0e', linewidth=line_width_main,
                  label=r'$\sqrt{P_{22}}$ (Ch2)', zorder=3)      # IEEE橙色虚线
        ax.loglog(f, spectra['P_33'], '-.', color='#2ca02c', linewidth=line_width_main,
                  label=r'$\sqrt{P_{33}}$ (Ch3)', zorder=3)      # IEEE绿色点划线

        # 标记点间隔（对数空间均匀分布，避免低频密集、高频稀疏）
        # 在对数空间选择约30个点
        valid_f_mask = f > 0  # 跳过零频
        valid_indices = np.where(valid_f_mask)[0]
        if len(valid_indices) > 30:
            # 在对数空间均匀选择30个索引
            log_indices = np.logspace(np.log10(valid_indices[0]), np.log10(valid_indices[-1]), 30)
            marker_indices = np.unique(log_indices.astype(int))
            # 确保所有索引在有效范围内
            marker_indices = marker_indices[marker_indices < len(f)]
        else:
            marker_indices = valid_indices

        # 绘制两条路径提取的自噪声（空心标记，IEEE标准配色）
        ax.loglog(f[marker_indices], spectra['N_1_path1'][marker_indices],
                  'o', markersize=marker_size, markerfacecolor='none',
                  markeredgecolor='#d62728', markeredgewidth=marker_edge_width,
                  label=r'$\sqrt{N_1^{(1)}}$ (Path 1)', zorder=4)  # IEEE红色圆圈
        ax.loglog(f[marker_indices], spectra['N_1_path2'][marker_indices],
                  's', markersize=marker_size, markerfacecolor='none',
                  markeredgecolor='#9467bd', markeredgewidth=marker_edge_width,
                  label=r'$\sqrt{N_1^{(2)}}$ (Path 2)', zorder=4)  # IEEE紫色方块

        # 绘制理论值（黑色虚线，简化标签）
        ax.axhline(y=spectra['theory'], color='k', linestyle='--',
                   linewidth=line_width_ref, alpha=0.7,
                   label=f"Theory ({spectra['theory']:.2e} ng/$\\sqrt{{Hz}}$)",
                   zorder=2)

        # 设置坐标轴标签（IEEE标准格式）
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'Amplitude Spectral Density (ng/$\sqrt{\mathrm{Hz}}$)')
        # ax.set_title('Three-Channel Frequency Domain Processing', pad=10)  # 已注释：去掉图标题

        # IEEE标准网格线（细线，低透明度）
        ax.grid(True, which='major', alpha=0.3, linewidth=0.5, linestyle='-', zorder=1)
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.3, linestyle=':', zorder=1)

        # 图例优化（IEEE标准：2列布局，位置避免遮挡数据）
        # 使用ncol=2减少垂直空间占用，放置在右下角避免遮挡10-100Hz数据
        ax.legend(loc='lower right', ncol=2, frameon=True, framealpha=0.95,
                  edgecolor='gray', fancybox=False, shadow=False,
                  columnspacing=1.0, handlelength=2.0)

        # 设置坐标轴范围
        ax.set_xlim([1, 1000])  # 显示1Hz到1000Hz范围

        # 优化布局（IEEE标准间距）
        plt.tight_layout(pad=0.3)

        # 保存图片（IEEE标准：600 DPI用于出版）
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
        print(f"[OK] Frequency processing figure saved to: {output_path}")

        # 关闭图形
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
    """主函数"""
    args = parse_args()
    # 应用IEEE风格（scienceplots库）
    plt.style.use(['science', 'ieee'])

    # 强制serif字体（IEEE标准要求，必须在style.use之后设置以覆盖默认）
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    plt.rcParams.update({
        'font.size': 8,           # 基础字号
        'axes.titlesize': 9,      # 子图标题
        'axes.labelsize': 8,      # 坐标轴标签
        'xtick.labelsize': 7,     # X轴刻度标签
        'ytick.labelsize': 7,     # Y轴刻度标签
        'legend.fontsize': 7      # 图例字号
    })

    print("=" * 60)
    print("三通道频域处理过程示意图生成器 (Figure 1b)")
    print("=" * 60)
    print("[Style] 使用 scienceplots IEEE 风格")

    # 1. 加载配置
    print("\n[1/5] 加载baseline实验配置...")
    config_path = Path(__file__).parent.parent / "exams" / "baseline" / "config.json"

    # 2. 创建临时算法实例用于信号生成
    print("[2/5] 生成三通道信号并计算频谱...")

    # 创建ThreeChannelCorrelation实例需要配置文件路径
    # 因此我们创建一个临时配置文件或直接使用现有的
    algo = ThreeChannelCorrelation(str(config_path))

    # 生成三通道信号
    ch1, ch2, ch3 = algo.generate_three_channel_signals()
    print(f"  - 采样率: {algo.fs} Hz")
    print(f"  - 信号长度: {algo.T} 秒")
    print(f"  - 目标频率: {algo.f0} Hz")

    # 计算所有需要的频谱
    print("[3/5] 计算PSD和CPSD...")
    f, P_11 = algo.compute_psd(ch1)
    _, P_22 = algo.compute_psd(ch2)
    _, P_33 = algo.compute_psd(ch3)

    _, P_12 = algo.compute_cross_psd(ch1, ch2)
    _, P_13 = algo.compute_cross_psd(ch1, ch3)
    _, P_23 = algo.compute_cross_psd(ch2, ch3)
    _, P_32 = algo.compute_cross_psd(ch3, ch2)

    # 计算两条路径的自噪声
    N_1_path1 = np.real(P_11 - (P_12 * P_13) / P_23)
    N_1_path2 = np.real(P_11 - (P_13 * P_12) / P_32)

    # 理论自噪声
    noise_asd_theory = algo.noise_asd_ng

    # 组装数据
    spectra = {
        'frequencies': f,
        'P_11': np.sqrt(P_11),  # ASD: ng/√Hz
        'P_22': np.sqrt(P_22),  # ASD: ng/√Hz
        'P_33': np.sqrt(P_33),  # ASD: ng/√Hz
        'N_1_path1': np.sqrt(np.abs(N_1_path1)),  # ASD: ng/√Hz
        'N_1_path2': np.sqrt(np.abs(N_1_path2)),  # ASD: ng/√Hz
        'theory': noise_asd_theory  # ASD: ng/√Hz
    }

    # 4. 绘制并保存图片和JSON数据
    print("[4/5] 绘制频域处理图...")
    default_output_dir = Path(__file__).parent.parent.parent / "manuscript" / "latex"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.output_basename or "E01_figure1b_frequency_processing"
    output_path_png = output_dir / f"{base_name}.png"
    output_path_json = output_dir / f"{base_name}.json"

    plot_frequency_processing(spectra, output_path_png)

    # 5. 保存JSON数据
    print("[5/5] 保存JSON数据...")
    # 加载配置以传递给JSON保存函数
    config = json.loads(config_path.read_text(encoding='utf-8'))
    save_json_data(spectra, config, output_path_json)

    # 输出统计信息
    print("\n" + "=" * 60)
    print("统计信息:")
    print("=" * 60)
    print(f"频率范围: {f[1]:.4f} - {f[-1]:.2f} Hz")
    print(f"频率点数: {len(f)}")

    # 计算10Hz处的值
    freq_idx = np.argmin(np.abs(f - algo.f0))
    print(f"\nValues at {f[freq_idx]:.2f} Hz:")
    print(f"  P_11: {P_11[freq_idx]:.4e} ng/√Hz")
    print(f"  P_22: {P_22[freq_idx]:.4e} ng/√Hz")
    print(f"  P_33: {P_33[freq_idx]:.4e} ng/√Hz")
    print(f"  N_1^(1): {N_1_path1[freq_idx]:.4e} ng/√Hz")
    print(f"  N_1^(2): {N_1_path2[freq_idx]:.4e} ng/√Hz")
    print(f"  Theory: {noise_asd_theory:.4e} ng/√Hz")

    # 计算两路径的相对偏差
    rel_diff = np.abs(N_1_path1 - N_1_path2) / (N_1_path1 + N_1_path2 + 1e-20) * 2
    print(f"\nTwo-path average relative deviation: {np.mean(rel_diff):.4f}")
    print(f"Two-path maximum relative deviation: {np.max(rel_diff):.4f}")

    print("\n" + "=" * 60)
    print("[OK] Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

