#!/usr/bin/env python3
"""
E01 Algorithm Verification - Specialized Visualizer V2
======================================================
重构版本：生成单独的子图，然后用 combine_figures.py 拼接

主要改进：
1. 每个子图作为单独的图片生成（无子图，无标题）
2. 使用 combine_figures.py 统一拼接
3. 标注 (A), (B), (C) 统一在底部居中
4. 图题由LaTeX添加，不在图片中
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
    """V2版本：生成单图 + 拼接"""

    def __init__(self, exam_name="E01-algorithm_verification", timestamp=None, base_dir=None):
        """初始化可视化器"""
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

        # 创建单图子目录
        self.individual_dir = self.figures_dir / "individual"
        self.individual_dir.mkdir(parents=True, exist_ok=True)

        # Load results and config
        self.load_data()
        self.setup_matplotlib_style()

    def load_data(self):
        """加载实验结果和配置"""
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

        # 强制serif字体（IEEE要求，必须在style.use之后设置以覆盖默认）
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

        # 中文字体配置（覆盖IEEE默认设置）
        import matplotlib.font_manager as fm
        if hasattr(fm, '_fmcache'):
            fm._fmcache.clear()

        # 设置中文字体（SimHei黑体）用于支持中文显示
        matplotlib.rcParams['font.sans-serif'] = [
            'SimHei',           # 优先使用黑体（中文）
            'DejaVu Sans',
            'Arial',
            'Helvetica'
        ]
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # IEEE风格的字体大小配置
        matplotlib.rcParams['font.size'] = 8
        matplotlib.rcParams['axes.labelsize'] = 9
        matplotlib.rcParams['axes.titlesize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['legend.fontsize'] = 7
        matplotlib.rcParams['figure.titlesize'] = 11

        # 其他IEEE风格配置
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.rcParams['savefig.dpi'] = 300

        # LaTeX渲染配置（使用matplotlib内置LaTeX，不依赖系统TeX）
        matplotlib.rcParams['text.usetex'] = False  # 使用matplotlib内置LaTeX
        matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern字体
        matplotlib.rcParams['mathtext.default'] = 'regular'

    def regenerate_signals(self, duration_seconds=0.1):
        """重新生成信号，同时返回相干噪声和自噪声分量"""
        import sys
        code_dir = Path(__file__).parent
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))

        from core_algorithm import ThreeChannelCorrelation

        config_file = str(self.exam_dir / "config.json")
        core = ThreeChannelCorrelation(config_file)

        # 获取配置参数
        fs = core.fs
        n_samples = int(duration_seconds * fs)
        
        # 生成相干信号和自噪声
        # 使用与 core_algorithm 相同的方法
        test_noise = np.random.normal(0, 1.0, n_samples)
        
        # 调整 Welch 参数以适应小样本
        nperseg = min(core.nperseg, n_samples)
        noverlap = int(nperseg * 0.5)  # 使用 50% 重叠
        
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
        
        # 生成相干信号（所有通道共享）
        coherent_signal = np.random.normal(0, sigma_signal, n_samples)
        
        # 生成各通道的自噪声
        noise1 = np.random.normal(0, sigma_noise * core.channel_noise_factors[0], n_samples)
        noise2 = np.random.normal(0, sigma_noise * core.channel_noise_factors[1], n_samples)
        noise3 = np.random.normal(0, sigma_noise * core.channel_noise_factors[2], n_samples)
        
        # 合成总信号
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
        生成单个面板（无标题，无标注）

        Parameters
        ----------
        data_dict : dict
            包含绘图数据和配置
        output_file : Path
            输出文件路径
        figsize : tuple
            图片尺寸
        xlabel, ylabel : str
            坐标轴标签
        """
        with plt.style.context(['science', 'ieee']):
            fig, ax = plt.subplots(figsize=figsize)

            # 根据数据类型绘图
            plot_type = data_dict.get('type', 'line')

            if plot_type == 'line':
                ax.plot(
                    data_dict['x'],
                    data_dict['y'],
                    color=data_dict.get('color', '#1f77b4'),
                    linewidth=data_dict.get('linewidth', 1.2),
                    alpha=data_dict.get('alpha', 0.7)
                )
                # 添加RMS标注（如果有）
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

                # 添加colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(data_dict.get('cbar_label', ''))

                # 添加数值标注
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

                # 添加水平线（如果有）
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

            # 设置坐标轴标签
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)

            # 添加网格
            if data_dict.get('grid', True):
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存
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
        生成F00的3个单独面板（时域波形）
        每个子图显示三条线：相干噪声、自噪声、总噪声
        只显示前0.1秒

        Returns
        -------
        list
            3个面板文件路径列表
        """
        print("\n" + "="*60)
        print("Generating F00: Individual Time Domain Panels")
        print("="*60)

        # 重新生成信号（包含分量），使用0.1秒
        signals = self.regenerate_signals(duration_seconds=0.1)
        time = signals['time']
        fs = signals['fs']
        
        # 只显示前0.1秒
        display_duration = 0.1
        display_samples = int(display_duration * fs)
        
        # 定义3个通道的配置
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

        # 先计算所有通道的y轴范围，确保一致
        all_y_values = []
        for ch_config in channels:
            all_y_values.extend(ch_config['coherent'])
            all_y_values.extend(ch_config['self_noise'])
            all_y_values.extend(ch_config['total'])
        y_min, y_max = np.min(all_y_values), np.max(all_y_values)
        y_margin = (y_max - y_min) * 0.05

        # 应用 IEEE 风格（完全模仿 E02）
        with plt.style.context(['science', 'ieee']):
            # 生成每个独立的子图
            for i, ch_config in enumerate(channels):
                output_file = self.individual_dir / ch_config['filename']
                
                # 创建图形
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # 绘制三条波形
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
                
                # 设置统一的y轴范围
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                
                # 设置坐标轴标签
                ax.set_xlabel(r'Time (s)')
                ax.set_ylabel(r'Amplitude (ng)')
                
                # 添加图例
                ax.legend(loc='best', framealpha=0.8, edgecolor='gray', fontsize=8)
                
                # 添加网格
                ax.grid(True, alpha=0.3)
                
                # 保存
                plt.tight_layout()
                fig.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"[OK] Saved panel: {output_file.name}")
                panel_files.append(output_file)

        return panel_files

    def generate_f00_combined(self):
        """
        拼接F00的3个面板成一个复合图

        Returns
        -------
        Path
            复合图文件路径
        """
        print("\n" + "="*60)
        print("Combining F00 panels...")
        print("="*60)

        # 生成单独的面板
        panel_files = self.generate_f00_individual_panels()

        # 拼接（横向排列）
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
        生成F01的4个单独面板（算法验证）

        Returns
        -------
        list
            4个面板文件路径列表
        """
        print("\n" + "="*60)
        print("Generating F01: Individual Algorithm Validation Panels")
        print("="*60)

        # 提取结果数据
        results_data = self.results['results']
        permutations = ['123', '132', '213', '231', '312', '321']
        physical_channels = ['physical_channel_1', 'physical_channel_2', 'physical_channel_3']

        # 提取自噪声值
        noise_values = {perm: [] for perm in permutations}
        for perm in permutations:
            for ch in physical_channels:
                if perm in results_data[ch]:
                    noise_values[perm].append(results_data[ch][perm])

        # 计算统计量
        perm_means = {perm: np.mean(vals) for perm, vals in noise_values.items()}
        perm_stds = {perm: np.std(vals) for perm, vals in noise_values.items()}
        perm_cvs = {perm: (perm_stds[perm] / perm_means[perm] * 100) if perm_means[perm] > 0 else 0
                   for perm in permutations}

        # 计算相对误差
        all_values = [val for vals in noise_values.values() for val in vals]
        global_mean = np.mean(all_values)
        relative_errors = {perm: (perm_means[perm] - global_mean) / global_mean * 100
                          for perm in permutations}

        # 创建热图数据
        heatmap_data = np.zeros((3, 6))
        for i, ch in enumerate(physical_channels):
            for j, perm in enumerate(permutations):
                if perm in results_data[ch]:
                    heatmap_data[i, j] = results_data[ch][perm]

        # 面板A：排列敏感性
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

        # 面板B：热图
        panel_b_dict = {
            'type': 'heatmap',
            'data': heatmap_data,
            'cmap': 'YlOrRd',
            'xticks': permutations,
            'yticks': ['Ch1', 'Ch2', 'Ch3'],
            'cbar_label': r'Self-Noise (ng/$\sqrt{\mathrm{Hz}}$)',
            'annotate': True
        }

        # 面板C：CV分布
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

        # 面板D：相对误差
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

        # 生成4个面板
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
        拼接F01的4个面板成一个2x2复合图

        Returns
        -------
        Path
            复合图文件路径
        """
        print("\n" + "="*60)
        print("Combining F01 panels...")
        print("="*60)

        # 生成单独的面板
        panel_files = self.generate_f01_individual_panels()

        # 拼接
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
        """生成所有图表（单图 + 复合图）"""
        print("\n" + "="*60)
        print(f"E01 Specialized Visualization V2")
        print(f"Experiment: {self.exam_name}")
        print(f"Timestamp: {self.timestamp}")
        print("="*60)

        # 确保目录存在
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.individual_dir.mkdir(parents=True, exist_ok=True)

        # 生成复合图（内部会先生成单图）
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
    """主函数"""
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
