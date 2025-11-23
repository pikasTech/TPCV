#!/usr/bin/env python3
"""
三通道相关算法核心实现（纯ng/√Hz单位系统）

基于权威文献的标准三通道相关算法实现：
许卫卫, 袁松湧, 艾印双等. 2017. 多通道相关分析用于宽频带地震仪自噪声检测.
地球物理学报, 60(9):3466-3474, doi:10.6038/cjg20170916.

原始文献：
Sleeman R, van Wettum A, Trampert J. 2006. Three-channel correlation analysis:
A new technique to measure instrument noise of digitizers and seismic sensors.
Bulletin of the Seismological Society of America, 96(1): 258-271.

理论基础：
通过三通道相关分析分离相干信号和独立噪声，基于以下假设：
1. 输入地动信号与地震仪自噪声之间不相关
2. 各通道对应的自噪声之间不相关
3. 各通道接收相同的输入信号

单位系统：
- 时域信号：ng（等价于 m/s² 但用 ng 表示，1 ng = 9.80665e-9 m/s²）
- 功率谱密度（PSD）：ng^2/Hz
- 振幅谱密度（ASD）：ng/√Hz
"""

import numpy as np
from scipy import signal
import json
import warnings

class ThreeChannelCorrelation:
    """三通道相关算法核心类（纯ng单位系统）"""

    def __init__(self, config_file="experiment_config.json"):
        self.config = self.load_config(config_file)
        self.setup_parameters()

    def load_config(self, config_file):
        """加载配置文件"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def setup_parameters(self):
        """设置实验参数（纯ng单位系统）"""
        signal_params = self.config["signal_parameters"]
        self.fs = signal_params["sampling_rate_hz"]
        self.T = signal_params["signal_length_seconds"]
        self.f0 = signal_params["target_frequency_hz"]

        # 直接读取ng/√Hz参数（唯一支持的单位）
        self.signal_asd_ng = signal_params["signal_asd_ng_sqrthz"]  # ng/√Hz
        self.noise_asd_ng = signal_params["noise_asd_ng_sqrthz"]    # ng/√Hz

        self.N = int(self.T * self.fs)

        # Welch参数
        welch_params = self.config["welch_parameters"]
        self.nperseg = welch_params["nperseg"]
        self.overlap_ratio = welch_params["overlap_ratio"]
        self.noverlap = int(self.nperseg * self.overlap_ratio)
        self.window = welch_params["window"]

        # 通道噪声因子（支持非对称噪声）
        self.channel_noise_factors = signal_params.get("channel_noise_factors", [1.0, 1.0, 1.0])

        # 创建噪声等级对象（纯ng单位）
        self.noise_level = {
            "amplitude_ng_sqrthz": self.noise_asd_ng,
            "name": "paper_reference",
            "snr_db": 20 * np.log10(self.signal_asd_ng / self.noise_asd_ng)
        }

        # 设置随机种子
        computation_settings = self.config["computation_settings"]
        if computation_settings.get("random_seed") is not None:
            np.random.seed(computation_settings["random_seed"])

    def generate_three_channel_signals(self):
        """
        生成三通道信号（纯ng单位系统）

        信号模型：x_i(t) = s(t) + n_i(t)

        单位系统：
        - 时域信号：ng（1 ng = 9.80665e-9 m/s²）
        - PSD：ng^2/Hz（Welch方法输出）
        - ASD：ng/√Hz（配置参数和最终结果）

        方法：
        1. 生成标准差=1ng的测试噪声
        2. 用Welch计算PSD，提取目标频率的ASD
        3. 反推所需时域标准差以达到目标ASD
        4. 生成相干信号和独立噪声
        """
        # 【实验验证】生成标准差=1ng的测试噪声
        test_noise_ng = np.random.normal(0, 1.0, self.N)  # 单位：ng

        # 用Welch计算PSD
        f_test, psd_test_ng2_hz = signal.welch(
            test_noise_ng, self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling='density'
        )  # 输出单位：ng^2/Hz

        # 提取目标频率的ASD
        test_freq_idx = np.argmin(np.abs(f_test - self.f0))
        measured_asd_ng = np.sqrt(psd_test_ng2_hz[test_freq_idx])  # ng/√Hz

        # 反推时域标准差
        if measured_asd_ng > 0:
            # 目标：self.signal_asd_ng, self.noise_asd_ng (ng/√Hz)
            # 测得：measured_asd_ng (ng/√Hz，来自σ=1ng的噪声)
            # 需要：σ_required = target_asd_ng / measured_asd_ng
            sigma_signal_ng = self.signal_asd_ng / measured_asd_ng  # ng
            sigma_noise_ng = self.noise_asd_ng / measured_asd_ng    # ng
        else:
            raise ValueError("实验验证失败：测得ASD为零")

        print(f"[ng单位系统] 目标信号ASD: {self.signal_asd_ng:.2e} ng/√Hz")
        print(f"[ng单位系统] 时域信号标准差: {sigma_signal_ng:.2e} ng")
        print(f"[ng单位系统] 目标噪声ASD: {self.noise_asd_ng:.2e} ng/√Hz")
        print(f"[ng单位系统] 时域噪声标准差: {sigma_noise_ng:.2e} ng")
        print(f"[实验验证] 标准差=1ng白噪声 → ASD={measured_asd_ng:.2e} ng/√Hz")

        # 生成相干信号（ng单位白噪声）
        coherent_signal_ng = np.random.normal(0, sigma_signal_ng, self.N)

        # 生成独立噪声（ng单位，支持非对称噪声因子）
        noise1_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[0], self.N)
        noise2_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[1], self.N)
        noise3_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[2], self.N)

        # 合成三通道信号
        channel1_ng = coherent_signal_ng + noise1_ng
        channel2_ng = coherent_signal_ng + noise2_ng
        channel3_ng = coherent_signal_ng + noise3_ng

        # 验证生成信号的统计特性
        actual_noise_std = np.std(noise1_ng)
        print(f"[验证] 通道1噪声标准差: 期望={sigma_noise_ng * self.channel_noise_factors[0]:.2e}, 实际={actual_noise_std:.2e}")

        return channel1_ng, channel2_ng, channel3_ng

    def generate_noise_only_signals(self):
        """
        生成纯噪声信号（用于分离测试）

        单位：ng
        """
        # 基于实验验证的方法生成噪声
        test_noise_ng = np.random.normal(0, 1.0, self.N)
        f_test, psd_test_ng2_hz = signal.welch(
            test_noise_ng, self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling='density'
        )

        test_freq_idx = np.argmin(np.abs(f_test - self.f0))
        measured_asd_ng = np.sqrt(psd_test_ng2_hz[test_freq_idx])

        if measured_asd_ng > 0:
            sigma_noise_ng = self.noise_asd_ng / measured_asd_ng
        else:
            raise ValueError("实验验证失败：测得ASD为零")

        # 生成三通道纯噪声（支持非对称噪声因子）
        noise1_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[0], self.N)
        noise2_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[1], self.N)
        noise3_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[2], self.N)

        return noise1_ng, noise2_ng, noise3_ng

    def apply_synchronization_delay(self, signals, delays):
        """
        应用同步延迟到多通道信号

        Parameters:
        - signals: 三通道信号元组 (channel1, channel2, channel3)，单位：ng
        - delays: 各通道的延迟样本数列表 [delay1, delay2, delay3]

        Returns:
        - delayed_signals: 应用延迟后的信号元组，单位：ng

        注意：使用np.roll实现循环移位，正值表示向右移（延迟），负值表示向左移（提前）
        """
        delayed_signals = []
        for i, (sig, delay) in enumerate(zip(signals, delays)):
            if delay != 0:
                delayed_signal = np.roll(sig, delay)
                delayed_signals.append(delayed_signal)
            else:
                delayed_signals.append(sig.copy())

        return tuple(delayed_signals)

    def generate_signal_only(self):
        """
        生成纯信号（用于分离测试）

        单位：ng（相干白噪声）
        """
        # 基于实验验证的方法生成信号
        test_noise_ng = np.random.normal(0, 1.0, self.N)
        f_test, psd_test_ng2_hz = signal.welch(
            test_noise_ng, self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling='density'
        )

        test_freq_idx = np.argmin(np.abs(f_test - self.f0))
        measured_asd_ng = np.sqrt(psd_test_ng2_hz[test_freq_idx])

        if measured_asd_ng > 0:
            sigma_signal_ng = self.signal_asd_ng / measured_asd_ng
        else:
            raise ValueError("实验验证失败：测得ASD为零")

        # 生成相干白噪声信号
        signal_component_ng = np.random.normal(0, sigma_signal_ng, self.N)

        print(f"[信号生成] 目标ASD: {self.signal_asd_ng:.2e} ng/√Hz")
        print(f"[信号生成] 时域标准差: {sigma_signal_ng:.2e} ng")

        return signal_component_ng

    def apply_permutation(self, ch1, ch2, ch3, permutation):
        """
        根据排列重新安排通道

        Parameters:
        - ch1, ch2, ch3: 三通道信号，单位：ng
        - permutation: 如"213"表示 [ch2, ch1, ch3]

        Returns:
        - 重新排列的通道列表，单位：ng
        """
        channels = [ch1, ch2, ch3]
        perm_indices = [int(permutation[i]) - 1 for i in range(3)]
        return [channels[idx] for idx in perm_indices]

    def compute_psd(self, x_ng):
        """
        计算功率谱密度

        Parameters:
        - x_ng: 时域信号，单位：ng

        Returns:
        - f: 频率数组，单位：Hz
        - psd_ng2_hz: 功率谱密度，单位：ng^2/Hz
        """
        f, psd_ng2_hz = signal.welch(
            x_ng, self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling='density'
        )
        return f, psd_ng2_hz

    def compute_cross_psd(self, x_ng, y_ng):
        """
        计算交叉功率谱密度

        Parameters:
        - x_ng, y_ng: 时域信号，单位：ng

        Returns:
        - f: 频率数组，单位：Hz
        - csd_ng2_hz: 交叉功率谱密度，单位：ng^2/Hz（复数）
        """
        f, csd_ng2_hz = signal.csd(
            x_ng, y_ng, self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling='density'
        )
        return f, csd_ng2_hz

    def three_channel_correlation(self, signal1_ng, signal2_ng, signal3_ng):
        """
        标准三通道相关算法 - 符合权威文献

        基于许卫卫等(2017)论文公式(5)：
        N_ii = P_ii - (P_ji * P_ik) / P_jk

        其中：
        - N_ii: 第i通道自噪声功率谱密度，单位：ng^2/Hz
        - P_ii: 第i通道自功率谱密度，单位：ng^2/Hz
        - P_ji, P_ik, P_jk: 互功率谱密度，单位：ng^2/Hz（复数）
        - i ≠ j ≠ k ∈ {1,2,3}

        算法原理：
        通过三通道相关分析分离相干信号和独立噪声。scipy.signal.csd
        已经正确处理了互功率谱的复数共轭运算，因此直接使用即可。

        参考文献：
        许卫卫等, 2017, 地球物理学报, 60(9):3466-3474, 公式(5)
        Sleeman R, et al. 2006, BSSA, 96(1): 258-271

        Parameters:
        - signal1_ng, signal2_ng, signal3_ng: 三通道时域信号，单位：ng

        Returns:
        - f: 频率数组，单位：Hz
        - N_11_ng2_hz: 通道1自噪声PSD，单位：ng^2/Hz
        """
        # 计算PSD和交叉PSD（单位：ng^2/Hz）
        f, P_11 = self.compute_psd(signal1_ng)
        _, P_12 = self.compute_cross_psd(signal1_ng, signal2_ng)  # P_ji = P_12
        _, P_13 = self.compute_cross_psd(signal1_ng, signal3_ng)  # P_ik = P_13
        _, P_23 = self.compute_cross_psd(signal2_ng, signal3_ng)  # P_jk = P_23

        # 应用论文公式(5): N_ii = P_ii - (P_ji * P_ik) / P_jk
        # 这里分析第一个通道 (i=1, j=2, k=3)
        N_11 = P_11 - (P_12 * P_13) / P_23

        # 【调试】在目标频率处检查计算过程
        freq_idx = self.find_frequency_index(f)
        if freq_idx < len(f):
            print(f"[三通道算法调试] {self.f0}Hz处的谱密度:")
            print(f"  P_11 (通道1自谱): {P_11[freq_idx]:.2e} ng^2/Hz")
            print(f"  P_12 (1-2交叉谱): {P_12[freq_idx]:.2e} ng^2/Hz")
            print(f"  P_13 (1-3交叉谱): {P_13[freq_idx]:.2e} ng^2/Hz")
            print(f"  P_23 (2-3交叉谱): {P_23[freq_idx]:.2e} ng^2/Hz")
            print(f"  交叉项 (P_12*P_13)/P_23: {(P_12[freq_idx] * P_13[freq_idx]) / P_23[freq_idx]:.2e} ng^2/Hz")
            print(f"  相关后 N_11: {N_11[freq_idx]:.2e} ng^2/Hz")
            print(f"  理论噪声PSD: {self.noise_asd_ng**2:.2e} ng^2/Hz")

        # 取实部并确保为正值（物理上噪声功率谱密度必须为正）
        N_11_ng2_hz = np.real(N_11)

        # 对于可能出现的负值进行处理（表明算法假设不满足）
        if np.any(N_11_ng2_hz < 0):
            negative_ratio = np.sum(N_11_ng2_hz < 0) / len(N_11_ng2_hz) * 100
            warnings.warn(f"检测到{negative_ratio:.1f}%的负值，可能表明信号模型假设不满足")

        return f, N_11_ng2_hz

    def find_frequency_index(self, frequencies):
        """找到目标频率的索引"""
        return np.argmin(np.abs(frequencies - self.f0))

    def extract_noise_amplitude_at_frequency(self, frequencies, noise_psd_ng2_hz):
        """
        提取目标频率处的噪声幅度

        Parameters:
        - frequencies: 频率数组，单位：Hz
        - noise_psd_ng2_hz: 噪声PSD，单位：ng^2/Hz

        Returns:
        - amplitude_ng_sqrthz: 噪声ASD，单位：ng/√Hz
        """
        freq_idx = self.find_frequency_index(frequencies)
        amplitude_ng_sqrthz = np.sqrt(np.abs(noise_psd_ng2_hz[freq_idx]))
        return amplitude_ng_sqrthz

    def get_noise_asd(self, noise_psd_ng2_hz):
        """
        从PSD计算ASD

        Parameters:
        - noise_psd_ng2_hz: 噪声PSD数组，单位：ng^2/Hz

        Returns:
        - noise_asd_ng_sqrthz: 噪声ASD数组，单位：ng/√Hz
        """
        return np.sqrt(np.abs(noise_psd_ng2_hz))

    def get_output_unit(self):
        """
        获取输出单位

        Returns:
        - 固定返回 "ng/sqrt(Hz)"
        """
        return "ng/sqrt(Hz)"
