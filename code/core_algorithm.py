#!/usr/bin/env python3
"""
Three-Channel Correlation Algorithm Core Implementation (Pure ng/√Hz Unit System)

Standard three-channel correlation algorithm implementation based on authoritative literature:
Xu W, Yuan S, Ai Y, et al. 2017. Multi-channel correlation analysis for self-noise
detection of broadband seismometers. Chinese Journal of Geophysics, 60(9):3466-3474,
doi:10.6038/cjg20170916.

Original reference:
Sleeman R, van Wettum A, Trampert J. 2006. Three-channel correlation analysis:
A new technique to measure instrument noise of digitizers and seismic sensors.
Bulletin of the Seismological Society of America, 96(1): 258-271.

Theoretical foundation:
Separates coherent signal and independent noise through three-channel correlation analysis,
based on the following assumptions:
1. Input ground motion signal is uncorrelated with seismometer self-noise
2. Self-noise of each channel is mutually uncorrelated
3. All channels receive identical input signal

Unit system:
- Time-domain signal: ng (equivalent to m/s² but expressed in ng, 1 ng = 9.80665e-9 m/s²)
- Power Spectral Density (PSD): ng^2/Hz
- Amplitude Spectral Density (ASD): ng/√Hz
"""

import numpy as np
from scipy import signal
import json
import warnings

class ThreeChannelCorrelation:
    """Three-channel correlation algorithm core class (pure ng unit system)"""

    def __init__(self, config_file="experiment_config.json"):
        self.config = self.load_config(config_file)
        self.setup_parameters()

    def load_config(self, config_file):
        """Load configuration file"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def setup_parameters(self):
        """Setup experimental parameters (pure ng unit system)"""
        signal_params = self.config["signal_parameters"]
        self.fs = signal_params["sampling_rate_hz"]
        self.T = signal_params["signal_length_seconds"]
        self.f0 = signal_params["target_frequency_hz"]

        # Read ng/√Hz parameters directly (only supported unit)
        self.signal_asd_ng = signal_params["signal_asd_ng_sqrthz"]  # ng/√Hz
        self.noise_asd_ng = signal_params["noise_asd_ng_sqrthz"]    # ng/√Hz

        self.N = int(self.T * self.fs)

        # Welch parameters
        welch_params = self.config["welch_parameters"]
        self.nperseg = welch_params["nperseg"]
        self.overlap_ratio = welch_params["overlap_ratio"]
        self.noverlap = int(self.nperseg * self.overlap_ratio)
        self.window = welch_params["window"]

        # Channel noise factors (support asymmetric noise)
        self.channel_noise_factors = signal_params.get("channel_noise_factors", [1.0, 1.0, 1.0])

        # Create noise level object (pure ng unit)
        self.noise_level = {
            "amplitude_ng_sqrthz": self.noise_asd_ng,
            "name": "paper_reference",
            "snr_db": 20 * np.log10(self.signal_asd_ng / self.noise_asd_ng)
        }

        # Set random seed
        computation_settings = self.config["computation_settings"]
        if computation_settings.get("random_seed") is not None:
            np.random.seed(computation_settings["random_seed"])

    def generate_three_channel_signals(self):
        """
        Generate three-channel signals (pure ng unit system)

        Signal model: x_i(t) = s(t) + n_i(t)

        Unit system:
        - Time-domain signal: ng (1 ng = 9.80665e-9 m/s²)
        - PSD: ng^2/Hz (Welch method output)
        - ASD: ng/√Hz (configuration parameters and final result)

        Method:
        1. Generate test noise with std=1ng
        2. Compute PSD using Welch method, extract ASD at target frequency
        3. Back-calculate required time-domain std to achieve target ASD
        4. Generate coherent signal and independent noise
        """
        # [Experimental verification] Generate test noise with std=1ng
        test_noise_ng = np.random.normal(0, 1.0, self.N)  # Unit: ng

        # Compute PSD using Welch method
        f_test, psd_test_ng2_hz = signal.welch(
            test_noise_ng, self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling='density'
        )  # Output unit: ng^2/Hz

        # Extract ASD at target frequency
        test_freq_idx = np.argmin(np.abs(f_test - self.f0))
        measured_asd_ng = np.sqrt(psd_test_ng2_hz[test_freq_idx])  # ng/√Hz

        # Back-calculate time-domain standard deviation
        if measured_asd_ng > 0:
            # Target: self.signal_asd_ng, self.noise_asd_ng (ng/√Hz)
            # Measured: measured_asd_ng (ng/√Hz, from noise with σ=1ng)
            # Required: σ_required = target_asd_ng / measured_asd_ng
            sigma_signal_ng = self.signal_asd_ng / measured_asd_ng  # ng
            sigma_noise_ng = self.noise_asd_ng / measured_asd_ng    # ng
        else:
            raise ValueError("Experimental verification failed: measured ASD is zero")

        print(f"[ng unit system] Target signal ASD: {self.signal_asd_ng:.2e} ng/√Hz")
        print(f"[ng unit system] Time-domain signal std: {sigma_signal_ng:.2e} ng")
        print(f"[ng unit system] Target noise ASD: {self.noise_asd_ng:.2e} ng/√Hz")
        print(f"[ng unit system] Time-domain noise std: {sigma_noise_ng:.2e} ng")
        print(f"[Experimental verification] White noise with std=1ng → ASD={measured_asd_ng:.2e} ng/√Hz")

        # Generate coherent signal (ng unit white noise)
        coherent_signal_ng = np.random.normal(0, sigma_signal_ng, self.N)

        # Generate independent noise (ng unit, support asymmetric noise factors)
        noise1_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[0], self.N)
        noise2_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[1], self.N)
        noise3_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[2], self.N)

        # Synthesize three-channel signals
        channel1_ng = coherent_signal_ng + noise1_ng
        channel2_ng = coherent_signal_ng + noise2_ng
        channel3_ng = coherent_signal_ng + noise3_ng

        # Verify statistical properties of generated signals
        actual_noise_std = np.std(noise1_ng)
        print(f"[Verification] Channel 1 noise std: Expected={sigma_noise_ng * self.channel_noise_factors[0]:.2e}, Actual={actual_noise_std:.2e}")

        return channel1_ng, channel2_ng, channel3_ng

    def generate_noise_only_signals(self):
        """
        Generate noise-only signals (for separation testing)

        Unit: ng
        """
        # Generate noise based on experimental verification method
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
            raise ValueError("Experimental verification failed: measured ASD is zero")

        # Generate three-channel pure noise (support asymmetric noise factors)
        noise1_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[0], self.N)
        noise2_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[1], self.N)
        noise3_ng = np.random.normal(0, sigma_noise_ng * self.channel_noise_factors[2], self.N)

        return noise1_ng, noise2_ng, noise3_ng

    def apply_synchronization_delay(self, signals, delays):
        """
        Apply synchronization delays to multi-channel signals

        Parameters:
        - signals: Three-channel signal tuple (channel1, channel2, channel3), unit: ng
        - delays: List of delay samples for each channel [delay1, delay2, delay3]

        Returns:
        - delayed_signals: Signal tuple after applying delays, unit: ng

        Note: Uses np.roll for circular shift, positive values shift right (delay),
        negative values shift left (advance)
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
        Generate signal-only (for separation testing)

        Unit: ng (coherent white noise)
        """
        # Generate signal based on experimental verification method
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
            raise ValueError("Experimental verification failed: measured ASD is zero")

        # Generate coherent white noise signal
        signal_component_ng = np.random.normal(0, sigma_signal_ng, self.N)

        print(f"[Signal generation] Target ASD: {self.signal_asd_ng:.2e} ng/√Hz")
        print(f"[Signal generation] Time-domain std: {sigma_signal_ng:.2e} ng")

        return signal_component_ng

    def apply_permutation(self, ch1, ch2, ch3, permutation):
        """
        Rearrange channels according to permutation

        Parameters:
        - ch1, ch2, ch3: Three-channel signals, unit: ng
        - permutation: e.g., "213" means [ch2, ch1, ch3]

        Returns:
        - Rearranged channel list, unit: ng
        """
        channels = [ch1, ch2, ch3]
        perm_indices = [int(permutation[i]) - 1 for i in range(3)]
        return [channels[idx] for idx in perm_indices]

    def compute_psd(self, x_ng):
        """
        Compute power spectral density

        Parameters:
        - x_ng: Time-domain signal, unit: ng

        Returns:
        - f: Frequency array, unit: Hz
        - psd_ng2_hz: Power spectral density, unit: ng^2/Hz
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
        Compute cross power spectral density

        Parameters:
        - x_ng, y_ng: Time-domain signals, unit: ng

        Returns:
        - f: Frequency array, unit: Hz
        - csd_ng2_hz: Cross power spectral density, unit: ng^2/Hz (complex)
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
        Standard three-channel correlation algorithm - compliant with authoritative literature

        Based on Xu et al. (2017) paper equation (5):
        N_ii = P_ii - (P_ji * P_ik) / P_jk

        Where:
        - N_ii: Channel i self-noise power spectral density, unit: ng^2/Hz
        - P_ii: Channel i auto power spectral density, unit: ng^2/Hz
        - P_ji, P_ik, P_jk: Cross power spectral density, unit: ng^2/Hz (complex)
        - i ≠ j ≠ k ∈ {1,2,3}

        Algorithm principle:
        Separates coherent signal and independent noise through three-channel correlation analysis.
        scipy.signal.csd handles complex conjugate operations for cross power spectra correctly,
        so direct usage is appropriate.

        References:
        Xu W et al., 2017, Chinese J. Geophysics, 60(9):3466-3474, Eq.(5)
        Sleeman R, et al. 2006, BSSA, 96(1): 258-271

        Parameters:
        - signal1_ng, signal2_ng, signal3_ng: Three-channel time-domain signals, unit: ng

        Returns:
        - f: Frequency array, unit: Hz
        - N_11_ng2_hz: Channel 1 self-noise PSD, unit: ng^2/Hz
        """
        # Compute PSD and cross-PSD (unit: ng^2/Hz)
        f, P_11 = self.compute_psd(signal1_ng)
        _, P_12 = self.compute_cross_psd(signal1_ng, signal2_ng)  # P_ji = P_12
        _, P_13 = self.compute_cross_psd(signal1_ng, signal3_ng)  # P_ik = P_13
        _, P_23 = self.compute_cross_psd(signal2_ng, signal3_ng)  # P_jk = P_23

        # Apply paper equation (5): N_ii = P_ii - (P_ji * P_ik) / P_jk
        # Analyze first channel here (i=1, j=2, k=3)
        N_11 = P_11 - (P_12 * P_13) / P_23

        # [Debug] Check computation at target frequency
        freq_idx = self.find_frequency_index(f)
        if freq_idx < len(f):
            print(f"[Three-channel algorithm debug] Spectral densities at {self.f0}Hz:")
            print(f"  P_11 (Ch1 auto-spectrum): {P_11[freq_idx]:.2e} ng^2/Hz")
            print(f"  P_12 (1-2 cross-spectrum): {P_12[freq_idx]:.2e} ng^2/Hz")
            print(f"  P_13 (1-3 cross-spectrum): {P_13[freq_idx]:.2e} ng^2/Hz")
            print(f"  P_23 (2-3 cross-spectrum): {P_23[freq_idx]:.2e} ng^2/Hz")
            print(f"  Cross term (P_12*P_13)/P_23: {(P_12[freq_idx] * P_13[freq_idx]) / P_23[freq_idx]:.2e} ng^2/Hz")
            print(f"  Correlated N_11: {N_11[freq_idx]:.2e} ng^2/Hz")
            print(f"  Theoretical noise PSD: {self.noise_asd_ng**2:.2e} ng^2/Hz")

        # Take real part and ensure positive values (physically noise PSD must be positive)
        N_11_ng2_hz = np.real(N_11)

        # Handle possible negative values (indicates algorithm assumptions not satisfied)
        if np.any(N_11_ng2_hz < 0):
            negative_ratio = np.sum(N_11_ng2_hz < 0) / len(N_11_ng2_hz) * 100
            warnings.warn(f"Detected {negative_ratio:.1f}% negative values, possibly indicating signal model assumptions not satisfied")

        return f, N_11_ng2_hz

    def find_frequency_index(self, frequencies):
        """Find index of target frequency"""
        return np.argmin(np.abs(frequencies - self.f0))

    def extract_noise_amplitude_at_frequency(self, frequencies, noise_psd_ng2_hz):
        """
        Extract noise amplitude at target frequency

        Parameters:
        - frequencies: Frequency array, unit: Hz
        - noise_psd_ng2_hz: Noise PSD, unit: ng^2/Hz

        Returns:
        - amplitude_ng_sqrthz: Noise ASD, unit: ng/√Hz
        """
        freq_idx = self.find_frequency_index(frequencies)
        amplitude_ng_sqrthz = np.sqrt(np.abs(noise_psd_ng2_hz[freq_idx]))
        return amplitude_ng_sqrthz

    def get_noise_asd(self, noise_psd_ng2_hz):
        """
        Compute ASD from PSD

        Parameters:
        - noise_psd_ng2_hz: Noise PSD array, unit: ng^2/Hz

        Returns:
        - noise_asd_ng_sqrthz: Noise ASD array, unit: ng/√Hz
        """
        return np.sqrt(np.abs(noise_psd_ng2_hz))

    def get_output_unit(self):
        """
        Get output unit

        Returns:
        - Fixed return "ng/sqrt(Hz)"
        """
        return "ng/sqrt(Hz)"
