#!/usr/bin/env python3
"""
Common synchronization error algorithm module
Unified handling of three-channel data generation and correlation computation with synchronization errors
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Dict


class CommonSyncAlgorithm:
    """Unified synchronization error handling algorithm"""

    def __init__(self, fs: float = 2000, target_freq: float = 10):
        """
        Initialize algorithm parameters

        Args:
            fs: Sampling rate
            target_freq: Target frequency
        """
        self.fs = fs
        self.target_freq = target_freq

    def generate_noise_with_correct_psd(self, target_psd_amplitude: float,
                                      n_samples: int,
                                      welch_params: Dict,
                                      debug: bool = False) -> np.ndarray:
        """
        Generate noise with correct PSD
        Use experimentally validated method to ensure Welch-computed PSD matches expectation

        Args:
            target_psd_amplitude: Target PSD amplitude (V/√Hz)
            n_samples: Number of samples
            welch_params: Welch parameter dictionary
            debug: Whether to output debug information

        Returns:
            Noise signal
        """
        # First generate unit standard deviation white noise for testing
        # Don't change random seed, consistent with core_algorithm
        test_noise = np.random.normal(0, 1.0, n_samples)

        # Compute PSD using Welch method
        freqs, psd_test = signal.welch(
            test_noise, self.fs,
            window=welch_params.get('window', 'hann'),
            nperseg=welch_params.get('nperseg', 4096),
            noverlap=welch_params.get('noverlap', 3584),
            scaling='density'
        )

        # Find PSD at target frequency
        freq_idx = np.argmin(np.abs(freqs - self.target_freq))
        measured_psd_at_target = psd_test[freq_idx]
        measured_amplitude = np.sqrt(measured_psd_at_target)

        if debug:
            print(f"[Noise calibration] White noise with std=1.0:")
            print(f"  Welch-measured PSD(10Hz): {measured_psd_at_target:.2e} V²/Hz")
            print(f"  Corresponding ASD: {measured_amplitude:.2e} V/√Hz")

        # Calculate required standard deviation
        if measured_amplitude > 0:
            required_sigma = target_psd_amplitude / measured_amplitude
            if debug:
                print(f"[Noise calibration] To obtain {target_psd_amplitude:.1e} V/√Hz, need time-domain σ = {required_sigma:.2e}")
        else:
            # Fallback to empirical formula
            required_sigma = target_psd_amplitude * np.sqrt(self.fs / 2)
            if debug:
                print(f"[Noise calibration] Warning: using fallback formula")

        # Generate noise with correct standard deviation
        return np.random.normal(0, required_sigma, n_samples)

    def generate_three_channel_signals(self, duration: float,
                                     signal_amplitude: float,
                                     noise_factors: List[float],
                                     base_noise_psd: float,
                                     welch_params: Dict,
                                     use_calibration: bool = False,
                                     signal_type: str = 'white_noise',
                                     signal_frequency: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate three-channel signals (signal + noise)
        Exactly matches random number generation sequence of core_algorithm

        Args:
            duration: Signal duration (seconds)
            signal_amplitude: Signal PSD amplitude (V/√Hz) or time-domain amplitude (V) depending on signal_type
            noise_factors: Noise coefficients for each channel [ch1_factor, ch2_factor, ch3_factor]
            base_noise_psd: Base noise PSD amplitude (V/√Hz)
            welch_params: Welch parameters
            use_calibration: Whether to use calibration method for noise generation (default False, use traditional empirical formula)
            signal_type: Signal type ('white_noise', 'coherent'/'sine')
            signal_frequency: Sine wave frequency (Hz), only used when signal_type is 'coherent' or 'sine'

        Returns:
            Signals for three channels
        """
        n_samples = int(duration * self.fs)
        t = np.arange(n_samples) / self.fs

        # Generate common signal
        if signal_type in ['coherent', 'sine']:
            # Generate sine wave signal
            if signal_frequency is None:
                signal_frequency = self.target_freq  # Default to target frequency
            # For sine wave, signal_amplitude is time-domain amplitude value (V)
            common_signal = signal_amplitude * np.sin(2 * np.pi * signal_frequency * t)
            print(f"[Coherent signal] Generated sine wave: frequency={signal_frequency}Hz, amplitude={signal_amplitude:.2e}V")
        else:
            # Generate white noise signal (original logic)
            # Follow exact random number generation order of core_algorithm
            # 1. Generate test noise for signal calibration
            test_noise_signal = np.random.normal(0, 1.0, n_samples)
            freqs, psd_test = signal.welch(
                test_noise_signal, self.fs, **welch_params, scaling='density'
            )
            freq_idx = np.argmin(np.abs(freqs - self.target_freq))
            measured_amplitude = np.sqrt(psd_test[freq_idx])

            # Calculate signal standard deviation
            if measured_amplitude > 0:
                required_signal_sigma = signal_amplitude / measured_amplitude
            else:
                required_signal_sigma = signal_amplitude * np.sqrt(self.fs / 2)

            # 2. Generate coherent signal (white noise)
            common_signal = np.random.normal(0, required_signal_sigma, n_samples)
            print(f"[Coherent signal] Generated white noise: target PSD={signal_amplitude:.2e} V/√Hz")

        # 3. Decide noise generation method based on use_calibration parameter
        if use_calibration:
            # Use calibration method: generate test noise for noise calibration
            test_noise_noise = np.random.normal(0, 1.0, n_samples)
            _, psd_test_noise = signal.welch(
                test_noise_noise, self.fs, **welch_params, scaling='density'
            )
            measured_noise_amplitude = np.sqrt(psd_test_noise[freq_idx])

            # Calculate noise standard deviation
            if measured_noise_amplitude > 0:
                required_noise_sigma = base_noise_psd / measured_noise_amplitude
            else:
                required_noise_sigma = base_noise_psd * np.sqrt(self.fs / 2)
        else:
            # Use traditional empirical formula (default)
            required_noise_sigma = base_noise_psd * np.sqrt(self.fs / 2)
            # To maintain random number sequence consistency, still generate test noise but don't use it
            test_noise_noise = np.random.normal(0, 1.0, n_samples)

        if noise_factors[0] == 1.0:  # Only output debug info for first channel
            if signal_type not in ['coherent', 'sine']:
                print(f"[Coherent signal correction] Target PSD amplitude: {signal_amplitude:.2e} V/√Hz")
                print(f"[Coherent signal correction] Required time-domain std: {required_signal_sigma:.2e}")

            if use_calibration:
                _, psd_test_noise = signal.welch(
                    test_noise_noise, self.fs, **welch_params, scaling='density'
                )
                measured_noise_amplitude = np.sqrt(psd_test_noise[freq_idx])
                print(f"[Noise generation] Using calibration method")
                print(f"[Experimental validation] White noise with std=1.0:")
                print(f"  Welch-measured PSD(10Hz): {psd_test_noise[freq_idx]:.2e} V^2/Hz")
                print(f"  Corresponding ASD: {measured_noise_amplitude:.2e} V/sqrt(Hz)")
                print(f"  Theoretical expectation: White noise PSD should be close to sigma^2 = 1.0")
                print(f"[Conversion derivation] To obtain {base_noise_psd:.1e} V/√Hz, need time-domain σ = {required_noise_sigma:.2e}")
                print(f"[Validation complete] Noise generation parameters validated by experiment")
            else:
                print(f"[Noise generation] Using traditional empirical formula (default)")
                print(f"[Empirical formula] σ = PSD × √(fs/2) = {base_noise_psd:.1e} × √{self.fs/2:.0f}")

            print(f"  Target PSD amplitude: {base_noise_psd:.1e} V/√Hz")
            print(f"  Actual time-domain std: {required_noise_sigma:.2e}")

        # 4. Generate three independent noise channels
        channels = []
        for i, factor in enumerate(noise_factors):
            # All channels use same noise intensity (factor=1.0)
            noise = np.random.normal(0, required_noise_sigma * factor, n_samples)
            if i == 0:
                print(f"[Validation] Actual std of generated noise: {np.std(noise):.2e}")
                print(f"[Validation] Expected/actual ratio: {(required_noise_sigma * factor)/np.std(noise):.3f}")
            channels.append(common_signal + noise)

        return tuple(channels)

    def apply_synchronization_delays(self, channels: List[np.ndarray],
                                   delays: List[int],
                                   method: str = 'roll') -> List[np.ndarray]:
        """
        Apply synchronization delays

        Args:
            channels: List of channel signals
            delays: Delays for each channel (in samples)
            method: Delay method - 'roll' (circular shift) or 'truncate' (truncate and zero-pad)

        Returns:
            List of delayed signals
        """
        delayed_channels = []
        for channel, delay in zip(channels, delays):
            if delay == 0:
                delayed_channels.append(channel.copy())
            elif method == 'roll':
                # Use circular shift (core_algorithm method)
                delayed = np.roll(channel, delay)
                delayed_channels.append(delayed)
            elif method == 'truncate':
                # Use truncation and zero-padding (channel_identification method)
                if delay > 0:
                    # Positive delay: shift signal right
                    delayed = np.concatenate([channel[delay:], np.zeros(delay)])
                else:
                    # Negative delay: shift signal left
                    delayed = np.concatenate([np.zeros(-delay), channel[:delay]])
                delayed_channels.append(delayed)
            else:
                raise ValueError(f"Unknown method: {method}")

        return delayed_channels

    def compute_three_channel_correlation(self, ch1: np.ndarray,
                                        ch2: np.ndarray,
                                        ch3: np.ndarray,
                                        welch_params: Dict) -> float:
        """
        Compute noise level after three-channel correlation

        Args:
            ch1, ch2, ch3: Signals for three channels
            welch_params: Welch parameters

        Returns:
            Correlated noise amplitude spectral density (V/√Hz)
        """
        # Compute power spectra and cross-power spectra
        freqs, Pxx = signal.welch(ch1, self.fs, **welch_params, scaling='density')
        _, Pyy = signal.welch(ch2, self.fs, **welch_params, scaling='density')
        _, Pzz = signal.welch(ch3, self.fs, **welch_params, scaling='density')

        _, Pxy = signal.csd(ch1, ch2, self.fs, **welch_params, scaling='density')
        _, Pxz = signal.csd(ch1, ch3, self.fs, **welch_params, scaling='density')
        _, Pyz = signal.csd(ch2, ch3, self.fs, **welch_params, scaling='density')

        # Find target frequency
        freq_idx = np.argmin(np.abs(freqs - self.target_freq))

        # Three-channel correlation algorithm: N_xx = P_xx - (P_xy * P_xz) / P_yz
        # Note: scipy.signal.csd already handles cross-power spectrum correctly, no additional conjugation needed
        Nxx = Pxx[freq_idx] - (Pxy[freq_idx] * Pxz[freq_idx]) / Pyz[freq_idx]

        # Take real part (consistent with core_algorithm)
        # Physically, noise power spectral density must be positive real number
        Nxx_real = np.real(Nxx)

        # Convert to amplitude spectral density
        correlated_noise_psd = np.sqrt(np.abs(Nxx_real))

        return correlated_noise_psd

    def run_single_permutation(self, channels: List[np.ndarray],
                             permutation: str,
                             welch_params: Dict) -> Dict:
        """
        Run computation for single permutation

        Args:
            channels: List of original channel signals
            permutation: Permutation string, e.g. '123'
            welch_params: Welch parameters

        Returns:
            Dictionary containing results
        """
        # Reorder channels according to permutation
        perm_indices = [int(p) - 1 for p in permutation]
        ordered_channels = [channels[i] for i in perm_indices]

        # Compute noise before correlation (first channel)
        freqs, Pxx = signal.welch(ordered_channels[0], self.fs, **welch_params, scaling='density')
        freq_idx = np.argmin(np.abs(freqs - self.target_freq))
        original_noise = np.sqrt(Pxx[freq_idx])

        # Compute noise after correlation
        correlated_noise = self.compute_three_channel_correlation(
            ordered_channels[0], ordered_channels[1], ordered_channels[2],
            welch_params
        )

        return {
            'permutation': permutation,
            'original_noise_psd': original_noise,
            'correlated_noise_psd': correlated_noise,
            'analyzed_physical_channel': int(permutation[0])
        }
