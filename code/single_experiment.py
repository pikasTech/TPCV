#!/usr/bin/env python3
"""
Single experiment calculation: compute results for 3 physical channels under 6 permutations
"""

from core_algorithm import ThreeChannelCorrelation
from common_sync_algorithm import CommonSyncAlgorithm
# from file_lifecycle_manager import FileLifecycleManager  # removed
import json
import os
import numpy as np
from datetime import datetime

class SingleExperiment:
    """Single experiment: generate data once, analyze with 6 permutations"""

    def __init__(self, exam_name):
        """
        Initialize experiment

        Args:
            exam_name: Experiment name (required)
        """
        if not exam_name:
            raise ValueError("exam_name parameter is required")

        self.exam_name = exam_name

        # Read configuration from exams directory
        config_file = os.path.join("exams", exam_name, "config.json")
        if not os.path.exists(config_file):
            raise ValueError(f"Configuration file not found: {config_file}")

        self.config_file = config_file
        self.core = ThreeChannelCorrelation(config_file)
        self.permutations = ["123", "132", "213", "231", "312", "321"]

        # Read experimental conditions (synchronization delays)
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.config = config  # Save complete configuration
        self.sync_delays = config.get('experimental_conditions', {}).get('synchronization_delays', {
            'channel_1_samples': 0,
            'channel_2_samples': 0,
            'channel_3_samples': 0
        })

    def run_single_experiment(self, save_waveform=False, waveform_duration_seconds=1.0):
        """
        Run single experiment

        Args:
            save_waveform: Whether to save raw waveform data
            waveform_duration_seconds: Duration of waveform to save (seconds)

        Returns: Calculation results for 3 physical channels under 6 permutations
        """
        print("Running single experiment: 3 physical channels × 6 permutations")
        print("=" * 50)

        # No backup needed, as each run uses a new timestamped directory

        # Generate three-channel data once
        physical_ch1, physical_ch2, physical_ch3 = self.core.generate_three_channel_signals()

        # Apply synchronization delays (if configured)
        delays = [
            self.sync_delays['channel_1_samples'],
            self.sync_delays['channel_2_samples'],
            self.sync_delays['channel_3_samples']
        ]

        if any(d != 0 for d in delays):
            print(f"\nApplying synchronization delays: Ch1={delays[0]}, Ch2={delays[1]}, Ch3={delays[2]} samples")
            physical_ch1, physical_ch2, physical_ch3 = self.core.apply_synchronization_delay(
                [physical_ch1, physical_ch2, physical_ch3], delays
            )

        # Save raw waveform data (if requested)
        waveform_data = None
        if save_waveform:
            # Calculate number of samples to save
            samples_to_save = int(waveform_duration_seconds * self.core.fs)

            # Save first N samples
            waveform_data = {
                'channel_1': physical_ch1[:samples_to_save].tolist(),
                'channel_2': physical_ch2[:samples_to_save].tolist(),
                'channel_3': physical_ch3[:samples_to_save].tolist(),
                'sampling_rate': self.core.fs,
                'duration_seconds': waveform_duration_seconds,
                'delays': delays,
                'time': (np.arange(samples_to_save) / self.core.fs).tolist()  # time axis
            }
            print(f"Saved {waveform_duration_seconds} seconds of raw waveform data")

        print(f"Noise level: {self.core.noise_asd_ng:.2e} ng/√Hz")
        print(f"Signal level: {self.core.signal_asd_ng:.2e} ng/√Hz")
        print(f"SNR: {self.core.noise_level['snr_db']:.1f} dB")
        print(f"Target frequency: {self.core.f0} Hz")
        print(f"Sampling rate: {self.core.fs} Hz")
        if any(d != 0 for d in delays):
            print(f"Synchronization delays: Ch1={delays[0]} samples ({delays[0]/self.core.fs*1000:.1f}ms), "
                  f"Ch2={delays[1]} samples ({delays[1]/self.core.fs*1000:.1f}ms), "
                  f"Ch3={delays[2]} samples ({delays[2]/self.core.fs*1000:.1f}ms)")
        print()

        # Calculate original noise ASD before correlation (at 10Hz)
        print("Computing original noise ASD before correlation...")

        # Academic integrity principle: must use actual measured values, absolutely forbidden to substitute with theoretical values
        # If actual measured values do not match expectations, must find and fix the root cause of calculation errors

        # Compute actual PSD
        f1, psd1 = self.core.compute_psd(physical_ch1)
        f2, psd2 = self.core.compute_psd(physical_ch2)
        f3, psd3 = self.core.compute_psd(physical_ch3)

        # Extract actual measured values at 10Hz (must use real data)
        original_noise_ch1 = self.core.extract_noise_amplitude_at_frequency(f1, psd1)
        original_noise_ch2 = self.core.extract_noise_amplitude_at_frequency(f2, psd2)
        original_noise_ch3 = self.core.extract_noise_amplitude_at_frequency(f3, psd3)

        # Separation test: compute pure noise and pure signal ASD separately to locate problem source
        print("Performing separation test: computing pure noise and pure signal ASD separately...")

        # Generate pure noise with same parameters (no signal)
        noise_only_ch1, noise_only_ch2, noise_only_ch3 = self.core.generate_noise_only_signals()
        f_noise, psd_noise = self.core.compute_psd(noise_only_ch1)
        noise_only_amplitude = self.core.extract_noise_amplitude_at_frequency(f_noise, psd_noise)

        # Generate pure signal with same parameters (no noise)
        signal_only_ch1 = self.core.generate_signal_only()
        f_signal, psd_signal = self.core.compute_psd(signal_only_ch1)
        signal_only_amplitude = self.core.extract_noise_amplitude_at_frequency(f_signal, psd_signal)

        print(f"[Separation Test Results]")
        print(f"  Pure noise ASD: {noise_only_amplitude:.2e} ng/√Hz (expected: {self.core.noise_asd_ng:.2e})")
        print(f"  Pure signal ASD: {signal_only_amplitude:.2e} ng/√Hz (expected: {self.core.signal_asd_ng:.2e})")
        print(f"  Mixed signal measured: {original_noise_ch1:.2e} ng/√Hz")

        # Theoretical calculation of mixed PSD
        theoretical_noise = self.core.noise_asd_ng
        theoretical_signal = self.core.signal_asd_ng
        theoretical_mixed_asd = np.sqrt(theoretical_noise**2 + theoretical_signal**2)
        experimental_mixed_asd = np.sqrt(noise_only_amplitude**2 + signal_only_amplitude**2)

        print(f"[Comparison Analysis]")
        print(f"  Theoretical mixed PSD: {theoretical_mixed_asd:.2e} ng/√Hz")
        print(f"  Experimental mixed PSD: {experimental_mixed_asd:.2e} ng/√Hz")
        print(f"  Actual measured ASD: {original_noise_ch1:.2e} ng/√Hz")

        # Check consistency of each component
        noise_ratio = noise_only_amplitude / theoretical_noise
        signal_ratio = signal_only_amplitude / theoretical_signal
        print(f"[Consistency Check]")
        print(f"  Noise measured/theoretical: {noise_ratio:.2f}")
        print(f"  Signal measured/theoretical: {signal_ratio:.2f}")

        print(f"Original noise before correlation - Physical channel 1: {original_noise_ch1:.2f} ng/√Hz")
        print(f"Original noise before correlation - Physical channel 2: {original_noise_ch2:.2f} ng/√Hz")
        print(f"Original noise before correlation - Physical channel 3: {original_noise_ch3:.2f} ng/√Hz")
        print()

        # Store results: each physical channel's results under 6 permutations and original noise
        results = {
            "physical_channel_1": {"original_noise": original_noise_ch1},
            "physical_channel_2": {"original_noise": original_noise_ch2},
            "physical_channel_3": {"original_noise": original_noise_ch3}
        }

        # Analyze each permutation
        for perm in self.permutations:
            print(f"Permutation {perm}:", end=" ")

            # Apply permutation
            arranged_signals = self.core.apply_permutation(
                physical_ch1, physical_ch2, physical_ch3, perm)

            # Perform three-channel correlation analysis (always analyze first position)
            frequencies, noise_psd = self.core.three_channel_correlation(
                arranged_signals[0], arranged_signals[1], arranged_signals[2])

            # Extract result at target frequency (pure ng system, output directly in ng/√Hz)
            amplitude = self.core.extract_noise_amplitude_at_frequency(frequencies, noise_psd)

            # Determine which original physical channel is being analyzed
            analyzed_physical_channel = int(perm[0])

            # Store result
            if analyzed_physical_channel == 1:
                results["physical_channel_1"][perm] = amplitude
                print(f"Analyzing physical channel 1 → {amplitude:.2f} ng/√Hz")
            elif analyzed_physical_channel == 2:
                results["physical_channel_2"][perm] = amplitude
                print(f"Analyzing physical channel 2 → {amplitude:.2f} ng/√Hz")
            else:  # analyzed_physical_channel == 3
                results["physical_channel_3"][perm] = amplitude
                print(f"Analyzing physical channel 3 → {amplitude:.2f} ng/√Hz")

        # If waveform data was saved, include it in results
        if waveform_data is not None:
            results["waveform_data"] = waveform_data

        return results

    def analyze_symmetry(self, results):
        """Analyze theoretical symmetry"""
        print("\\nTheoretical Symmetry Analysis")
        print("=" * 30)

        # Theoretical grouping: each physical channel should be analyzed by 2 permutations
        theoretical_groups = {
            "Physical channel 1": ["123", "132"],  # 123 and 132 both place physical channel 1 first
            "Physical channel 2": ["213", "231"],  # 213 and 231 both place physical channel 2 first
            "Physical channel 3": ["312", "321"]   # 312 and 321 both place physical channel 3 first
        }

        symmetry_verified = True

        for channel_name, perms in theoretical_groups.items():
            channel_key = channel_name.replace("Physical channel ", "physical_channel_")

            # Get the two results for this physical channel
            if channel_key in results:
                channel_results = results[channel_key]

                if len(channel_results) >= 2:
                    values = list(channel_results.values())
                    val1, val2 = values[0], values[1]
                    diff = abs(val1 - val2)
                    diff_percent = diff / val1 * 100 if val1 != 0 else 0

                    print(f"{channel_name}: {val1:.2f} vs {val2:.2f}")
                    print(f"  Difference: {diff:.2e} ({diff_percent:.4f}%)")

                    if diff_percent > 0.01:  # Consider significant if > 0.01%
                        symmetry_verified = False
                        print(f"  Warning: Symmetry verification failed")
                    else:
                        print(f"  Passed: Symmetry verification successful")
                else:
                    print(f"{channel_name}: Insufficient data")
                    symmetry_verified = False
            else:
                print(f"{channel_name}: No data")
                symmetry_verified = False

        print(f"\\nOverall symmetry verification: {'Passed' if symmetry_verified else 'Failed'}")
        return symmetry_verified, theoretical_groups

    def save_results(self, results, symmetry_verified, theoretical_groups):
        """Save experiment results"""
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to exams/{exam_name}/output/{timestamp}/
        output_dir = os.path.join("exams", self.exam_name, "output", timestamp)

        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "results.json")

        # Convert to JSON-compatible format
        json_results = {}
        waveform_data = None  # Initialize waveform data variable

        for channel, channel_data in results.items():
            # Special handling for waveform_data
            if channel == 'waveform_data':
                waveform_data = channel_data
                continue

            json_results[channel] = {
                perm: float(value) for perm, value in channel_data.items()
            }

        # Add metadata (pure ng unit system)
        metadata = {
            "experiment_type": self.config.get('experiment_type', 'single_three_channel_experiment'),
            "experiment_id": self.exam_name if self.exam_name else "default",
            "timestamp": timestamp,
            "timestamp_iso": datetime.now().isoformat(),
            "config_file": self.config_file,
            "noise_amplitude_ng_sqrthz": self.core.noise_asd_ng,
            "signal_amplitude_ng_sqrthz": self.core.signal_asd_ng,
            "snr_db": self.core.noise_level["snr_db"],
            "target_frequency_hz": self.core.f0,
            "sampling_rate_hz": self.core.fs,
            "synchronization_delays": self.sync_delays,
            "algorithm": "three_channel_correlation_sleeman",
            "theoretical_symmetry_verified": symmetry_verified,
            "theoretical_groups": theoretical_groups,
            "note": f"All results are noise amplitude values at {self.core.f0}Hz",
            "units": {
                "noise_asd": "ng/sqrt(Hz)",
                "frequency": "Hz"
            }
        }

        output_data = {
            "metadata": metadata,
            "results": json_results
        }

        # If waveform data exists, add it separately
        if waveform_data is not None:
            output_data["waveform_data"] = waveform_data
            print(f"  Saved waveform data: {len(waveform_data['channel_1'])} sample points")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\\nExperiment results saved to: {output_file}")
        return output_file

#
#     print("[ERROR] Direct invocation of this module is prohibited!")
#     print("=" * 50)
#     print("This module is an experiment control component and cannot be run directly.")
#     print()
#     print("Correct usage:")
#     print("cd code/")
#     print("python cli.py --experiment single")
#     print()
#     print("[Project Guidelines]")
#     print("  - All functionality must be accessed through the cli.py unified interface")
#     print("  - Direct invocation of any module in lib/ directory is prohibited")
#     print("  - This ensures system consistency and maintainability")
#     print("=" * 50)
#     exit(1)
