#!/usr/bin/env python3
"""
External Interference Intensity Gradient Analysis Module

Systematically studies the effect of external interference signal intensity and
synchronization errors on self-noise measurement accuracy under fixed self-noise conditions.

Based on the standard three-channel correlation algorithm from authoritative paper
by Xu et al. (2017), conducting systematic tests across multiple external interference
intensity levels.
"""

import os
import json
import numpy as np
from datetime import datetime
from core_algorithm import ThreeChannelCorrelation
from common_sync_algorithm import CommonSyncAlgorithm

class NoiseGradientAnalysis:
    """External interference intensity gradient analysis class"""

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_parameters()

    def load_config(self):
        """Load experimental configuration"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def setup_parameters(self):
        """Setup experimental parameters"""
        self.gradient_config = self.config["external_interference_config"]
        self.fixed_noise = self.gradient_config["fixed_noise_level_ng_sqrthz"]
        self.signal_multipliers = self.gradient_config["signal_multipliers"]
        self.signal_level_labels = self.gradient_config["signal_level_labels"]

        # Setup synchronization error configuration
        self.sync_config = self.config["experimental_conditions"]["synchronization_error_config"]
        self.enable_sync_analysis = self.sync_config.get("enable_sync_analysis", False)
        self.sync_conditions = self.sync_config.get("sync_error_conditions", [])
        self.sync_seed = self.sync_config.get("sync_seed", 123)

        # Setup output directory - using unified timestamp
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_datetime = datetime.now()
        exam_name = self.config["experiment_name"]
        self.output_dir = os.path.join("exams", exam_name, "output", self.experiment_timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_sync_delays(self, delay_spec, sync_rng):
        """
        Generate synchronization error delays based on configuration

        Args:
            delay_spec: Delay specification (list or string)
            sync_rng: Random number generator

        Returns:
            [ch1_delay, ch2_delay, ch3_delay] List of delays in samples
        """
        if isinstance(delay_spec, list):
            # Fixed delay
            return delay_spec
        elif isinstance(delay_spec, str):
            if delay_spec.startswith("random_within_"):
                # Extract value from string
                max_delay = int(delay_spec.split("_")[-1])
                # Generate random delays within ±max_delay range
                delays = sync_rng.randint(-max_delay, max_delay + 1, size=3)
                return delays.tolist()
            else:
                raise ValueError(f"Unsupported delay specification: {delay_spec}")
        else:
            raise ValueError(f"Invalid delay specification type: {type(delay_spec)}")
    
    def run_analysis(self):
        """Run complete external interference gradient analysis (supports 2D: external interference × sync error)"""
        print("External Interference Intensity Gradient Analysis + Synchronization Error Analysis")
        print("=" * 60)
        print(f"Fixed self-noise intensity: {self.fixed_noise} ng/√Hz")
        print(f"External interference multipliers: {self.signal_multipliers}")

        if self.enable_sync_analysis:
            print(f"Synchronization error conditions: {len(self.sync_conditions)} types")
            for sync_cond in self.sync_conditions:
                print(f"  - {sync_cond['label']}: {sync_cond['description']}")
        else:
            print("Synchronization error analysis: Disabled")

        print(f"Output directory: {self.output_dir}")
        print()

        # Initialize synchronization error random number generator
        sync_rng = np.random.RandomState(self.sync_seed)

        # Collect results from all experimental conditions - 2D matrix
        gradient_results = []
        experiment_count = 0
        total_experiments = len(self.signal_multipliers) * (len(self.sync_conditions) if self.enable_sync_analysis else 1)
        
        for i, (signal_multiplier, signal_label) in enumerate(zip(self.signal_multipliers, self.signal_level_labels)):
            current_signal = self.config["signal_parameters"]["signal_asd_ng_sqrthz"] * signal_multiplier
            
            print(f"\n{'='*50}")
            print(f"[Interference level {i+1}/{len(self.signal_multipliers)}] {signal_label}")
            print(f"External interference intensity: {current_signal:.2e} ng/√Hz")
            print(f"Fixed self-noise intensity: {self.fixed_noise:.2e} ng/√Hz")
            print(f"{'='*50}")

            # Get synchronization condition list (if sync analysis enabled)
            sync_conditions_to_test = self.sync_conditions if self.enable_sync_analysis else [
                {"label": "Ideal sync", "delays": [0, 0, 0], "description": "No sync error"}
            ]

            for j, sync_condition in enumerate(sync_conditions_to_test):
                experiment_count += 1
                sync_label = sync_condition["label"]

                print(f"\n[{experiment_count}/{total_experiments}] Sync condition: {sync_label}")

                # Generate synchronization delays
                delays = self.generate_sync_delays(sync_condition["delays"], sync_rng)
                print(f"Sync delays: CH1={delays[0]}, CH2={delays[1]}, CH3={delays[2]} samples")

                # Create temporary configuration for current condition
                temp_config = self.create_temp_config(current_signal, delays)

                # Run three-channel correlation analysis
                from single_experiment import SingleExperiment

                # Create temporary experiment directory structure
                temp_exam_name = f"temp_signal_{i}_sync_{j}"
                temp_exam_dir = os.path.join(self.output_dir, temp_exam_name)
                os.makedirs(temp_exam_dir, exist_ok=True)

                # Copy configuration file to temporary directory
                temp_config_path = os.path.join(temp_exam_dir, "config.json")
                with open(temp_config_path, 'w', encoding='utf-8') as f:
                    json.dump(temp_config, f, indent=2, ensure_ascii=False)

                # Create temporary experiment directory structure
                os.makedirs(os.path.join("exams", temp_exam_name), exist_ok=True)
                with open(os.path.join("exams", temp_exam_name, "config.json"), 'w', encoding='utf-8') as f:
                    json.dump(temp_config, f, indent=2, ensure_ascii=False)

                # Run experiment
                experiment = SingleExperiment(temp_exam_name)
                result = experiment.run_single_experiment()

                # Clean up temporary experiment directories
                import shutil
                shutil.rmtree(os.path.join("exams", temp_exam_name))
                shutil.rmtree(temp_exam_dir)

                # Calculate key performance metrics
                performance_metrics = self.calculate_performance_metrics(result, signal_multiplier, delays)
                
                gradient_results.append({
                    "signal_multiplier": signal_multiplier,
                    "external_signal_level_ng_sqrthz": current_signal,
                    "fixed_noise_level_ng_sqrthz": self.fixed_noise,
                    "signal_label": signal_label,
                    "sync_condition": sync_label,
                    "sync_delays": delays,
                    "sync_description": sync_condition["description"],
                    "raw_results": result,
                    "performance_metrics": performance_metrics
                })
                
                print(f"[OK] Completed {signal_label} × {sync_label}")

        print(f"\n[DONE] All experiments completed! Total {len(gradient_results)} condition combinations")

        # Save comprehensive analysis results (compliant with project standard format)
        analysis_results = {
            "metadata": {
                "experiment_type": self.config["experiment_type"],
                "experiment_id": self.config["experiment_name"],
                "timestamp": self.experiment_timestamp,
                "timestamp_iso": self.experiment_datetime.isoformat(),
                "config_file": self.config_file,
                "fixed_noise_level_ng_sqrthz": self.fixed_noise,
                "signal_multipliers": self.signal_multipliers,
                "signal_amplitude_ng_sqrthz": self.config["signal_parameters"]["signal_asd_ng_sqrthz"],
                "algorithm": "three_channel_correlation_interference_gradient_2d",
                "description": self.config["description"],
                # 2D experiment metadata
                "experiment_dimensions": "2D_signal_sync_analysis",
                "sync_analysis_enabled": self.enable_sync_analysis,
                "sync_conditions": len(self.sync_conditions) if self.enable_sync_analysis else 0,
                "total_condition_combinations": len(gradient_results)
            },
            "gradient_analysis": gradient_results,
            "summary_statistics": self.calculate_summary_statistics(gradient_results),
            "results": {
                "gradient_data": gradient_results,
                "analysis_complete": True,
                "total_condition_combinations": len(gradient_results),
                "signal_levels_tested": len(self.signal_multipliers),
                "sync_conditions_tested": len(self.sync_conditions) if self.enable_sync_analysis else 1,
                "experiment_matrix_size": f"{len(self.signal_multipliers)}×{len(self.sync_conditions) if self.enable_sync_analysis else 1}"
            }
        }

        # Save results file
        results_file = os.path.join(self.output_dir, "results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        print(f"External interference gradient analysis completed!")
        print(f"Results saved to: {results_file}")

        return self.output_dir, analysis_results

    def create_temp_config(self, signal_level, sync_delays):
        """Create temporary configuration for specific external interference level and sync delays"""
        import copy
        temp_config = copy.deepcopy(self.config)
        # Set external signal intensity (interference)
        temp_config["signal_parameters"]["signal_asd_ng_sqrthz"] = signal_level
        # Set fixed self-noise intensity
        temp_config["signal_parameters"]["noise_asd_ng_sqrthz"] = self.fixed_noise

        # Set synchronization delays
        temp_config["experimental_conditions"]["synchronization_delays"] = {
            "channel_1_samples": sync_delays[0],
            "channel_2_samples": sync_delays[1],
            "channel_3_samples": sync_delays[2],
            "description": f"Experimental condition: External interference={signal_level:.2e} ng/√Hz, Fixed self-noise={self.fixed_noise:.2e} ng/√Hz, Sync delays=[{sync_delays[0]}, {sync_delays[1]}, {sync_delays[2]}]"
        }

        return temp_config

    def calculate_performance_metrics(self, result, signal_multiplier, sync_delays):
        """Calculate performance metrics for current external interference level and sync condition"""
        # Extract data from SingleExperiment results
        # result format: {"physical_channel_1": {"original_noise": x, "123": y, "132": z}, ...}

        if not result:
            return {"error": "No valid experimental results"}

        # Extract noise estimates from all permutations
        noise_estimates = []
        permutation_labels = []

        for channel_key, channel_data in result.items():
            if channel_key.startswith("physical_channel_"):
                for perm_key, perm_value in channel_data.items():
                    if perm_key != "original_noise" and isinstance(perm_value, (int, float)):
                        noise_estimates.append(perm_value)
                        permutation_labels.append(perm_key)

        if not noise_estimates:
            return {"error": "No valid permutation results"}

        # Calculate key metrics
        mean_noise = np.mean(noise_estimates)
        std_noise = np.std(noise_estimates)
        cv_noise = std_noise / mean_noise if mean_noise > 0 else 0

        # Theoretical self-noise (fixed value)
        theoretical_noise = self.fixed_noise

        # Calculate synchronization error related metrics
        max_sync_error = max(abs(d) for d in sync_delays)
        sync_error_rms = np.sqrt(np.mean([d**2 for d in sync_delays]))

        # Performance metrics
        metrics = {
            "mean_estimated_noise": mean_noise,
            "std_estimated_noise": std_noise,
            "coefficient_of_variation": cv_noise,
            "theoretical_noise": theoretical_noise,
            "estimation_accuracy": mean_noise / theoretical_noise if theoretical_noise > 0 else 0,
            "permutation_consistency": 1.0 - cv_noise if cv_noise < 1.0 else 0.0,  # Consistency metric, prevent negative values
            "interference_impact": signal_multiplier,  # External interference multiplier reflects interference intensity variation
            "total_permutations": len(noise_estimates),
            "permutation_labels": permutation_labels,
            # Synchronization error related metrics
            "sync_delays": sync_delays,
            "max_sync_error_samples": max_sync_error,
            "sync_error_rms_samples": sync_error_rms,
            "sync_impact_factor": max_sync_error / 20.0  # Ratio relative to maximum test range
        }

        return metrics

    def calculate_summary_statistics(self, gradient_results):
        """Calculate summary statistics for 2D gradient analysis (external interference × sync error)"""
        signal_multipliers = [r["signal_multiplier"] for r in gradient_results]

        # Filter valid performance metrics (exclude results containing error messages)
        valid_results = []
        for r in gradient_results:
            metrics = r["performance_metrics"]
            if "error" not in metrics and "estimation_accuracy" in metrics:
                valid_results.append(r)

        if not valid_results:
            # If no valid results, return default statistics
            return {
                "signal_range": {
                    "min_multiplier": min(signal_multipliers),
                    "max_multiplier": max(signal_multipliers),
                    "span": max(signal_multipliers) - min(signal_multipliers)
                },
                "sync_error_analysis": {
                    "max_sync_error": 0,
                    "sync_conditions_tested": 0,
                    "sync_impact_detected": False
                },
                "performance_trends": {
                    "accuracy_trend_slope": 0.0,
                    "consistency_trend_slope": 0.0,
                    "overall_robustness": 0.0
                },
                "optimal_conditions": {
                    "best_accuracy_condition": None,
                    "best_consistency_condition": None
                },
                "2d_analysis": {
                    "matrix_completeness": 0.0,
                    "interaction_effects_detected": False
                },
                "valid_measurements": 0,
                "total_measurements": len(gradient_results)
            }

        # Extract metrics from valid results
        valid_signal_multipliers = [r["signal_multiplier"] for r in valid_results]
        accuracies = [r["performance_metrics"]["estimation_accuracy"] for r in valid_results]
        consistencies = [r["performance_metrics"]["permutation_consistency"] for r in valid_results]
        sync_errors = [r["performance_metrics"]["max_sync_error_samples"] for r in valid_results]

        # Find optimal conditions
        # For accuracy, best value is closest to 1.0 (i.e., measured value closest to theoretical value)
        best_accuracy_idx = np.argmin(np.abs(np.array(accuracies) - 1.0)) if accuracies else 0
        # For consistency, best value is closest to 1.0 (i.e., coefficient of variation smallest)
        best_consistency_idx = np.argmax(consistencies) if consistencies else 0

        # Calculate performance trend with external interference variation
        if len(valid_results) > 1:
            # Linear fit to calculate trend
            accuracy_trend = np.polyfit(valid_signal_multipliers, accuracies, 1)[0]  # Slope
            consistency_trend = np.polyfit(valid_signal_multipliers, consistencies, 1)[0]
        else:
            accuracy_trend = 0.0
            consistency_trend = 0.0

        # Analyze synchronization error impact
        unique_sync_conditions = len(set(r.get("sync_condition", "unknown") for r in valid_results))
        max_sync_error = max(sync_errors) if sync_errors else 0

        # Detect interaction effects: compare performance differences under different sync conditions
        interaction_detected = False
        if unique_sync_conditions > 1:
            # Grouped analysis: performance differences under different sync conditions at same interference level
            signal_groups = {}
            for r in valid_results:
                signal_mult = r["signal_multiplier"]
                if signal_mult not in signal_groups:
                    signal_groups[signal_mult] = []
                signal_groups[signal_mult].append(r["performance_metrics"]["estimation_accuracy"])

            # Check for significant within-group variation (simple indicator of interaction effects)
            for signal_mult, accuracies_group in signal_groups.items():
                if len(accuracies_group) > 1 and np.std(accuracies_group) > 0.01:  # 1% threshold
                    interaction_detected = True
                    break
        
        return {
            "signal_range": {
                "min_multiplier": min(signal_multipliers),
                "max_multiplier": max(signal_multipliers),
                "span": max(signal_multipliers) - min(signal_multipliers)
            },
            "sync_error_analysis": {
                "max_sync_error_samples": max_sync_error,
                "sync_conditions_tested": unique_sync_conditions,
                "sync_impact_detected": max_sync_error > 0,
                "sync_error_range": [min(sync_errors), max(sync_errors)] if sync_errors else [0, 0]
            },
            "performance_trends": {
                "accuracy_trend_slope": accuracy_trend,
                "consistency_trend_slope": consistency_trend,
                "overall_robustness": np.mean(consistencies) if consistencies else 0.0
            },
            "optimal_conditions": {
                "best_accuracy_condition": {
                    "signal_multiplier": valid_results[best_accuracy_idx]["signal_multiplier"],
                    "sync_condition": valid_results[best_accuracy_idx].get("sync_condition", "unknown"),
                    "accuracy_value": accuracies[best_accuracy_idx]
                } if accuracies else None,
                "best_consistency_condition": {
                    "signal_multiplier": valid_results[best_consistency_idx]["signal_multiplier"],
                    "sync_condition": valid_results[best_consistency_idx].get("sync_condition", "unknown"),
                    "consistency_value": consistencies[best_consistency_idx]
                } if consistencies else None
            },
            "2d_analysis": {
                "matrix_completeness": len(valid_results) / len(gradient_results),
                "interaction_effects_detected": interaction_detected,
                "total_condition_combinations": len(gradient_results),
                "valid_combinations": len(valid_results)
            },
            "valid_measurements": len(valid_results),
            "total_measurements": len(gradient_results)
        }


def run_noise_gradient_analysis(exam_name, config):
    """Entry function for running external interference gradient analysis"""
    config_file = os.path.join("exams", exam_name, "config.json")

    analyzer = NoiseGradientAnalysis(config_file)
    output_dir, results = analyzer.run_analysis()

    return output_dir, results