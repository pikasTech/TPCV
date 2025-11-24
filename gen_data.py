#!/usr/bin/env python3
"""
Experimental Data Generation Entry Point - Code Release Version
================================================================
Generate simulation data for all experiments (in code_release/exams directory)

Experiment List:
1. baseline - Baseline experiment
2. E01-algorithm_verification - Algorithm verification
3. E02-sync_level1_v2 ~ level4_v2 - Synchronization error level tests (4 experiments)
4. E02-sync_sensitivity - Synchronization sensitivity overview
5. E04-combined_effects - Combined effects analysis (10×10 factorial design)

Usage:
    python gen_data.py                  # Generate all experiment data
    python gen_data.py --exam E01-algorithm_verification  # Generate specific experiment
    python gen_data.py --list           # List all available experiments
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / "code"))

from experiment_runner import run_single_experiment


class DataGenerator:
    """Data generation controller"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.exams_dir = self.project_root / "exams"

        # Experiment list (grouped by type)
        self.single_experiments = [
            "baseline",
            "E01-algorithm_verification",
            "E02-sync_level1_v2",
            "E02-sync_level2_v2",
            "E02-sync_level3_v2",
            "E02-sync_level4_v2",
            "E02-sync_sensitivity",
        ]

        self.gradient_experiments = [
            "E04-combined_effects",
        ]

    def run_experiment(self, exam_name):
        """Run specified experiment (auto-detect type)"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {exam_name}")
        print(f"{'='*60}\n")

        # Switch to project root directory (ensure correct paths)
        original_dir = os.getcwd()
        os.chdir(self.project_root)

        try:
            # Use experiment_runner for unified scheduling
            output_file = run_single_experiment(exam_name)
            print(f"\n[OK] Experiment completed!")
            print(f"Results file: {output_file}")
            return output_file
        finally:
            os.chdir(original_dir)

    def generate_all(self):
        """Generate all experiment data"""
        print("\n" + "="*60)
        print("Starting generation of all experiment data")
        print("="*60)

        results = {}

        # Run all experiments
        all_experiments = self.single_experiments + self.gradient_experiments
        for exam_name in all_experiments:
            try:
                output = self.run_experiment(exam_name)
                results[exam_name] = {"status": "success", "output": output}
            except Exception as e:
                print(f"\nExperiment {exam_name} failed: {e}")
                import traceback
                traceback.print_exc()
                results[exam_name] = {"status": "failed", "error": str(e)}

        # Print summary
        self.print_summary(results)
        return results

    def generate_single(self, exam_name):
        """Generate data for specified experiment"""
        if exam_name in self.single_experiments + self.gradient_experiments:
            return self.run_experiment(exam_name)
        else:
            raise ValueError(f"Unknown experiment: {exam_name}")

    def list_experiments(self):
        """List all available experiments"""
        print("\nAvailable experiments:")
        print("\nSingle experiment type (single_three_channel_experiment):")
        for exam in self.single_experiments:
            print(f"  - {exam}")
        print("\nGradient analysis experiment type (noise_gradient_analysis):")
        for exam in self.gradient_experiments:
            print(f"  - {exam}")
        print(f"\nTotal: {len(self.single_experiments) + len(self.gradient_experiments)} experiments\n")

    def print_summary(self, results):
        """Print generation result summary"""
        print("\n" + "="*60)
        print("Data Generation Summary")
        print("="*60)

        success_count = sum(1 for r in results.values() if r["status"] == "success")
        failed_count = sum(1 for r in results.values() if r["status"] == "failed")

        print(f"\n[OK] Success: {success_count} experiments")
        print(f"[FAIL] Failed: {failed_count} experiments")

        if failed_count > 0:
            print("\nFailed experiments:")
            for exam, result in results.items():
                if result["status"] == "failed":
                    print(f"  - {exam}: {result['error']}")

        print("\n" + "="*60)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Experimental data generation tool")
    parser.add_argument("--exam", type=str, help="Generate specific experiment data (e.g., E01-algorithm_verification)")
    parser.add_argument("--list", action="store_true", help="List all available experiments")

    args = parser.parse_args()

    generator = DataGenerator()

    if args.list:
        generator.list_experiments()
    elif args.exam:
        generator.generate_single(args.exam)
    else:
        generator.generate_all()


if __name__ == "__main__":
    main()
