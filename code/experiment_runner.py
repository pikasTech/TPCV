#!/usr/bin/env python3
"""
Experiment Runner Module (Code Release Simplified Version)
Only retains data generation functionality, visualization and reporting removed
"""
import os
import json
from datetime import datetime
from single_experiment import SingleExperiment


def run_single_experiment(exam_name):
    """Run single three-channel permutation experiment or comparison experiment

    Args:
        exam_name: Experiment name, corresponding to subdirectory under exams/ (required)
    """
    if not exam_name:
        raise ValueError("exam_name parameter is required")

    # Check if this is a comparison experiment
    config_file = os.path.join("exams", exam_name, "config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Check if this is a noise gradient analysis experiment
        if config.get('experiment_type') == 'noise_gradient_analysis':
            from noise_gradient_analysis import run_noise_gradient_analysis

            print("Running dynamic noise strength gradient analysis")
            print("=" * 50)

            output_dir, analysis = run_noise_gradient_analysis(exam_name, config)

            if output_dir:
                output_file = os.path.join(output_dir, "results.json")
                print(f"\nNoise gradient analysis complete!")
                print(f"Results file: {output_file}")
                return output_file
            else:
                return None

    # Run regular experiment
    print("Running single three-channel permutation experiment")
    print("=" * 50)

    # Create experiment instance, pass in exam_name
    experiment = SingleExperiment(exam_name=exam_name)

    # Detect if E02 series experiment, needs to save waveform data for time-domain visualization
    save_waveform = exam_name.startswith('E02-')
    waveform_duration = 1.0  # Save 1 second of waveform data

    # Run experiment
    if save_waveform:
        print(f"Detected E02 series experiment, will save {waveform_duration}s waveform data for time-domain analysis")
        results = experiment.run_single_experiment(
            save_waveform=True,
            waveform_duration_seconds=waveform_duration
        )
    else:
        results = experiment.run_single_experiment()

    # Analyze symmetry
    symmetry_verified, groups = experiment.analyze_symmetry(results)

    # Save results
    output_file = experiment.save_results(results, symmetry_verified, groups)

    print(f"\nExperiment complete! Symmetry verification: {'Passed' if symmetry_verified else 'Failed'}")
    print(f"Results file: {output_file}")

    return output_file
