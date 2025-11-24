#!/usr/bin/env python3
"""
Paper Figure Generation Entry Point - Code Release Version
===========================================================

Purpose: One-click generation of all figures for the paper

Generated figures (referenced in main.tex):
1. E01_figure0_time_domain.png - Time domain waveform plot
2. E01_figure1b_frequency_processing.png - Frequency domain processing plot
3. figure3_sync_error_time_domain.png - Synchronization error time domain plot
4. figure4_tpcv_sync_error.png - TPCV vs synchronization error plot
5. figure7_tpcv_interaction_heatmap.png - TPCV interaction heatmap
6. figure8_main_effects.png - Main effects plot
7. figure11c_error_vs_tpcv_scatter.png - Error vs TPCV scatter plot
8. figure11c_error_vs_tpcv_scatter_with_cave.png - Error vs TPCV scatter plot (with cave data)

Usage:
    python gen_fig.py

Output directory: output/{timestamp}/
"""

import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


class FigureGenerator:
    """Paper figure generator"""

    def __init__(self):
        """Initialize generator"""
        self.project_root = Path(__file__).parent
        self.code_dir = self.project_root / "code"
        self.exams_dir = self.project_root / "exams"

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.project_root / "output" / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {self.output_dir}")

    def run_python_code(self, code, description="", cwd=None):
        """Execute Python code snippet"""
        print(f"\n{'='*60}")
        print(f"Executing: {description}")
        print(f"{'='*60}")

        if cwd is None:
            cwd = self.project_root

        cmd = [sys.executable, "-c", code]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.stdout:
                print(result.stdout)

            if result.returncode != 0:
                print(f"Warning: Non-zero exit code {result.returncode}")
                if result.stderr:
                    print(f"Error message:\n{result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            print(f"Error: Execution timeout (>300 seconds)")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def copy_and_rename(self, source, dest_name, description=""):
        """Copy and rename figure to output directory"""
        if isinstance(source, str):
            source = Path(source)

        if not source.exists():
            print(f"[WARN] Source file does not exist: {source}")
            return False

        dest = self.output_dir / dest_name
        shutil.copy2(source, dest)
        print(f"[OK] Copied: {dest_name} - {description}")

        # Also copy JSON file if exists
        json_source = source.with_suffix('.json')
        if json_source.exists():
            json_dest = self.output_dir / Path(dest_name).with_suffix('.json').name
            shutil.copy2(json_source, json_dest)
            print(f"[OK] Copied JSON: {json_dest.name}")

        return True

    def generate_e01_figures(self):
        """Generate E01 experiment figures"""
        print("\n" + "="*60)
        print("Step 1: Generate E01 algorithm verification figures")
        print("="*60)

        code_dir_str = str(self.code_dir).replace('\\', '/')
        project_root_str = str(self.project_root).replace('\\', '/')
        code = f"""
import sys
sys.path.insert(0, '{code_dir_str}')
from e01_specialized_visualizer_v2 import E01SpecializedVisualizerV2

viz = E01SpecializedVisualizerV2(exam_name="E01-algorithm_verification", base_dir='{project_root_str}')
files = viz.generate_all_figures()
print(f"\\n[SUCCESS] E01 figures generated: {{files}}")
"""
        success = self.run_python_code(code, "E01 algorithm verification visualization", cwd=self.project_root)

        if success:
            exam_dir = self.exams_dir / "E01-algorithm_verification"
            output_dir = exam_dir / "output"
            timestamps = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
            if timestamps:
                latest = timestamps[-1]
                figures_dir = output_dir / latest / "figures"

                src0 = figures_dir / "figure0_time_domain_waveforms.png"
                if src0.exists():
                    self.copy_and_rename(src0, "E01_figure0_time_domain.png", "Time domain waveform plot")

    def generate_e01_figure1b(self):
        """Generate E01 figure1b frequency domain processing plot"""
        print("\n" + "="*60)
        print("Step 2: Generate E01 figure1b frequency domain processing plot")
        print("="*60)

        code_dir_str = str(self.code_dir).replace('\\', '/')
        config_path_str = str(self.exams_dir / "baseline" / "config.json").replace('\\', '/')
        output_dir_str = str(self.output_dir).replace('\\', '/')
        code = f"""
import sys
sys.path.insert(0, '{code_dir_str}')

# Modify configuration path in plot_figure1b
import plot_figure1b
from pathlib import Path

# Override configuration loading function
original_load = plot_figure1b.load_baseline_config
def patched_load():
    config_path = Path('{config_path_str}')
    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
plot_figure1b.load_baseline_config = patched_load

# Generate figure
output_dir = Path('{output_dir_str}')
output_basename = "E01_figure1b_frequency_processing"
# Pass parameters via sys.argv
sys.argv = ['plot_figure1b.py', '--output-dir', str(output_dir), '--output-basename', output_basename]
plot_figure1b.main()
print("[SUCCESS] figure1b generated")
"""
        self.run_python_code(code, "Three-channel frequency domain processing diagram", cwd=self.project_root)

    def generate_e02_figures(self):
        """Generate E02 synchronization error figures"""
        print("\n" + "="*60)
        print("Step 3: Generate E02 synchronization error figures")
        print("="*60)

        code_dir_str = str(self.code_dir).replace('\\', '/')
        project_root_str = str(self.project_root).replace('\\', '/')
        code = f"""
import sys
sys.path.insert(0, '{code_dir_str}')
from e02_specialized_visualizer import E02SpecializedVisualizer

viz = E02SpecializedVisualizer(base_dir='{project_root_str}')

try:
    results = viz.generate_all_figures()
    print(f"\\n[SUCCESS] E02 figures generated")
except Exception as e:
    print(f"[ERROR] E02 figure generation failed: {{e}}")
    import traceback
    traceback.print_exc()
"""
        success = self.run_python_code(code, "E02 synchronization error visualization", cwd=self.project_root)

        if success:
            exam_dir = self.exams_dir / "E02-sync_sensitivity"
            output_dir = exam_dir / "output"
            if output_dir.exists():
                timestamps = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
                if timestamps:
                    latest = timestamps[-1]
                    figures_dir = output_dir / latest / "figures"

                    src_fig2 = figures_dir / "figure2_sync_error_time_domain.png"
                    if src_fig2.exists():
                        self.copy_and_rename(src_fig2, "figure3_sync_error_time_domain.png", "Time domain sync error comparison plot")

                    src_fig5 = figures_dir / "figure5_tpcv_sync_error.png"
                    if src_fig5.exists():
                        self.copy_and_rename(src_fig5, "figure4_tpcv_sync_error.png", "TPCV vs synchronization error plot")

    def generate_e04_figures(self):
        """Generate E04 combined effects figures"""
        print("\n" + "="*60)
        print("Step 4: Generate E04 combined effects figures")
        print("="*60)

        exam_dir = self.exams_dir / "E04-combined_effects"
        output_dir = exam_dir / "output"

        if not output_dir.exists():
            print("[ERROR] E04 experiment not run, no output directory")
            return

        timestamps = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
        if not timestamps:
            print("[ERROR] E04 experiment has no output data")
            return

        latest = timestamps[-1]
        results_json = output_dir / latest / "results.json"

        if not results_json.exists():
            print(f"[ERROR] results.json does not exist: {results_json}")
            return

        code_dir_str = str(self.code_dir).replace('\\', '/')
        results_json_str = str(results_json).replace('\\', '/')
        code = f"""
import sys
sys.path.insert(0, '{code_dir_str}')
from e04_specialized_visualizer import (
    load_e04_data, extract_experimental_parameters,
    generate_f05_interaction_heatmap, generate_f05c_error_vs_tpcv_scatter,
    generate_f05c_error_vs_tpcv_scatter_with_cave, generate_f06_main_effects
)
from pathlib import Path

results_path = Path('{results_json_str}')
output_dir = results_path.parent / 'figures'
output_dir.mkdir(exist_ok=True)

print(f"Loading data from: {{results_path}}")
data = load_e04_data(results_path)
params = extract_experimental_parameters(results_path)

print("Generating figures...")
generate_f05_interaction_heatmap(data, output_dir / 'figure11_tpcv_interaction_heatmap.png', params)
generate_f05c_error_vs_tpcv_scatter(data, output_dir / 'figure11c_error_vs_tpcv_scatter.png', params)
generate_f05c_error_vs_tpcv_scatter_with_cave(data, output_dir / 'figure11c_error_vs_tpcv_scatter_with_cave.png', params)
generate_f06_main_effects(data, output_dir / 'figure12_main_effects.png', params)

print("=== All E04 figures generated ===")
"""
        success = self.run_python_code(code, "E04 combined effects visualization", cwd=self.project_root)

        if success:
            figures_dir = output_dir / latest / "figures"
            figure_mapping = {
                "figure11_tpcv_interaction_heatmap.png": ("figure7_tpcv_interaction_heatmap.png", "TPCV interaction effect heatmap"),
                "figure12_main_effects.png": ("figure8_main_effects.png", "Main effects analysis plot"),
                "figure11c_error_vs_tpcv_scatter.png": ("figure11c_error_vs_tpcv_scatter.png", "Relative error vs TPCV scatter plot"),
                "figure11c_error_vs_tpcv_scatter_with_cave.png": ("figure11c_error_vs_tpcv_scatter_with_cave.png", "Relative error vs TPCV scatter plot (with cave test)")
            }

            for src_name, (dest_name, description) in figure_mapping.items():
                src_file = figures_dir / src_name
                if src_file.exists():
                    self.copy_and_rename(src_file, dest_name, description)
                else:
                    print(f"[WARN] {src_name} does not exist")

    def run(self):
        """Execute complete figure generation workflow"""
        print("\n" + "="*70)
        print(" "*15 + "Paper Figure Generator (Code Release)")
        print("="*70)
        print(f"Project directory: {self.project_root}")
        print(f"Output directory: {self.output_dir}")

        try:
            self.generate_e01_figures()
            self.generate_e01_figure1b()
            self.generate_e02_figures()
            self.generate_e04_figures()

            generated_files = list(self.output_dir.glob("*.png"))

            print("\n" + "="*70)
            print(" "*25 + "Generation Complete!")
            print("="*70)
            print(f"Generated {len(generated_files)} figure files")
            print(f"Output directory: {self.output_dir}")
            print("\nGenerated figures:")
            for i, fig in enumerate(sorted(generated_files), 1):
                print(f"  {i:2d}. {fig.name}")
            print("="*70)

        except Exception as e:
            print(f"\n[ERROR] Error during generation: {e}")
            import traceback
            traceback.print_exc()


def main():
    generator = FigureGenerator()
    generator.run()


if __name__ == "__main__":
    main()
