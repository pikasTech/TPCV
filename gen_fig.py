#!/usr/bin/env python3
"""
论文图表生成入口 - Code Release版本
=====================================

用途：一键生成论文所需的所有图表

生成的图表（对应main.tex引用）：
1. E01_figure0_time_domain.png - 时域波形图
2. E01_figure1b_frequency_processing.png - 频域处理图
3. figure3_sync_error_time_domain.png - 同步误差时域图
4. figure4_tpcv_sync_error.png - TPCV与同步误差图
5. figure7_tpcv_interaction_heatmap.png - TPCV交互热图
6. figure8_main_effects.png - 主效应图
7. figure11c_error_vs_tpcv_scatter.png - 误差 vs TPCV散点图
8. figure11c_error_vs_tpcv_scatter_with_cave.png - 误差 vs TPCV散点图（含山洞数据）

使用方法：
    python gen_fig.py

输出目录：output/{timestamp}/
"""

import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


class FigureGenerator:
    """论文图表生成器"""

    def __init__(self):
        """初始化"""
        self.project_root = Path(__file__).parent
        self.code_dir = self.project_root / "code"
        self.exams_dir = self.project_root / "exams"

        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.project_root / "output" / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"输出目录: {self.output_dir}")

    def run_python_code(self, code, description="", cwd=None):
        """运行Python代码片段"""
        print(f"\n{'='*60}")
        print(f"执行: {description}")
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
                print(f"警告: 返回非零退出码 {result.returncode}")
                if result.stderr:
                    print(f"错误信息:\n{result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            print(f"错误: 执行超时（>300秒）")
            return False
        except Exception as e:
            print(f"错误: {e}")
            return False

    def copy_and_rename(self, source, dest_name, description=""):
        """复制并重命名图片到输出目录"""
        if isinstance(source, str):
            source = Path(source)

        if not source.exists():
            print(f"[WARN] 源文件不存在: {source}")
            return False

        dest = self.output_dir / dest_name
        shutil.copy2(source, dest)
        print(f"[OK] 已复制: {dest_name} - {description}")

        # 同时复制JSON文件
        json_source = source.with_suffix('.json')
        if json_source.exists():
            json_dest = self.output_dir / Path(dest_name).with_suffix('.json').name
            shutil.copy2(json_source, json_dest)
            print(f"[OK] 已复制JSON: {json_dest.name}")

        return True

    def generate_e01_figures(self):
        """生成E01实验图表"""
        print("\n" + "="*60)
        print("步骤1: 生成E01算法验证图表")
        print("="*60)

        code_dir_str = str(self.code_dir).replace('\\', '/')
        project_root_str = str(self.project_root).replace('\\', '/')
        code = f"""
import sys
sys.path.insert(0, '{code_dir_str}')
from e01_specialized_visualizer_v2 import E01SpecializedVisualizerV2

viz = E01SpecializedVisualizerV2(exam_name="E01-algorithm_verification", base_dir='{project_root_str}')
files = viz.generate_all_figures()
print(f"\\n[SUCCESS] E01图表已生成: {{files}}")
"""
        success = self.run_python_code(code, "E01算法验证可视化", cwd=self.project_root)

        if success:
            exam_dir = self.exams_dir / "E01-algorithm_verification"
            output_dir = exam_dir / "output"
            timestamps = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
            if timestamps:
                latest = timestamps[-1]
                figures_dir = output_dir / latest / "figures"

                src0 = figures_dir / "figure0_time_domain_waveforms.png"
                if src0.exists():
                    self.copy_and_rename(src0, "E01_figure0_time_domain.png", "时域波形图")

    def generate_e01_figure1b(self):
        """生成E01_figure1b频域处理图"""
        print("\n" + "="*60)
        print("步骤2: 生成E01_figure1b频域处理图")
        print("="*60)

        code_dir_str = str(self.code_dir).replace('\\', '/')
        config_path_str = str(self.exams_dir / "baseline" / "config.json").replace('\\', '/')
        output_dir_str = str(self.output_dir).replace('\\', '/')
        code = f"""
import sys
sys.path.insert(0, '{code_dir_str}')

# 修改plot_figure1b中的配置路径
import plot_figure1b
from pathlib import Path

# 覆盖配置加载函数
original_load = plot_figure1b.load_baseline_config
def patched_load():
    config_path = Path('{config_path_str}')
    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
plot_figure1b.load_baseline_config = patched_load

# 生成图表
output_dir = Path('{output_dir_str}')
output_basename = "E01_figure1b_frequency_processing"
# 使用sys.argv传递参数
sys.argv = ['plot_figure1b.py', '--output-dir', str(output_dir), '--output-basename', output_basename]
plot_figure1b.main()
print("[SUCCESS] figure1b已生成")
"""
        self.run_python_code(code, "三通道频域处理示意图", cwd=self.project_root)

    def generate_e02_figures(self):
        """生成E02同步误差图表"""
        print("\n" + "="*60)
        print("步骤3: 生成E02同步误差图表")
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
    print(f"\\n[SUCCESS] E02图表已生成")
except Exception as e:
    print(f"[ERROR] E02图表生成失败: {{e}}")
    import traceback
    traceback.print_exc()
"""
        success = self.run_python_code(code, "E02同步误差可视化", cwd=self.project_root)

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
                        self.copy_and_rename(src_fig2, "figure3_sync_error_time_domain.png", "时域同步误差对比图")

                    src_fig5 = figures_dir / "figure5_tpcv_sync_error.png"
                    if src_fig5.exists():
                        self.copy_and_rename(src_fig5, "figure4_tpcv_sync_error.png", "TPCV与同步误差关系图")

    def generate_e04_figures(self):
        """生成E04综合效应图表"""
        print("\n" + "="*60)
        print("步骤4: 生成E04综合效应图表")
        print("="*60)

        exam_dir = self.exams_dir / "E04-combined_effects"
        output_dir = exam_dir / "output"

        if not output_dir.exists():
            print("[ERROR] E04实验未运行，无output目录")
            return

        timestamps = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
        if not timestamps:
            print("[ERROR] E04实验无输出数据")
            return

        latest = timestamps[-1]
        results_json = output_dir / latest / "results.json"

        if not results_json.exists():
            print(f"[ERROR] results.json不存在: {results_json}")
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
        success = self.run_python_code(code, "E04综合效应可视化", cwd=self.project_root)

        if success:
            figures_dir = output_dir / latest / "figures"
            figure_mapping = {
                "figure11_tpcv_interaction_heatmap.png": ("figure7_tpcv_interaction_heatmap.png", "TPCV交互效应热图"),
                "figure12_main_effects.png": ("figure8_main_effects.png", "主效应分析图"),
                "figure11c_error_vs_tpcv_scatter.png": ("figure11c_error_vs_tpcv_scatter.png", "相对误差 vs TPCV散点图"),
                "figure11c_error_vs_tpcv_scatter_with_cave.png": ("figure11c_error_vs_tpcv_scatter_with_cave.png", "相对误差 vs TPCV散点图（含山洞测试）")
            }

            for src_name, (dest_name, description) in figure_mapping.items():
                src_file = figures_dir / src_name
                if src_file.exists():
                    self.copy_and_rename(src_file, dest_name, description)
                else:
                    print(f"[WARN] {src_name} 不存在")

    def run(self):
        """执行完整的图表生成流程"""
        print("\n" + "="*70)
        print(" "*15 + "论文图表生成器 (Code Release)")
        print("="*70)
        print(f"项目目录: {self.project_root}")
        print(f"输出目录: {self.output_dir}")

        try:
            self.generate_e01_figures()
            self.generate_e01_figure1b()
            self.generate_e02_figures()
            self.generate_e04_figures()

            generated_files = list(self.output_dir.glob("*.png"))

            print("\n" + "="*70)
            print(" "*25 + "生成完成！")
            print("="*70)
            print(f"共生成 {len(generated_files)} 个图表文件")
            print(f"输出目录: {self.output_dir}")
            print("\n生成的图表:")
            for i, fig in enumerate(sorted(generated_files), 1):
                print(f"  {i:2d}. {fig.name}")
            print("="*70)

        except Exception as e:
            print(f"\n[ERROR] 生成过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    generator = FigureGenerator()
    generator.run()


if __name__ == "__main__":
    main()
