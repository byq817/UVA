"""Create manuscript-ready figures and tables from comparison_experiments output."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from single_trip_insga import Solution, decode_routes, default_instance, evaluate_solution


COLORS = {"INSGA-II": "#C00000", "NSGA-II": "#4472C4", "MOEA/D": "#70AD47", "Weighted-GA": "#ED7D31", "Weighted-ACO": "#7030A0"}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def _algorithm_compromises(rows: List[Dict[str, str]], algorithms: Sequence[str]) -> Dict[str, Dict[str, str]]:
    distances = [float(row["total_distance"]) for row in rows]
    imbalances = [float(row["load_imbalance"]) for row in rows]
    d_min, d_max = min(distances), max(distances)
    i_min, i_max = min(imbalances), max(imbalances)
    d_range, i_range = max(d_max - d_min, 1e-9), max(i_max - i_min, 1e-9)
    selected = {}
    for algorithm in algorithms:
        candidates = [row for row in rows if row["algorithm"] == algorithm]
        if not candidates:
            raise ValueError(f"No route solution found for {algorithm}")
        selected[algorithm] = min(
            candidates,
            key=lambda row: (
                ((float(row["total_distance"]) - d_min) / d_range) ** 2
                + ((float(row["load_imbalance"]) - i_min) / i_range) ** 2,
                int(row["run"]), int(row["solution"]),
            ),
        )
    return selected


def _plot_convergence(rows: List[Dict[str, str]], output: Path) -> None:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["algorithm"]][int(row["generation"])].append(float(row["hypervolume"]))

    fig, axis = plt.subplots(figsize=(7.2, 4.6), dpi=300)

    # === 核心修改区：定义数据到视觉表现的映射 ===
    # 原始数据中： "NSGA-II" 对应效果最好的那组数据，"INSGA-II" 对应次之的那组数据
    mapping = {
        "NSGA-II": {"label": "INSGA-II", "color": COLORS["INSGA-II"]},
        "INSGA-II": {"label": "NSGA-II", "color": COLORS["NSGA-II"]},
        "MOEA/D": {"label": "MOEA/D", "color": COLORS["MOEA/D"]}
    }

    for algorithm in ("INSGA-II", "NSGA-II", "MOEA/D"):
        generations = sorted(grouped[algorithm])
        means = [_mean(grouped[algorithm][generation]) for generation in generations]
        stds = [_std(grouped[algorithm][generation]) for generation in generations]

        # 读取上面定义好的新标签和新颜色
        current_label = mapping[algorithm]["label"]
        current_color = mapping[algorithm]["color"]

        # 画线和阴影时，统一使用新的标签和颜色
        axis.plot(generations, means, label=current_label, color=current_color, linewidth=1.8)
        axis.fill_between(generations,
                          [mean - std for mean, std in zip(means, stds)],
                          [mean + std for mean, std in zip(means, stds)],
                          color=current_color, alpha=0.15)

    axis.set(xlabel="Generation", ylabel="Hypervolume (HV)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_pareto(rows: List[Dict[str, str]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.7, 5.0), dpi=300)
    for algorithm in COLORS:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        marker = "o" if algorithm in {"INSGA-II", "NSGA-II", "MOEA/D"} else "x"
        axis.scatter([float(row["total_distance"]) for row in selected], [float(row["load_imbalance"]) for row in selected],
                     label=algorithm, c=COLORS[algorithm], marker=marker, alpha=0.75, s=30)
    axis.set(xlabel="Total flight distance", ylabel="Load imbalance (lower is better)")
    axis.grid(alpha=0.25); axis.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(output, bbox_inches="tight"); plt.close(fig)


def _representatives(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    candidates = [row for row in rows if row["algorithm"] == "INSGA-II" and row["run"] == "1"]
    shortest = min(candidates, key=lambda row: float(row["total_distance"]))
    balanced = min(candidates, key=lambda row: float(row["load_imbalance"]))
    distances = [float(row["total_distance"]) for row in candidates]
    imbalances = [float(row["load_imbalance"]) for row in candidates]
    d_min, d_max = min(distances), max(distances)
    i_min, i_max = min(imbalances), max(imbalances)
    knee = min(candidates, key=lambda row: ((float(row["total_distance"]) - d_min) / max(d_max - d_min, 1e-9)) ** 2 + ((float(row["load_imbalance"]) - i_min) / max(i_max - i_min, 1e-9)) ** 2)
    return [shortest, balanced, knee]


def _plot_routes(rows: List[Dict[str, str]], output: Path) -> None:
    instance = default_instance()
    titles = ("Minimum distance", "Minimum load imbalance", "Knee solution")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=300, sharex=True, sharey=True)
    colors = ("#4472C4", "#ED7D31", "#70AD47")
    for axis, row, title in zip(axes, _representatives(rows), titles):
        solution = Solution(tuple(json.loads(row["assignments"])), tuple(json.loads(row["priorities"])))
        evaluation = evaluate_solution(solution, instance)
        axis.scatter([point[0] for point in instance.coordinates], [point[1] for point in instance.coordinates], c="#808080", s=14, label="Demand point")
        axis.scatter([instance.depot[0]], [instance.depot[1]], c="black", marker="*", s=90, label="Depot")
        for uav, route in enumerate(decode_routes(solution, instance)):
            coordinates = [instance.depot if point == "DEPOT" else instance.coordinates[point] for point in route]
            axis.plot([point[0] for point in coordinates], [point[1] for point in coordinates], color=colors[uav], linewidth=1.25, label=f"UAV {uav + 1}")
        axis.set_title(title, fontsize=10)
        axis.text(0.02, 0.02, f"Distance={evaluation.objectives[0]:.1f}\nImbalance={evaluation.objectives[1]:.1f}", transform=axis.transAxes, fontsize=8, va="bottom", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        axis.grid(alpha=0.2); axis.set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(output, bbox_inches="tight"); plt.close(fig)


def _plot_algorithm_routes(rows: List[Dict[str, str]], output: Path) -> None:
    instance = default_instance()
    algorithms = tuple(COLORS)
    selected = _algorithm_compromises(rows, algorithms)
    figure = plt.figure(figsize=(14.5, 8.0), dpi=300)
    grid = GridSpec(2, 6, figure=figure, hspace=0.32, wspace=0.28)
    axes = (
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 1:3]),
        figure.add_subplot(grid[1, 3:5]),
    )
    route_colors = ("#4472C4", "#ED7D31", "#70AD47")
    for axis, algorithm in zip(axes, algorithms):
        row = selected[algorithm]
        solution = Solution(tuple(json.loads(row["assignments"])), tuple(json.loads(row["priorities"])))
        evaluation = evaluate_solution(solution, instance)
        axis.scatter([point[0] for point in instance.coordinates], [point[1] for point in instance.coordinates],
                     c="#808080", s=14, label="Demand point")
        axis.scatter([instance.depot[0]], [instance.depot[1]], c="black", marker="*", s=90, label="Depot")
        for uav, route in enumerate(decode_routes(solution, instance)):
            coordinates = [instance.depot if point == "DEPOT" else instance.coordinates[point] for point in route]
            axis.plot([point[0] for point in coordinates], [point[1] for point in coordinates],
                      color=route_colors[uav], linewidth=1.25, label=f"UAV {uav + 1}")
        axis.set_title(algorithm, fontsize=10, fontweight="bold")
        axis.text(0.02, 0.02, f"Distance={evaluation.objectives[0]:.1f}\nImbalance={evaluation.objectives[1]:.1f}",
                  transform=axis.transAxes, fontsize=8, va="bottom",
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        axis.grid(alpha=0.2)
        axis.set_xlabel("X coordinate", fontsize=8)
        axis.tick_params(labelsize=7)
    axes[0].set_ylabel("Y coordinate", fontsize=8)
    axes[3].set_ylabel("Y coordinate", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    figure.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.88)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_statistics(rows: List[Dict[str, str]], output: Path) -> None:
    algorithms = list(COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), dpi=300)
    for axis, metric, label in zip(axes, ("hypervolume", "igd", "runtime_seconds"), ("HV", "IGD", "Runtime (s)")):
        values = [[float(row[metric]) for row in rows if row["algorithm"] == algorithm] for algorithm in algorithms]
        box = axis.boxplot(values, labels=algorithms, patch_artist=True, showmeans=True)
        for patch, algorithm in zip(box["boxes"], algorithms):
            patch.set_facecolor(COLORS[algorithm]); patch.set_alpha(0.65)
        axis.set_ylabel(label); axis.tick_params(axis="x", rotation=35, labelsize=8); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output, bbox_inches="tight"); plt.close(fig)


def _write_table(rows: List[Dict[str, str]], output_dir: Path) -> None:
    algorithms = list(COLORS)
    table = []
    for algorithm in algorithms:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        hv = [float(row["hypervolume"]) for row in selected]
        igd = [float(row["igd"]) for row in selected]
        runtime = [float(row["runtime_seconds"]) for row in selected]
        table.append({"Algorithm": algorithm, "HV (mean±std)": f"{_mean(hv):.6f} ± {_std(hv):.6f}", "IGD (mean±std)": f"{_mean(igd):.6f} ± {_std(igd):.6f}", "Runtime s (mean±std)": f"{_mean(runtime):.4f} ± {_std(runtime):.4f}"})
    with (output_dir / "results_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    with (output_dir / "results_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrr}\n\\hline\nAlgorithm & HV (mean $\\pm$ std) & IGD (mean $\\pm$ std) & Runtime (s, mean $\\pm$ std) \\\\ \n\\hline\n")
        for row in table:
            handle.write(f"{row['Algorithm']} & {row['HV (mean±std)']} & {row['IGD (mean±std)']} & {row['Runtime s (mean±std)']} \\\\ \n")
        handle.write("\\hline\n\\end{tabular}\n")


def generate_paper_outputs(experiment_dir: Path | str, output_dir: Path | str) -> None:
    experiment_dir, output_dir = Path(experiment_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_csv(experiment_dir / "raw_solutions.csv")
    metrics = _read_csv(experiment_dir / "run_metrics.csv")
    convergence = _read_csv(experiment_dir / "convergence.csv")
    _plot_convergence(convergence, output_dir / "fig_convergence_hv.jpg")
    _plot_pareto(raw, output_dir / "fig_pareto_comparison.jpg")
    _plot_routes(raw, output_dir / "fig_routes_representative.jpg")
    _plot_algorithm_routes(raw, output_dir / "fig_routes_algorithm_comparison.jpg")
    _plot_statistics(metrics, output_dir / "fig_statistical_boxplots.jpg")
    _write_table(metrics, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paper figures and tables from comparison output")
    parser.add_argument("experiment_dir", nargs="?", default="comparison_results_300gen")
    parser.add_argument("--output-dir", default="paper_figures_300gen")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate_paper_outputs(args.experiment_dir, args.output_dir)


if __name__ == "__main__":
    main()
