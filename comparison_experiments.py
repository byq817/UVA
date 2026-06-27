"""Fair baseline experiments for the single-trip UAV delivery instance.

Multi-objective algorithms return nondominated feasible sets.  GA and ACO are
scalarized baselines: each requested weight is a separate optimization run, and
their results are only merged for plotting/evaluation afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from single_trip_insga import (
    DEPOT,
    ProblemInstance,
    Solution,
    _crossover,
    _environmental_selection,
    _mutate,
    _random_solution,
    _rank_and_crowding,
    _tournament,
    decode_routes,
    default_instance,
    dominates,
    evaluate_solution,
    repair_solution,
    run_insga2,
)


DEFAULT_WEIGHTS = tuple(round(index / 10, 1) for index in range(1, 10))
OBJECTIVE_SCALES = (1000.0, 3000.0)


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _nondominated_unique(solutions: Iterable[Solution], instance: ProblemInstance) -> List[Solution]:
    unique = list(dict.fromkeys(solutions))
    if not unique:
        return []
    fronts, _, _, _ = _rank_and_crowding(unique, instance)
    return [unique[index] for index in fronts[0]]


def _scalar_fitness(solution: Solution, instance: ProblemInstance, weight: float) -> float:
    distance, imbalance = evaluate_solution(solution, instance).objectives
    return weight * distance / OBJECTIVE_SCALES[0] + (1.0 - weight) * imbalance / OBJECTIVE_SCALES[1]


def run_standard_nsga2(instance: ProblemInstance, population_size: int = 100,
                       generations: int = 100, seed: int = 42, record_history: bool = False):
    """Standard NSGA-II: no immune clones and no generation-dependent mutation."""
    rng = random.Random(seed)
    population = [_random_solution(instance, rng) for _ in range(population_size)]
    history = []
    for _ in range(generations):
        _, _, ranks, crowding = _rank_and_crowding(population, instance)
        offspring = []
        while len(offspring) < population_size:
            parent_a = _tournament(population, ranks, crowding, rng)
            parent_b = _tournament(population, ranks, crowding, rng)
            child = _crossover(parent_a, parent_b, instance, rng)
            offspring.append(_mutate(child, instance, 0.15, rng))
        population = _environmental_selection([*population, *offspring], population_size, instance)
        history.append(tuple(_nondominated_unique(population, instance)))
    front = _nondominated_unique(population, instance)
    return (front, tuple(history)) if record_history else front


def _moead_weights(count: int = 11) -> List[Tuple[float, float]]:
    return [(index / (count - 1), 1.0 - index / (count - 1)) for index in range(count)]


def moead_weight_count(population_size: int) -> int:
    """Match the common evaluation budget without using fewer than 11 subproblems."""
    return max(11, population_size)


def _tchebycheff(solution: Solution, instance: ProblemInstance, weight: Tuple[float, float],
                 ideal: Tuple[float, float]) -> float:
    objective = evaluate_solution(solution, instance).objectives
    return max(
        weight[index] * abs((objective[index] - ideal[index]) / OBJECTIVE_SCALES[index])
        for index in range(2)
    )


def run_moead(instance: ProblemInstance, population_size: int = 100,
              generations: int = 100, seed: int = 42, record_history: bool = False):
    """MOEA/D with 11 fixed two-objective decomposition weights and neighborhoods."""
    rng = random.Random(seed)
    weights = _moead_weights(moead_weight_count(population_size))
    population = [_random_solution(instance, rng) for _ in weights]
    neighborhoods = []
    for index, weight in enumerate(weights):
        ordered = sorted(range(len(weights)), key=lambda other: _euclidean(weight, weights[other]))
        neighborhoods.append(ordered[:3])
    values = [evaluate_solution(solution, instance).objectives for solution in population]
    ideal = (min(value[0] for value in values), min(value[1] for value in values))
    history = []
    for _ in range(generations):
        for index in range(len(weights)):
            neighbors = neighborhoods[index]
            parent_a = population[rng.choice(neighbors)]
            parent_b = population[rng.choice(neighbors)]
            child = _mutate(_crossover(parent_a, parent_b, instance, rng), instance, 0.15, rng)
            child_value = evaluate_solution(child, instance).objectives
            ideal = (min(ideal[0], child_value[0]), min(ideal[1], child_value[1]))
            for neighbor in neighbors:
                if _tchebycheff(child, instance, weights[neighbor], ideal) <= _tchebycheff(
                    population[neighbor], instance, weights[neighbor], ideal
                ):
                    population[neighbor] = child
        history.append(tuple(_nondominated_unique(population, instance)))
    front = _nondominated_unique(population, instance)
    return (front, tuple(history)) if record_history else front


def _scalar_tournament(population: Sequence[Solution], instance: ProblemInstance,
                       weight: float, rng: random.Random) -> Solution:
    left, right = rng.sample(range(len(population)), 2)
    return min((population[left], population[right]), key=lambda solution: _scalar_fitness(solution, instance, weight))


def run_weighted_ga(instance: ProblemInstance, weights: Sequence[float] = DEFAULT_WEIGHTS,
                    population_size: int = 100, generations: int = 100,
                    seed: int = 42) -> Dict[float, List[Solution]]:
    """One scalarized GA run per weight; returns one selected solution per weight."""
    results: Dict[float, List[Solution]] = {}
    for offset, weight in enumerate(weights):
        rng = random.Random(seed + 10_000 + offset)
        population = [_random_solution(instance, rng) for _ in range(population_size)]
        for _ in range(generations):
            next_population = []
            while len(next_population) < population_size:
                parent_a = _scalar_tournament(population, instance, weight, rng)
                parent_b = _scalar_tournament(population, instance, weight, rng)
                child = _crossover(parent_a, parent_b, instance, rng)
                next_population.append(_mutate(child, instance, 0.15, rng))
            population = next_population
        best = min(population, key=lambda solution: _scalar_fitness(solution, instance, weight))
        results[weight] = [best]
    return results


def _route_node(point: object, instance: ProblemInstance) -> int:
    return instance.n_points if point == DEPOT else int(point)


def _construct_aco_solution(instance: ProblemInstance, pheromone: List[List[float]],
                            alpha: float, beta: float, rng: random.Random) -> Solution:
    assignments = [-1] * instance.n_points
    priorities = [0.0] * instance.n_points
    unserved = set(range(instance.n_points))
    for uav in range(instance.n_uavs):
        current: object = DEPOT
        remaining_payload = instance.max_payload
        order = 0
        while unserved:
            feasible = [point for point in unserved if instance.demands[point] <= remaining_payload]
            if not feasible:
                break
            current_xy = instance.depot if current == DEPOT else instance.coordinates[int(current)]
            scores = []
            for point in feasible:
                distance = _euclidean(current_xy, instance.coordinates[point])
                scores.append((pheromone[_route_node(current, instance)][point] ** alpha) * ((1.0 / max(distance, 1e-9)) ** beta))
            point = rng.choices(feasible, weights=scores, k=1)[0]
            assignments[point] = uav
            priorities[point] = float(order)
            remaining_payload -= instance.demands[point]
            unserved.remove(point)
            current = point
            order += 1
    if unserved:
        # Repair uses the same payload/range feasibility rules as every other method.
        for point in unserved:
            assignments[point] = min(range(instance.n_uavs), key=lambda uav: sum(
                instance.demands[index] for index, owner in enumerate(assignments) if owner == uav
            ))
    return repair_solution(Solution(tuple(assignments), tuple(priorities)), instance, rng.randrange(1_000_000_000))


def run_weighted_aco(instance: ProblemInstance, weights: Sequence[float] = DEFAULT_WEIGHTS,
                     population_size: int = 100, generations: int = 100,
                     seed: int = 42) -> Dict[float, List[Solution]]:
    """Route-construction ACO, run separately for every scalarization weight."""
    results: Dict[float, List[Solution]] = {}
    node_count = instance.n_points + 1
    for offset, weight in enumerate(weights):
        rng = random.Random(seed + 20_000 + offset)
        pheromone = [[1.0] * node_count for _ in range(node_count)]
        best = None
        best_score = math.inf
        for _ in range(generations):
            ants = [_construct_aco_solution(instance, pheromone, 1.0, 2.0, rng) for _ in range(population_size)]
            scores = [_scalar_fitness(solution, instance, weight) for solution in ants]
            generation_best = ants[scores.index(min(scores))]
            if _scalar_fitness(generation_best, instance, weight) < best_score:
                best, best_score = generation_best, _scalar_fitness(generation_best, instance, weight)
            for left in range(node_count):
                for right in range(node_count):
                    pheromone[left][right] *= 0.9
            for solution, score in zip(ants, scores):
                deposit = 1.0 / max(score, 1e-9)
                for route in decode_routes(solution, instance):
                    for left, right in zip(route, route[1:]):
                        source, destination = _route_node(left, instance), _route_node(right, instance)
                        pheromone[source][destination] += deposit
        results[weight] = [best]
    return results


def _normalize(objectives: Tuple[float, float], lower: Tuple[float, float], upper: Tuple[float, float]) -> Tuple[float, float]:
    return tuple((objectives[index] - lower[index]) / max(upper[index] - lower[index], 1e-12) for index in range(2))


def _hypervolume(approximation: Sequence[Solution], reference: Sequence[Solution], instance: ProblemInstance) -> float:
    all_values = [evaluate_solution(solution, instance).objectives for solution in [*approximation, *reference]]
    lower = (min(value[0] for value in all_values), min(value[1] for value in all_values))
    upper = (max(value[0] for value in all_values), max(value[1] for value in all_values))
    points = sorted({_normalize(evaluate_solution(solution, instance).objectives, lower, upper) for solution in approximation})
    reference_point = (1.1, 1.1)
    height = reference_point[1]
    volume = 0.0
    for distance, imbalance in points:
        if imbalance < height:
            volume += max(0.0, reference_point[0] - distance) * (height - imbalance)
            height = imbalance
    return volume


def _igd(approximation: Sequence[Solution], reference: Sequence[Solution], instance: ProblemInstance) -> float:
    all_values = [evaluate_solution(solution, instance).objectives for solution in [*approximation, *reference]]
    lower = (min(value[0] for value in all_values), min(value[1] for value in all_values))
    upper = (max(value[0] for value in all_values), max(value[1] for value in all_values))
    approximation_points = [_normalize(evaluate_solution(solution, instance).objectives, lower, upper) for solution in approximation]
    reference_points = [_normalize(evaluate_solution(solution, instance).objectives, lower, upper) for solution in reference]
    return sum(min(_euclidean(point, candidate) for candidate in approximation_points) for point in reference_points) / len(reference_points)


def _flatten_weighted(results: Dict[float, List[Solution]], instance: ProblemInstance) -> List[Solution]:
    return _nondominated_unique((solution for solutions in results.values() for solution in solutions), instance)


def run_experiment(instance: ProblemInstance | None = None, runs: int = 30, population_size: int = 100,
                   generations: int = 100, weights: Sequence[float] = DEFAULT_WEIGHTS,
                   output_dir: Path | str = "comparison_results", seed: int = 42) -> Dict[str, Dict[str, float]]:
    instance = instance or default_instance()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithm_runs: Dict[str, List[Tuple[int, List[Solution], float, Tuple[Tuple[Solution, ...], ...]]]] = defaultdict(list)
    for run in range(runs):
        run_seed = seed + run
        started = time.perf_counter()
        insga = run_insga2(instance, population_size, generations, run_seed)
        algorithm_runs["INSGA-II"].append((run + 1, list(insga.front), time.perf_counter() - started, insga.front_history))
        started = time.perf_counter()
        nsga_front, nsga_history = run_standard_nsga2(instance, population_size, generations, run_seed, record_history=True)
        algorithm_runs["NSGA-II"].append((run + 1, nsga_front, time.perf_counter() - started, nsga_history))
        started = time.perf_counter()
        moead_front, moead_history = run_moead(instance, population_size, generations, run_seed, record_history=True)
        algorithm_runs["MOEA/D"].append((run + 1, moead_front, time.perf_counter() - started, moead_history))
        started = time.perf_counter()
        ga_front = _flatten_weighted(run_weighted_ga(instance, weights, population_size, generations, run_seed), instance)
        algorithm_runs["Weighted-GA"].append((run + 1, ga_front, time.perf_counter() - started, ()))
        started = time.perf_counter()
        aco_front = _flatten_weighted(run_weighted_aco(instance, weights, population_size, generations, run_seed), instance)
        algorithm_runs["Weighted-ACO"].append((run + 1, aco_front, time.perf_counter() - started, ()))
    reference = _nondominated_unique(
        (solution for records in algorithm_runs.values() for _, solutions, _, _ in records for solution in solutions), instance
    )
    summary: Dict[str, Dict[str, float]] = {}
    raw_rows = []
    metric_rows = []
    convergence_rows = []
    for name, records in algorithm_runs.items():
        hvs, igds, runtimes = [], [], []
        for run, solutions, elapsed, history in records:
            hv, igd = _hypervolume(solutions, reference, instance), _igd(solutions, reference, instance)
            hvs.append(hv); igds.append(igd); runtimes.append(elapsed)
            metric_rows.append({"algorithm": name, "run": run, "hypervolume": hv, "igd": igd, "runtime_seconds": elapsed})
            for solution_number, solution in enumerate(solutions, 1):
                evaluation = evaluate_solution(solution, instance)
                raw_rows.append({"algorithm": name, "run": run, "solution": solution_number,
                                 "total_distance": evaluation.objectives[0], "load_imbalance": evaluation.objectives[1],
                                 "loads": ";".join(map(str, evaluation.loads)),
                                 "route_distances": ";".join(f"{value:.6f}" for value in evaluation.route_distances),
                                 "assignments": json.dumps(solution.assignments),
                                 "priorities": json.dumps(solution.priorities)})
            for generation, snapshot in enumerate(history, 1):
                convergence_rows.append({"algorithm": name, "run": run, "generation": generation,
                                         "hypervolume": _hypervolume(snapshot, reference, instance),
                                         "igd": _igd(snapshot, reference, instance), "front_size": len(snapshot)})
        summary[name] = {"mean_hypervolume": sum(hvs) / len(hvs), "mean_igd": sum(igds) / len(igds),
                         "mean_runtime_seconds": sum(runtimes) / len(runtimes)}
    _write_csv(output_dir / "raw_solutions.csv", raw_rows)
    _write_csv(output_dir / "run_metrics.csv", metric_rows)
    _write_csv(output_dir / "summary.csv", [{"algorithm": name, **values} for name, values in summary.items()])
    _write_csv(output_dir / "convergence.csv", convergence_rows)
    return summary


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible UAV algorithm comparisons")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--output-dir", default="comparison_results_300gen")
    args = parser.parse_args()
    weights = tuple(float(value) for value in args.weights.split(","))
    summary = run_experiment(default_instance(), args.runs, args.population_size, args.generations,
                             weights, Path(args.output_dir), args.seed)
    for name, values in summary.items():
        print(f"{name}: HV={values['mean_hypervolume']:.6f}, IGD={values['mean_igd']:.6f}, runtime={values['mean_runtime_seconds']:.3f}s")


if __name__ == "__main__":
    main()
