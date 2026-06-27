"""Single-trip, capacitated INSGA-II for the manuscript's UAV delivery case.

Each UAV leaves the depot once, visits its assigned demand points, and returns
once.  Demands are indivisible; no mid-route replenishment is permitted.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


DEPOT = "DEPOT"
Point = Tuple[float, float]
Objectives = Tuple[float, float]


@dataclass(frozen=True)
class ProblemInstance:
    coordinates: Tuple[Point, ...]
    demands: Tuple[int, ...]
    depot: Point
    n_uavs: int = 3
    max_payload: int = 400
    max_route_distance: float = 600.0

    @property
    def n_points(self) -> int:
        return len(self.demands)


@dataclass(frozen=True)
class Solution:
    assignments: Tuple[int, ...]
    priorities: Tuple[float, ...]


@dataclass(frozen=True)
class Evaluation:
    feasible: bool
    objectives: Objectives
    loads: Tuple[int, ...]
    route_distances: Tuple[float, ...]
    routes: Tuple[Tuple[object, ...], ...]


@dataclass(frozen=True)
class RunResult:
    front: Tuple[Solution, ...]
    history: Tuple[Dict[str, float], ...]
    front_history: Tuple[Tuple[Solution, ...], ...] = ()


def default_instance() -> ProblemInstance:
    """Return the 30-point, seed-42 scenario used by the legacy scripts."""
    coordinates = (
        (37.454011884736246, 95.07143064099162), (73.1993941811405, 59.86584841970366),
        (15.601864044243651, 15.599452033620265), (5.8083612168199465, 86.61761457749351),
        (60.11150117432088, 70.80725777960456), (2.0584494295802447, 96.99098521619943),
        (83.24426408004217, 21.233911067827), (18.182496720710063, 18.34045098534338),
        (30.42422429595377, 52.475643163223786), (43.194501864211574, 29.122914019804192),
        (61.18528947223795, 13.949386065204184), (29.214464853521815, 36.63618432936917),
        (45.606998421703594, 78.51759613930136), (19.967378215835975, 51.42344384136116),
        (59.24145688620425, 4.645041271999773), (60.75448519014384, 17.052412368729154),
        (6.505159298527952, 94.88855372533332), (96.56320330745594, 80.83973481164611),
        (30.46137691733707, 9.767211400638388), (68.42330265121569, 44.01524937396013),
        (12.203823484477883, 49.51769101112702), (3.4388521115218396, 90.9320402078782),
        (25.87799816000169, 66.2522284353982), (31.171107608941096, 52.00680211778108),
        (54.67102793432797, 18.485445552552704), (96.95846277645586, 77.51328233611146),
        (93.9498941564189, 89.48273504276489), (59.78999788110851, 92.18742350231169),
        (8.84925020519195, 19.59828624191452), (4.522728891053807, 32.53303307632643),
    )
    demands = (32, 47, 23, 18, 41, 36, 29, 15, 49, 27, 38, 21, 44, 19, 33,
               26, 40, 31, 24, 17, 42, 28, 35, 16, 48, 30, 22, 45, 20, 34)
    return ProblemInstance(coordinates=coordinates, demands=demands, depot=(60.0, 60.0))


def _distance(left: Point, right: Point) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def assignment_loads(assignments: Sequence[int], instance: ProblemInstance) -> List[int]:
    loads = [0] * instance.n_uavs
    for point, owner in enumerate(assignments):
        if 0 <= owner < instance.n_uavs:
            loads[owner] += instance.demands[point]
    return loads


def _nearest_neighbor_priorities(assignments: Sequence[int], instance: ProblemInstance) -> Tuple[float, ...]:
    priorities = [0.0] * instance.n_points
    for uav in range(instance.n_uavs):
        remaining = [point for point, owner in enumerate(assignments) if owner == uav]
        current = instance.depot
        rank = 0
        while remaining:
            point = min(remaining, key=lambda idx: _distance(current, instance.coordinates[idx]))
            priorities[point] = float(rank)
            rank += 1
            current = instance.coordinates[point]
            remaining.remove(point)
    return tuple(priorities)


def decode_routes(solution: Solution, instance: ProblemInstance) -> Tuple[Tuple[object, ...], ...]:
    routes = []
    for uav in range(instance.n_uavs):
        points = [point for point, owner in enumerate(solution.assignments) if owner == uav]
        points.sort(key=lambda point: solution.priorities[point])
        routes.append((DEPOT, *points, DEPOT))
    return tuple(routes)


def _route_distance(route: Sequence[object], instance: ProblemInstance) -> float:
    locations = [instance.depot if point == DEPOT else instance.coordinates[point] for point in route]
    return sum(_distance(locations[index], locations[index + 1]) for index in range(len(locations) - 1))


def evaluate_solution(solution: Solution, instance: ProblemInstance) -> Evaluation:
    routes = decode_routes(solution, instance)
    served = [point for route in routes for point in route[1:-1]]
    valid_owners = len(solution.assignments) == instance.n_points and all(
        0 <= owner < instance.n_uavs for owner in solution.assignments
    )
    covered_once = sorted(served) == list(range(instance.n_points))
    loads = tuple(assignment_loads(solution.assignments, instance))
    route_distances = tuple(_route_distance(route, instance) for route in routes)
    feasible = (
        valid_owners
        and covered_once
        and all(load <= instance.max_payload for load in loads)
        and all(distance <= instance.max_route_distance for distance in route_distances)
    )
    if not feasible:
        return Evaluation(False, (math.inf, math.inf), loads, route_distances, routes)
    average_load = sum(loads) / instance.n_uavs
    imbalance = sum((load - average_load) ** 2 for load in loads)
    return Evaluation(True, (sum(route_distances), imbalance), loads, route_distances, routes)


def _repair_payload(assignments: List[int], instance: ProblemInstance, rng: random.Random) -> None:
    loads = assignment_loads(assignments, instance)
    while max(loads) > instance.max_payload:
        source = max(range(instance.n_uavs), key=loads.__getitem__)
        candidates = sorted(
            (point for point, owner in enumerate(assignments) if owner == source),
            key=lambda point: (instance.demands[point], rng.random()),
            reverse=True,
        )
        moved = False
        for point in candidates:
            destinations = sorted(
                (uav for uav in range(instance.n_uavs) if uav != source), key=loads.__getitem__
            )
            for destination in destinations:
                if loads[destination] + instance.demands[point] <= instance.max_payload:
                    assignments[point] = destination
                    loads[source] -= instance.demands[point]
                    loads[destination] += instance.demands[point]
                    moved = True
                    break
            if moved:
                break
        if not moved:
            raise ValueError("No payload-feasible single-trip assignment exists")


def repair_solution(solution: Solution, instance: ProblemInstance, seed: int | None = None) -> Solution:
    """Repair payload violations while retaining a feasible inherited route order."""
    if len(solution.assignments) != instance.n_points:
        raise ValueError("Assignments must contain exactly one owner for each demand point")
    rng = random.Random(seed)
    assignments = [owner if 0 <= owner < instance.n_uavs else rng.randrange(instance.n_uavs)
                   for owner in solution.assignments]
    _repair_payload(assignments, instance, rng)
    inherited_priorities = solution.priorities
    if len(inherited_priorities) != instance.n_points:
        inherited_priorities = _nearest_neighbor_priorities(assignments, instance)
    repaired = Solution(tuple(assignments), inherited_priorities)
    evaluation = evaluate_solution(repaired, instance)
    if evaluation.feasible:
        return repaired

    repaired = Solution(tuple(assignments), _nearest_neighbor_priorities(assignments, instance))
    if evaluate_solution(repaired, instance).feasible:
        return repaired

    # If a nearest-neighbor route exceeds the range, move a farthest stop away from
    # the longest route to an UAV with payload slack and rebuild all route orders.
    for _ in range(instance.n_points * instance.n_uavs):
        evaluation = evaluate_solution(repaired, instance)
        if evaluation.feasible:
            return repaired
        source = max(range(instance.n_uavs), key=lambda uav: evaluation.route_distances[uav])
        source_points = [point for point, owner in enumerate(assignments) if owner == source]
        source_points.sort(key=lambda point: _distance(instance.depot, instance.coordinates[point]), reverse=True)
        loads = assignment_loads(assignments, instance)
        moved = False
        for point in source_points:
            for destination in sorted(range(instance.n_uavs), key=loads.__getitem__):
                if destination != source and loads[destination] + instance.demands[point] <= instance.max_payload:
                    assignments[point] = destination
                    repaired = Solution(tuple(assignments), _nearest_neighbor_priorities(assignments, instance))
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    if not evaluate_solution(repaired, instance).feasible:
        raise ValueError("No route-feasible single-trip assignment found for this instance")
    return repaired


def make_seed_solution(instance: ProblemInstance, seed: int = 42) -> Solution:
    rng = random.Random(seed)
    assignments = [-1] * instance.n_points
    loads = [0] * instance.n_uavs
    for point in sorted(range(instance.n_points), key=lambda idx: instance.demands[idx], reverse=True):
        feasible_uavs = [uav for uav in range(instance.n_uavs)
                         if loads[uav] + instance.demands[point] <= instance.max_payload]
        if not feasible_uavs:
            raise ValueError("Payload capacity is insufficient for the configured instance")
        minimum = min(loads[uav] for uav in feasible_uavs)
        choices = [uav for uav in feasible_uavs if loads[uav] == minimum]
        owner = rng.choice(choices)
        assignments[point] = owner
        loads[owner] += instance.demands[point]
    return repair_solution(Solution(tuple(assignments), tuple(range(instance.n_points))), instance, seed)


def _random_solution(instance: ProblemInstance, rng: random.Random) -> Solution:
    for attempt in range(100):
        base = make_seed_solution(instance, seed=rng.randrange(1_000_000_000))
        assignments = list(base.assignments)
        for point in range(instance.n_points):
            if rng.random() < 0.25:
                assignments[point] = rng.randrange(instance.n_uavs)
        try:
            return repair_solution(Solution(tuple(assignments), tuple(rng.random() for _ in assignments)), instance,
                                   rng.randrange(1_000_000_000))
        except ValueError:
            continue
    raise RuntimeError("Unable to create a feasible initial population")


def dominates(left: Objectives, right: Objectives) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def _fast_non_dominated_sort(population: Sequence[Solution], instance: ProblemInstance) -> Tuple[List[List[int]], List[Objectives]]:
    objectives = [evaluate_solution(solution, instance).objectives for solution in population]
    domination_sets: List[List[int]] = [[] for _ in population]
    dominated_by_count = [0] * len(population)
    fronts: List[List[int]] = [[]]
    for left in range(len(population)):
        for right in range(len(population)):
            if left == right:
                continue
            if dominates(objectives[left], objectives[right]):
                domination_sets[left].append(right)
            elif dominates(objectives[right], objectives[left]):
                dominated_by_count[left] += 1
        if dominated_by_count[left] == 0:
            fronts[0].append(left)
    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: List[int] = []
        for left in fronts[front_index]:
            for right in domination_sets[left]:
                dominated_by_count[right] -= 1
                if dominated_by_count[right] == 0:
                    next_front.append(right)
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return fronts, objectives


def _rank_and_crowding(population: Sequence[Solution], instance: ProblemInstance):
    fronts, objectives = _fast_non_dominated_sort(population, instance)
    ranks = [math.inf] * len(population)
    crowding = [0.0] * len(population)
    for rank, front in enumerate(fronts):
        for index in front:
            ranks[index] = rank
        if len(front) <= 2:
            for index in front:
                crowding[index] = math.inf
            continue
        for objective_index in range(2):
            ordered = sorted(front, key=lambda index: objectives[index][objective_index])
            crowding[ordered[0]] = math.inf
            crowding[ordered[-1]] = math.inf
            lower = objectives[ordered[0]][objective_index]
            upper = objectives[ordered[-1]][objective_index]
            if upper == lower:
                continue
            for position in range(1, len(ordered) - 1):
                index = ordered[position]
                if not math.isinf(crowding[index]):
                    crowding[index] += (
                        objectives[ordered[position + 1]][objective_index]
                        - objectives[ordered[position - 1]][objective_index]
                    ) / (upper - lower)
    return fronts, objectives, ranks, crowding


def _better(left: int, right: int, ranks: Sequence[float], crowding: Sequence[float]) -> int:
    if ranks[left] != ranks[right]:
        return left if ranks[left] < ranks[right] else right
    return left if crowding[left] >= crowding[right] else right


def _tournament(population: Sequence[Solution], ranks, crowding, rng: random.Random) -> Solution:
    left, right = rng.sample(range(len(population)), 2)
    return population[_better(left, right, ranks, crowding)]


def _crossover(left: Solution, right: Solution, instance: ProblemInstance, rng: random.Random) -> Solution:
    assignments = tuple(left.assignments[i] if rng.random() < 0.5 else right.assignments[i]
                        for i in range(instance.n_points))
    priorities = tuple(left.priorities[i] if rng.random() < 0.5 else right.priorities[i]
                       for i in range(instance.n_points))
    return repair_solution(Solution(assignments, priorities), instance, rng.randrange(1_000_000_000))


def _mutate(solution: Solution, instance: ProblemInstance, mutation_rate: float, rng: random.Random) -> Solution:
    assignments = list(solution.assignments)
    priorities = list(solution.priorities)
    if rng.random() < mutation_rate:
        point = rng.randrange(instance.n_points)
        assignments[point] = rng.randrange(instance.n_uavs)
    if rng.random() < mutation_rate:
        left, right = rng.sample(range(instance.n_points), 2)
        priorities[left], priorities[right] = priorities[right], priorities[left]
    try:
        candidate = repair_solution(Solution(tuple(assignments), tuple(priorities)), instance,
                                    rng.randrange(1_000_000_000))
        # Keep route-order variation when it remains single-trip feasible.
        ordered = Solution(candidate.assignments, tuple(priorities))
        return ordered if evaluate_solution(ordered, instance).feasible else candidate
    except ValueError:
        return solution


def _environmental_selection(candidates: Sequence[Solution], size: int, instance: ProblemInstance) -> List[Solution]:
    fronts, _, _, crowding = _rank_and_crowding(candidates, instance)
    selected: List[Solution] = []
    for front in fronts:
        remaining = size - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(candidates[index] for index in front)
        else:
            selected.extend(candidates[index] for index in sorted(front, key=lambda index: crowding[index], reverse=True)[:remaining])
            break
    return selected


def _immune_clones(population: Sequence[Solution], instance: ProblemInstance, generation: int,
                   generations: int, rng: random.Random, clone_factor: int = 2) -> List[Solution]:
    _, _, ranks, crowding = _rank_and_crowding(population, instance)
    progress = generation / max(1, generations - 1)
    dynamic_mutation_rate = 0.30 + progress * (0.05 - 0.30)
    clones: List[Solution] = []
    for index, solution in enumerate(population):
        if ranks[index] > 1:
            continue
        stimulation = 1.0 / (1.0 + ranks[index])
        if math.isinf(crowding[index]) or crowding[index] > 0.5:
            stimulation += 1.0
        clone_count = max(1, int(math.ceil(clone_factor * stimulation)))
        for _ in range(clone_count):
            clones.append(_mutate(solution, instance, dynamic_mutation_rate, rng))
    return clones


def run_insga2(instance: ProblemInstance, population_size: int = 100, generations: int = 100,
               seed: int = 42) -> RunResult:
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    rng = random.Random(seed)
    population = [_random_solution(instance, rng) for _ in range(population_size)]
    history: List[Dict[str, float]] = []
    front_history: List[Tuple[Solution, ...]] = []
    for generation in range(generations):
        _, objectives, ranks, crowding = _rank_and_crowding(population, instance)
        progress = generation / max(1, generations - 1)
        mutation_rate = 0.30 + progress * (0.05 - 0.30)
        offspring: List[Solution] = []
        while len(offspring) < population_size:
            parent_a = _tournament(population, ranks, crowding, rng)
            parent_b = _tournament(population, ranks, crowding, rng)
            offspring.append(_mutate(_crossover(parent_a, parent_b, instance, rng), instance, mutation_rate, rng))
        clones = _immune_clones(population, instance, generation, generations, rng)
        population = _environmental_selection([*population, *offspring, *clones], population_size, instance)
        generation_fronts, _, _, _ = _rank_and_crowding(population, instance)
        generation_front = []
        seen_generation = set()
        for index in generation_fronts[0]:
            candidate = population[index]
            if candidate not in seen_generation:
                generation_front.append(candidate)
                seen_generation.add(candidate)
        front_history.append(tuple(generation_front))
        feasible_objectives = [objective for objective in objectives if math.isfinite(objective[0])]
        history.append({
            "generation": float(generation + 1),
            "mutation_rate": mutation_rate,
            "best_distance": min(objective[0] for objective in feasible_objectives),
            "best_imbalance": min(objective[1] for objective in feasible_objectives),
        })
    fronts, _, _, _ = _rank_and_crowding(population, instance)
    front: List[Solution] = []
    seen = set()
    for index in fronts[0]:
        solution = population[index]
        if solution not in seen:
            front.append(solution)
            seen.add(solution)
    return RunResult(tuple(front), tuple(history), tuple(front_history))


def summarize_solution(solution: Solution, instance: ProblemInstance) -> Dict[str, object]:
    evaluation = evaluate_solution(solution, instance)
    return {
        "feasible": evaluation.feasible,
        "objectives": evaluation.objectives,
        "loads": evaluation.loads,
        "route_distances": evaluation.route_distances,
        "routes": evaluation.routes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-trip INSGA-II experiment")
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    instance = default_instance()
    result = run_insga2(instance, args.population_size, args.generations, args.seed)
    print(f"Instance: {instance.n_points} points, {instance.n_uavs} UAVs, payload={instance.max_payload}, range={instance.max_route_distance}")
    print(f"Nondominated feasible solutions: {len(result.front)}")
    for number, solution in enumerate(result.front, 1):
        summary = summarize_solution(solution, instance)
        print(f"Solution {number}: distance={summary['objectives'][0]:.3f}, load_imbalance={summary['objectives'][1]:.3f}")
        for uav, (load, distance, route) in enumerate(zip(summary["loads"], summary["route_distances"], summary["routes"]), 1):
            printable = " -> ".join("Depot" if point == DEPOT else str(point + 1) for point in route)
            print(f"  UAV {uav}: load={load}, route_distance={distance:.3f}, route={printable}")


if __name__ == "__main__":
    main()
