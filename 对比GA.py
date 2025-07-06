import random
import numpy as np
import matplotlib.pyplot as plt

# —— 参数和数据 —— #
NUM_DEMAND_POINTS = 30
NUM_UAV = 3
MAX_LOAD = 300
MAX_DISTANCE = 300  # 用于距离归一化
d_load = 0.2

points = np.random.rand(NUM_DEMAND_POINTS, 2) * 100
demands = [32, 47, 23, 18, 41, 36, 29, 15, 49, 27,
           38, 21, 44, 19, 33, 26, 40, 31, 24, 17,
           42, 28, 35, 16, 48, 30, 22, 45, 20, 34]
start_point = np.array([60, 60])

# —— 计算函数 —— #
def calculate_total_distance(tasks):
    total_distance = 0
    for u in range(NUM_UAV):
        assigned_points = [points[i] for i in range(NUM_DEMAND_POINTS) if tasks[i] == u]
        assigned_demands = [demands[i] for i in range(NUM_DEMAND_POINTS) if tasks[i] == u]
        path = [start_point]
        current_load = 0
        dist = 0
        for pt, dm in zip(assigned_points, assigned_demands):
            d = np.linalg.norm(pt - path[-1])
            d *= (1 + d_load * (current_load / MAX_LOAD))
            dist += d
            path.append(pt)
            current_load += dm
        path.append(start_point)
        dist += np.sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))
        total_distance += dist
    return total_distance

def calculate_load_imbalance(tasks):
    loads = [0]*NUM_UAV
    for u in range(NUM_UAV):
        loads[u] = sum(demands[i] for i in range(NUM_DEMAND_POINTS) if tasks[i] == u)
    avg = sum(loads)/NUM_UAV
    return sum((L - avg)**2 for L in loads)

def fitness_function(tasks, max_imbalance):
    dist = calculate_total_distance(tasks)
    imb = calculate_load_imbalance(tasks)
    normalized_dist = dist / MAX_DISTANCE
    normalized_imb = imb / max_imbalance
    return normalized_dist, normalized_imb, dist

def weighted_fitness(tasks, max_imbalance, alpha=0.5):
    norm_dist, norm_imb, _ = fitness_function(tasks, max_imbalance)
    return alpha * norm_dist + (1 - alpha) * norm_imb

# —— GA基本操作 —— #
def generate_individual():
    return [random.randint(0, NUM_UAV - 1) for _ in range(NUM_DEMAND_POINTS)]

def crossover(parent1, parent2):
    point = random.randint(1, NUM_DEMAND_POINTS - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, mutation_rate=0.05):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, NUM_UAV - 1)

def tournament_selection(population, fitnesses, k=3):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        aspirants = random.sample(range(pop_size), k)
        best = min(aspirants, key=lambda idx: fitnesses[idx])
        selected.append(population[best])
    return selected

# —— GA主循环 —— #
def genetic_algorithm(pop_size=100, generations=100, crossover_rate=0.7, mutation_rate=0.05, alpha=0.5):
    population = [generate_individual() for _ in range(pop_size)]
    max_imbalance = max(calculate_load_imbalance(ind) for ind in population)

    best_solution = None
    best_fitness = float('inf')

    history_best_fitness = []
    history_best_norm_dist = []
    history_best_norm_imb = []

    for gen in range(generations):
        fitnesses = [weighted_fitness(ind, max_imbalance, alpha) for ind in population]

        min_fit = min(fitnesses)
        best_idx = fitnesses.index(min_fit)
        if min_fit < best_fitness:
            best_fitness = min_fit
            best_solution = population[best_idx]

        norm_dist, norm_imb, raw_dist = fitness_function(best_solution, max_imbalance)

        history_best_fitness.append(min_fit)
        history_best_norm_dist.append(norm_dist)
        history_best_norm_imb.append(norm_imb)

        selected = tournament_selection(population, fitnesses)

        next_population = []
        while len(next_population) < pop_size:
            if random.random() < crossover_rate and len(selected) >= 2:
                p1 = random.choice(selected)
                p2 = random.choice(selected)
                c1, c2 = crossover(p1, p2)
                mutate(c1, mutation_rate)
                mutate(c2, mutation_rate)
                next_population.extend([c1, c2])
            else:
                ind = random.choice(selected).copy()
                mutate(ind, mutation_rate)
                next_population.append(ind)

        population = next_population[:pop_size]

        print(f"Generation {gen+1}: Best Weighted Fitness = {best_fitness:.4f}")

    return best_solution, best_fitness, raw_dist, norm_imb, \
           history_best_fitness, history_best_norm_dist, history_best_norm_imb

# —— 绘制路径函数 —— #
def plot_solution_paths(tasks):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(points[:, 0], points[:, 1], c='red', marker='x', s=100, label='Demand Points')
    for i, pt in enumerate(points):
        ax.annotate(str(demands[i]), (pt[0], pt[1]), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=8)

    colors = ['blue', 'green', 'orange']
    for u in range(NUM_UAV):
        assigned_pts = [points[i] for i in range(NUM_DEMAND_POINTS) if tasks[i] == u]
        path = [start_point] + assigned_pts + [start_point]
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], marker='o', label=f'UAV {u+1}', color=colors[u % len(colors)])

    ax.scatter(start_point[0], start_point[1], c='magenta', marker='*', s=200, label='Warehouse')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Best UAV Routes (GA Result)')
    ax.legend()
    plt.grid(True)
    plt.show()

# —— 主程序 —— #
if __name__ == "__main__":
    best_tasks, best_fit, best_dist, best_imb, hist_fit, hist_norm_dist, hist_norm_imb = genetic_algorithm(
        pop_size=100, generations=100, crossover_rate=0.7, mutation_rate=0.05, alpha=0.6
    )

    print("Best task assignment:", best_tasks)
    print(f"Total Flight Distance (raw): {best_dist:.2f}")
    print(f"Normalized Load Imbalance: {best_imb:.4f}")

    generations = range(1, len(hist_fit) + 1)
    plt.figure(figsize=(10,6))
    plt.plot(generations, hist_fit, label='Best Weighted Fitness')
    # plt.plot(generations, hist_norm_dist, label='Best Normalized Distance')
    # plt.plot(generations, hist_norm_imb, label='Best Normalized Load Imbalance')
    plt.xlabel('Generation')
    plt.ylabel('Normalized Value')
    plt.title('Fitness and Normalized Objectives over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_solution_paths(best_tasks)
