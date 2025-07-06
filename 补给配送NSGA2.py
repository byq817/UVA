import random
import numpy as np
import matplotlib.pyplot as plt
import time
time.sleep(1)


random.seed(42)  # 设置 random 模块的随机种子
np.random.seed(42)  # 设置 numpy 模块的随机种子

# 设置常量
NUM_DEMAND_POINTS = 30  # 需求点数量
NUM_UAV = 3  # 无人机数量
MAX_LOAD = 300  # 每个无人机的最大载荷
MAX_DISTANCE = 300  # 每个无人机的最大飞行范围
d_load = 0.2  # 载荷对飞行距离增加的影响系数

# 初始化需求点与无人机信息
# demands = [random.randint(15, 50) for _ in range(NUM_DEMAND_POINTS)]  # 需求量
points = np.random.rand(NUM_DEMAND_POINTS, 2) * 100  # 需求点的坐标
demands =[32, 47, 23, 18, 41, 36, 29, 15, 49, 27,
38, 21, 44, 19, 33, 26, 40, 31, 24, 17,
42, 28, 35, 16, 48, 30, 22, 45, 20, 34]
# points = np.random.rand(NUM_DEMAND_POINTS, 2) * 100  # 需求点的坐标

start_point = np.array([60, 60])  # 所有无人机都从仓库出发


# 任务分配编码：每个无人机任务分配一个集合
def generate_individual():
    return [random.randint(0, NUM_UAV - 1) for _ in range(NUM_DEMAND_POINTS)]  # 任务分配为列表


# 计算飞行距离
def calculate_total_distance(tasks):
    total_distance = 0
    # print(f"Tasks: {tasks}")  # 添加调试输出
    for i in range(NUM_UAV):
        # 打印当前无人机的任务分配情况
        assigned_points = [points[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]
        assigned_demands = [demands[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]

        # print(f"UAV {i} assigned points: {assigned_points}, demands: {assigned_demands}")  # 调试输出

        path = [start_point]  # 从仓库出发
        current_load = 0  # 当前载荷
        current_distance = 0  # 当前飞行距离

        for j, demand in zip(assigned_points, assigned_demands):
            distance_to_point = np.linalg.norm(j - path[-1])
            adjusted_distance = distance_to_point * (1 + d_load * (current_load / MAX_LOAD))
            current_distance += adjusted_distance
            path.append(j)
            current_load += demand
        path.append(start_point)
        total_distance += np.sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))  # 计算最后一段路径

    return total_distance


# 计算载荷不平衡
def calculate_load_imbalance(tasks):
    load_per_uav = [0] * NUM_UAV
    for i in range(NUM_UAV):
        assigned_demands = [demands[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]
        load_per_uav[i] = sum(assigned_demands)

    avg_load = sum(load_per_uav) / NUM_UAV
    imbalance = sum((load - avg_load) ** 2 for load in load_per_uav)
    return imbalance


# 适应度函数：多个目标
def fitness_function(tasks, max_imbalance):
    total_distance = calculate_total_distance(tasks)
    load_imbalance = calculate_load_imbalance(tasks)

    # 对飞行距离进行归一化
    normalized_distance = total_distance / MAX_DISTANCE

    # 对载荷不平衡进行归一化
    normalized_imbalance = load_imbalance / max_imbalance

    return normalized_distance, normalized_imbalance


# 锦标赛选择函数：在整个种群中选择
def tournament_selection(population, max_imbalance, tournament_size=4):
    selected_parents = []
    # 确保tournament_size不超过population的大小
    tournament_size = min(tournament_size, len(population))

    # 在整个种群中进行锦标赛选择
    while len(selected_parents) < len(population):  # 确保选出足够的父代
        tournament = random.sample(population, tournament_size)  # 从种群中随机选择tournament_size个个体
        # 选择适应度最好的个体
        best_individual = min(tournament, key=lambda ind: fitness_function(ind, max_imbalance))
        selected_parents.append(best_individual)
    return selected_parents


def nsga2_algorithm(pop_size=100, generations=100, crossover_rate=0.7, mutation_rate=0.2):
    population = [generate_individual() for _ in range(pop_size)]
    pareto_fronts = []
    solutions_with_paths = []

    max_imb = max(calculate_load_imbalance(ind) for ind in population)

    for gen in range(generations):
        fitness_vals = [fitness_function(ind, max_imb) for ind in population]

        # 非支配排序
        pareto_front = non_dominated_sorting(fitness_vals)
        pareto_fronts.append(pareto_front)

        # 选择、交叉、变异
        parents = tournament_selection(population, max_imb)
        population = crossover_and_mutation(parents, crossover_rate, mutation_rate)

        # 记录这代最前沿解对应的路径
        flat_front = [idx for front in pareto_front for idx in front]
        sol_paths = []
        for idx in flat_front:
            sol = population[idx]
            paths = []
            for u in range(NUM_UAV):
                pts = [points[j] for j in range(NUM_DEMAND_POINTS) if sol[j]==u]
                p = [start_point] + pts + [start_point]
                paths.append(np.array(p))
            sol_paths.append((sol, paths))
        solutions_with_paths.append(sol_paths)

        print(f"Gen {gen+1}, Pareto fronts: {pareto_front}")

    # 只取最后一代的第一层（非支配）Pareto 前沿
    final_first_front = pareto_fronts[-1][0]
    return final_first_front, solutions_with_paths[-1], population


# Pareto前沿排序
def non_dominated_sorting(fitness_values):
    fronts = []  # 存储每一层Pareto前沿
    remaining = list(range(len(fitness_values)))  # 所有个体的索引

    while remaining:
        front = []  # 当前代的Pareto前沿
        for i in remaining:
            dominated = False
            for j in remaining:
                if dominates(fitness_values[j], fitness_values[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        # 将当前代的Pareto前沿从剩余解中移除
        remaining = [i for i in remaining if i not in front]

        # 添加当前代的Pareto前沿解
        fronts.append(front)

    return fronts



# 判断一个解是否支配另一个解
def dominates(fit1, fit2):
    return (fit1[0] <= fit2[0] and fit1[1] < fit2[1]) or (fit1[0] < fit2[0] and fit1[1] <= fit2[1])


# 交叉和变异操作
def crossover_and_mutation(parents, crossover_rate, mutation_rate):
    next_generation = []
    for _ in range(len(parents) // 2):
        parent1, parent2 = random.sample(parents, 2)
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, NUM_DEMAND_POINTS - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1, parent2
        next_generation.append(child1)
        next_generation.append(child2)

    # 变异
    for i in range(len(next_generation)):
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, NUM_DEMAND_POINTS - 1)
            next_generation[i][mutation_point] = random.randint(0, NUM_UAV - 1)

    return next_generation


# 绘制Pareto前沿和对应路径
# 修改后的plot_pareto_front函数
def plot_pareto_front(pareto_front, solutions_with_paths):
    # 计算最大载荷不平衡值
    max_imbalance = max(calculate_load_imbalance(ind) for ind, _ in solutions_with_paths)

    # 获取 Pareto 前沿的适应度值
    distances = [fitness_function(ind, max_imbalance)[0] for ind, _ in solutions_with_paths]
    imbalances = [fitness_function(ind, max_imbalance)[1] for ind, _ in solutions_with_paths]

    # 绘制Pareto前沿
    plt.scatter(distances, imbalances, color='red', label="Pareto Front")
    plt.xlabel('Total Distance')
    plt.ylabel('Load Imbalance')
    plt.title('Pareto Front')
    plt.show()

    # 绘制对应路径
    for solution, paths in solutions_with_paths:
        uav_paths = paths
        colors = ['blue', 'green', 'orange']

        # 绘制路径
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], c='red', label="Demand Points", s=100, marker='x')

        # 为每个需求点添加需求量标签
        for i, point in enumerate(points):
            ax.annotate(f'{demands[i]}', (point[0], point[1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black')

        for i, path in enumerate(uav_paths):
            ax.plot(path[:, 0], path[:, 1], marker='o', label=f'UAV {i + 1}', color=colors[i % NUM_UAV])

        plt.scatter(start_point[0], start_point[1], color='red', marker='*', s=200, label='Warehouse')
        plt.legend()
        plt.title("UAV Path for Pareto Optimal Solution")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()

def plot_all_and_front(population, pareto_front, max_imbalance):
    # population: 最终种群列表
    # pareto_front: [i, j, k, ...]  —— 一维的整数索引列表
    fits = [fitness_function(ind, max_imbalance) for ind in population]
    dists = [f[0] for f in fits]
    imbs  = [f[1] for f in fits]

    fd = [dists[i] for i in pareto_front]
    fi = [imbs[i]  for i in pareto_front]

    plt.figure(figsize=(6,5))
    plt.scatter(dists, imbs, c='gray', alpha=0.5, label='All solutions')
    plt.scatter(fd,  fi,   c='red',   s=50, alpha=0.8, label='Pareto front')
    plt.xlabel('Normalized Distance')
    plt.ylabel('Normalized Imbalance')
    plt.title('All Solutions vs. Pareto Front')
    plt.legend()
    plt.show()


# 使用NSGA-II进行优化
# pareto_front = nsga2_algorithm()
# pop_size=100
# population = [generate_individual() for _ in range(pop_size)]  # 生成种群
# # 绘制Pareto前沿及对应的路径
# plot_pareto_front(pareto_front, population)
# 使用NSGA-II进行优化
# pareto_front, solutions_with_paths = nsga2_algorithm()



pareto_front, solutions_with_paths, final_population = nsga2_algorithm()
print('pareto_front:',pareto_front)
# 重算最大不平衡用于归一化
max_imbalance = max(calculate_load_imbalance(ind) for ind in final_population)

# 绘制Pareto前沿及对应的路径
""""""
# plot_pareto_front(pareto_front, solutions_with_paths)
# 1) 绘制所有解 & 高亮 Pareto 前沿
# plot_all_and_front(final_population, pareto_front, max_imbalance)


# 任务分配编码：每个无人机任务分配一个集合
def generate_individual():
    return [random.randint(0, NUM_UAV - 1) for _ in range(NUM_DEMAND_POINTS)]  # 任务分配为列表

# 计算飞行距离
def calculate_total_distance(tasks):
    total_distance = 0
    for i in range(NUM_UAV):
        assigned_points = [points[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]
        assigned_demands = [demands[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]

        path = [start_point]  # 从仓库出发
        current_load = 0  # 当前载荷
        current_distance = 0  # 当前飞行距离

        for j, demand in zip(assigned_points, assigned_demands):
            distance_to_point = np.linalg.norm(j - path[-1])
            adjusted_distance = distance_to_point * (1 + d_load * (current_load / MAX_LOAD))
            current_distance += adjusted_distance
            path.append(j)
            current_load += demand
        path.append(start_point)
        total_distance += np.sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))  # 计算最后一段路径

    return total_distance

# 计算载荷不平衡
def calculate_load_imbalance(tasks):
    load_per_uav = [0] * NUM_UAV
    for i in range(NUM_UAV):
        assigned_demands = [demands[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]
        load_per_uav[i] = sum(assigned_demands)

    avg_load = sum(load_per_uav) / NUM_UAV
    imbalance = sum((load - avg_load) ** 2 for load in load_per_uav)
    return imbalance

# 适应度函数：多个目标
def fitness_function(tasks):
    total_distance = calculate_total_distance(tasks)
    load_imbalance = calculate_load_imbalance(tasks)

    # 对飞行距离进行归一化
    normalized_distance = total_distance / MAX_DISTANCE
    # 对载荷不平衡进行归一化
    normalized_imbalance = load_imbalance / (MAX_LOAD * NUM_UAV)

    return normalized_distance + normalized_imbalance  # 目标是最小化飞行距离和载荷不平衡的总和

# 选择函数：选择适应度最好的个体
def select_parents(population):
    population_fitness = [fitness_function(ind) for ind in population]
    selected_parents = random.choices(population, weights=[1/f for f in population_fitness], k=len(population))
    return selected_parents

# 交叉操作
def crossover(parent1, parent2):
    crossover_point = random.randint(1, NUM_DEMAND_POINTS - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# 变异操作
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        mutation_point = random.randint(0, NUM_DEMAND_POINTS - 1)
        individual[mutation_point] = random.randint(0, NUM_UAV - 1)
    return individual

# 遗传算法
def simple_ga(pop_size=100, generations=100, crossover_rate=0.7, mutation_rate=0.2):
    population = [generate_individual() for _ in range(pop_size)]

    for gen in range(generations):
        parents = select_parents(population)

        # 交叉和变异
        next_generation = []
        for _ in range(len(parents) // 2):
            parent1, parent2 = random.sample(parents, 2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            next_generation.append(mutate(child1, mutation_rate))
            next_generation.append(mutate(child2, mutation_rate))

        population = next_generation

        # 输出当前代的最优适应度
        best_fitness = min(fitness_function(ind) for ind in population)
        print(f"Generation {gen + 1}, Best fitness: {best_fitness}")

    # 选择最优解并绘制路径
    best_solution = min(population, key=lambda ind: fitness_function(ind))
    return best_solution

# 执行简单遗传算法
best_solution_ga = simple_ga()

# 绘制 GA 最优解的路径
paths_ga = []
for u in range(NUM_UAV):
    pts = [points[j] for j in range(NUM_DEMAND_POINTS) if best_solution_ga[j] == u]
    p = [start_point] + pts + [start_point]
    paths_ga.append(np.array(p))

# 绘制路径
fig, ax = plt.subplots()
ax.scatter(points[:, 0], points[:, 1], c='red', label="Demand Points", s=100, marker='x')

for i, path in enumerate(paths_ga):
    ax.plot(path[:, 0], path[:, 1], marker='o', label=f'UAV {i + 1}')

ax.scatter(start_point[0], start_point[1], color='red', marker='*', s=200, label='Warehouse')
ax.legend()
ax.set_title("UAV Path for GA Optimal Solution")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
plt.show()

# 蚂蚁群算法 (ACO) 实现
d_load = 0.2  # 载荷对飞行距离增加的影响系数
alpha = 1.0  # 信息素的影响因子
beta = 2.0   # 启发函数的影响因子
rho = 0.1    # 信息素蒸发率
Q = 1.0      # 信息素的总量
# 任务分配编码：每个无人机任务分配一个集合
def generate_individual():
    return [random.randint(0, NUM_UAV - 1) for _ in range(NUM_DEMAND_POINTS)]  # 任务分配为列表

# 计算飞行距离
def calculate_total_distance(tasks):
    total_distance = 0
    for i in range(NUM_UAV):
        assigned_points = [points[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]
        assigned_demands = [demands[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]

        path = [start_point]  # 从仓库出发
        current_load = 0  # 当前载荷
        current_distance = 0  # 当前飞行距离

        for j, demand in zip(assigned_points, assigned_demands):
            distance_to_point = np.linalg.norm(j - path[-1])
            adjusted_distance = distance_to_point * (1 + d_load * (current_load / MAX_LOAD))
            current_distance += adjusted_distance
            path.append(j)
            current_load += demand
        path.append(start_point)
        total_distance += np.sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))  # 计算最后一段路径

    return total_distance

# 计算载荷不平衡
def calculate_load_imbalance(tasks):
    load_per_uav = [0] * NUM_UAV
    for i in range(NUM_UAV):
        assigned_demands = [demands[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]
        load_per_uav[i] = sum(assigned_demands)

    avg_load = sum(load_per_uav) / NUM_UAV
    imbalance = sum((load - avg_load) ** 2 for load in load_per_uav)
    return imbalance

# 适应度函数：多个目标
def evaluate_fitness(tasks):
    total_distance = calculate_total_distance(tasks)
    load_imbalance = calculate_load_imbalance(tasks)

    # 对飞行距离进行归一化
    normalized_distance = total_distance / MAX_DISTANCE
    # 对载荷不平衡进行归一化
    normalized_imbalance = load_imbalance / (MAX_LOAD * NUM_UAV)

    return normalized_distance + normalized_imbalance

# 初始化信息素矩阵
def initialize_pheromone():
    pheromone = np.ones((NUM_DEMAND_POINTS, NUM_UAV))  # 初始化所有路径的信息素为1
    return pheromone

# 更新信息素
def update_pheromone(pheromone, paths, fitness_values):
    pheromone *= (1 - rho)  # 信息素蒸发
    for path, fitness in zip(paths, fitness_values):
        for point in path:
            pheromone[point] += Q / fitness  # 根据信息素更新公式

# ACO算法
def aco_algorithm(pop_size=50, generations=100, alpha=1.0, beta=2.0, rho=0.1, Q=1.0):
    pheromone = initialize_pheromone()
    best_solution = None
    best_fitness = float('inf')

    for gen in range(generations):
        solutions = []
        fitness_values = []

        # 每只蚂蚁选择路径
        for _ in range(pop_size):
            tasks = []
            for i in range(NUM_DEMAND_POINTS):
                # 计算每个任务点选择每个无人机的概率
                # pheromone[i, :] 是任务i到每个无人机的路径信息素
                # (1 / (np.array(demands) + 1)) ** beta 是启发函数（需求量的倒数）
                probabilities = pheromone[i, :] ** alpha * (1 / (np.array(demands[i]) + 1)) ** beta
                probabilities /= np.sum(probabilities)  # 归一化概率

                # 选择任务点分配的无人机
                task = np.random.choice(NUM_UAV, p=probabilities)
                tasks.append(task)

            solutions.append(tasks)
            fitness = evaluate_fitness(tasks)
            fitness_values.append(fitness)

            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = tasks

        # 更新信息素
        update_pheromone(pheromone, solutions, fitness_values)

        # 输出当前代的最优适应度
        print(f"Generation {gen + 1}, Best fitness: {best_fitness}")

    return best_solution

# 执行 ACO 算法
best_solution_aco = aco_algorithm(alpha=1.0, beta=2.0, rho=0.1, Q=1.0)

# 绘制 ACO 最优解的路径
paths_aco = []
for u in range(NUM_UAV):
    pts = [points[j] for j in range(NUM_DEMAND_POINTS) if best_solution_aco[j] == u]
    p = [start_point] + pts + [start_point]
    paths_aco.append(np.array(p))

# 绘制路径
fig, ax = plt.subplots()
ax.scatter(points[:, 0], points[:, 1], c='red', label="Demand Points", s=100, marker='x')

for i, path in enumerate(paths_aco):
    ax.plot(path[:, 0], path[:, 1], marker='o', label=f'UAV {i + 1}')

ax.scatter(start_point[0], start_point[1], color='red', marker='*', s=200, label='Warehouse')
ax.legend()
ax.set_title("UAV Path for ACO Optimal Solution")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
plt.show()


# 计算飞行距离
def calculate_total_distance(tasks):
    total_distance = 0
    for i in range(NUM_UAV):
        assigned_points = [points[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]
        assigned_demands = [demands[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]

        path = [start_point]  # 从仓库出发
        current_load = 0  # 当前载荷
        current_distance = 0  # 当前飞行距离

        for j, demand in zip(assigned_points, assigned_demands):
            distance_to_point = np.linalg.norm(j - path[-1])
            adjusted_distance = distance_to_point * (1 + d_load * (current_load / MAX_LOAD))
            current_distance += adjusted_distance
            path.append(j)
            current_load += demand
        path.append(start_point)
        total_distance += np.sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))  # 计算最后一段路径

    return total_distance

# 计算载荷不平衡
def calculate_load_imbalance(tasks):
    load_per_uav = [0] * NUM_UAV
    for i in range(NUM_UAV):
        assigned_demands = [demands[j] for j in range(NUM_DEMAND_POINTS) if tasks[j] == i]
        load_per_uav[i] = sum(assigned_demands)

    avg_load = sum(load_per_uav) / NUM_UAV
    imbalance = sum((load - avg_load) ** 2 for load in load_per_uav)
    return imbalance

# 计算GA最优解对应的目标函数值
total_distance_ga = calculate_total_distance(best_solution_ga)
load_imbalance_ga = calculate_load_imbalance(best_solution_ga)
print(f"GA 最优解对应的飞行距离: {total_distance_ga}")
print(f"GA 最优解对应的载荷不平衡: {load_imbalance_ga}")

# 计算ACO最优解对应的目标函数值
total_distance_aco = calculate_total_distance(best_solution_aco)
load_imbalance_aco = calculate_load_imbalance(best_solution_aco)
print(f"ACO 最优解对应的飞行距离: {total_distance_aco}")
print(f"ACO 最优解对应的载荷不平衡: {load_imbalance_aco}")


# 计算每个Pareto前沿解的目标函数值
def calculate_pareto_front_fitness(pareto_front, population):
    pareto_fitness = []

    # 遍历 Pareto 前沿中的每个解
    for idx in pareto_front:
        tasks = population[idx]  # 获取该解的任务分配方案

        # 计算飞行距离和载荷不平衡
        total_distance = calculate_total_distance(tasks)
        load_imbalance = calculate_load_imbalance(tasks)

        # 将结果存储在 pareto_fitness 中
        pareto_fitness.append((total_distance, load_imbalance))

    return pareto_fitness


# 获取 Pareto 前沿解对应的目标函数值
pareto_fitness = calculate_pareto_front_fitness(pareto_front, final_population)

# 输出每个 Pareto 前沿解的飞行距离和载荷不平衡
for i, (total_distance, load_imbalance) in enumerate(pareto_fitness):
    print(f"Pareto 解 {i + 1} 对应的飞行距离: {total_distance}")
    print(f"Pareto 解 {i + 1} 对应的载荷不平衡: {load_imbalance}")