import random
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# —— 参数 & 数据 —— #
NUM_DEMAND_POINTS = 30  # 需求点数量
NUM_UAV = 3             # 无人机数量
MAX_LOAD = 300          # 每架无人机最大载荷
MAX_DISTANCE = 300      # 理论最大飞行距离（用来归一化）
d_load = 0.2            # 载荷对飞行距离影响系数

# 随机或固定需求点坐标与需求量
points = np.random.rand(NUM_DEMAND_POINTS, 2) * 100
demands = [32, 47, 23, 18, 41, 36, 29, 15, 49, 27,
           38, 21, 44, 19, 33, 26, 40, 31, 24, 17,
           42, 28, 35, 16, 48, 30, 22, 45, 20, 34]
start_point = np.array([60, 60])  # 仓库坐标

# —— 辅助函数 —— #
def generate_individual():
    """随机生成一个任务分配染色体"""
    return [random.randint(0, NUM_UAV - 1) for _ in range(NUM_DEMAND_POINTS)]

def calculate_total_distance(tasks):
    """计算一组任务分配的总飞行距离（考虑载荷加成）"""
    total = 0.0
    for u in range(NUM_UAV):
        assigned = [points[i] for i in range(NUM_DEMAND_POINTS) if tasks[i] == u]
        demands_u = [demands[i] for i in range(NUM_DEMAND_POINTS) if tasks[i] == u]
        path = [start_point]
        load = 0.0
        dist = 0.0
        for pt, dm in zip(assigned, demands_u):
            d = np.linalg.norm(pt - path[-1])
            d *= (1 + d_load * (load / MAX_LOAD))
            dist += d
            path.append(pt)
            load += dm
        path.append(start_point)
        dist += np.sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))
        total += dist
    return total

def calculate_load_imbalance(tasks):
    """计算载荷不平衡度 = ∑(Li - L̄)²"""
    loads = [0.0]*NUM_UAV
    for u in range(NUM_UAV):
        loads[u] = sum(demands[i] for i in range(NUM_DEMAND_POINTS) if tasks[i] == u)
    avg = sum(loads)/NUM_UAV
    return sum((L-avg)**2 for L in loads)

def fitness_function(tasks, max_imbalance):
    """返回 (归一化飞行距离, 归一化不平衡, 真实飞行距离)"""
    dist = calculate_total_distance(tasks)
    imb  = calculate_load_imbalance(tasks)
    normalized_dist = dist / MAX_DISTANCE
    normalized_imb = imb / max_imbalance
    return normalized_dist, normalized_imb, dist

# —— 模拟退火相关 —— #
def energy(tasks, max_imbalance, alpha=0.5):
    """加权和能量函数，使用归一化目标"""
    norm_dist, norm_imb, _ = fitness_function(tasks, max_imbalance)
    return alpha * norm_dist + (1 - alpha) * norm_imb

def generate_neighbor(tasks):
    """随机选一个点，改变它的无人机分配"""
    nbr = tasks.copy()
    i = random.randrange(NUM_DEMAND_POINTS)
    choices = [u for u in range(NUM_UAV) if u != tasks[i]]
    nbr[i] = random.choice(choices)
    return nbr

def simulated_annealing(max_iter=5000, T0=1.0, gamma=0.995, alpha=0.5):
    # 预估最大不平衡度
    samples = [calculate_load_imbalance(generate_individual()) for _ in range(200)]
    max_imb = max(samples)

    # 初始化
    current = generate_individual()
    e_curr = energy(current, max_imb, alpha)
    best, e_best = current, e_curr
    T = T0

    # 用来记录能量曲线
    history_curr = []
    history_best = []

    for it in range(max_iter):
        nbr = generate_neighbor(current)
        e_nbr = energy(nbr, max_imb, alpha)

        # 接受准则
        if e_nbr < e_curr or random.random() < math.exp((e_curr - e_nbr)/T):
            current, e_curr = nbr, e_nbr
            if e_nbr < e_best:
                best, e_best = nbr, e_nbr

        # 记录归一化能量（适应度）
        history_curr.append(e_curr)
        history_best.append(e_best)

        # 降温
        T *= gamma
        if T < 1e-6:
            break

    # 返回归一化指标和真实距离
    norm_dist, norm_imb, raw_dist = fitness_function(best, max_imb)
    return best, norm_dist, norm_imb, raw_dist, history_curr, history_best

def plot_solution_paths(tasks):
    """
    根据 tasks 列表画出每架 UAV 的最优访问路径。
    tasks[i] = 第 i 个需求点分给哪架 UAV（0~NUM_UAV-1）
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(points[:,0], points[:,1], c='red', marker='x', s=100, label='需求点')
    for i, pt in enumerate(points):
        ax.annotate(str(demands[i]), (pt[0], pt[1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    colors = ['blue', 'green', 'orange']
    for u in range(NUM_UAV):
        assigned = [points[i] for i in range(NUM_DEMAND_POINTS) if tasks[i] == u]
        path = [start_point] + assigned + [start_point]
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], marker='o', label=f'UAV {u+1}', color=colors[u % len(colors)])

    ax.scatter(start_point[0], start_point[1], c='magenta', marker='*', s=200, label='仓库')
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.set_title('无人机路径（模拟退火结果）')
    ax.legend()
    plt.grid(True)
    plt.show()

# —— 主程序 —— #
if __name__ == "__main__":
    best_tasks, norm_dist, norm_imb, raw_dist, hist_curr, hist_best = simulated_annealing(
        max_iter=5000,
        T0=1.0,
        gamma=0.995,
        alpha=0.6
    )

    print("最优任务分配:", best_tasks)
    print(f"归一化距离: {norm_dist:.4f}")
    print(f"归一化载荷不平衡: {norm_imb:.4f}")
    print(f"总飞行距离（真实值）: {raw_dist:.2f}")

    plt.figure(figsize=(6,4))
    plt.plot(hist_curr, label="当前解能量")
    plt.plot(hist_best, label="全局最优能量")
    plt.xlabel("迭代次数")
    plt.ylabel("归一化能量（越小越好）")
    plt.title("模拟退火能量下降曲线")
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_solution_paths(best_tasks)
