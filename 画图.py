import matplotlib.pyplot as plt
from matplotlib import rcParams

# 配置中文字体
rcParams['font.family'] = 'SimHei'  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 允许显示负号

# 数据
tasks = [10, 20, 30, 40, 50, 60]
response_time = [1, 3, 7, 15, 30, 60]

# 创建折线图
plt.figure(figsize=(8, 6))
plt.plot(tasks, response_time, marker='o')

# 标注性能瓶颈
plt.axvline(x=30, color='red', linestyle='--')
plt.text(31, 7, '性能瓶颈\n资源耗尽', color='red')

# 图表标题和标签
plt.title('单一计算节点响应时间随任务量变化')
plt.xlabel('任务量')
plt.ylabel('响应时间（单位）')

# 显示图表
plt.grid(True)
plt.show()
