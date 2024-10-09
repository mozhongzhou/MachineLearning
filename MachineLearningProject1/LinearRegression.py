import numpy as np

import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import time

# 生成人工线性数据
np.random.seed(0)
x = np.random.rand(100) * 10  # 100个随机点，范围在 [0, 10]
y = 3 * x + 7 + np.random.randn(100)  # 带有噪声的线性关系

# 方法1：使用公式1
#计算时间
start_time = time.time()
#核心
x_mean = np.mean(x)
y_mean = np.mean(y)
w1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
w0 = y_mean - w1 * x_mean
print(f"方法1: w1 = {w1}, w0 = {w0}")

time_method_1 = time.time() - start_time
##########################################################
# 方法2：使用正规方程
#计算时间
start_time = time.time()
#核心
X = np.vstack([np.ones(len(x)), x]).T
w = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"方法2: w = {w}")

time_method_2 = time.time() - start_time
###########################################################
# 比较运行时间
print(f"方法1的时间: {time_method_1}")
print(f"方法2的时间: {time_method_2}")

# 绘制结果
plt.scatter(x, y, label='数据点')
plt.plot(x, w1 * x + w0, color='red', label='方法1的拟合线')
plt.plot(x, X @ w, color='blue', linestyle='--', label='方法2的拟合线')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
