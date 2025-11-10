import numpy as np

# 1. 定义采样参数
N = 1024       # 采样点数 n
dt = 1   # 采样间隔 d (秒)

# 采样频率 fs = 1 / dt = 1000 Hz

# 2. 使用 rfftfreq 计算频率坐标
frequencies = np.fft.rfftfreq(n=N, d=dt)

print(f"信号长度 N: {N}")
print(f"频率数组的长度: {len(frequencies)}")
print(f"前 5 个频率: {frequencies[:-5]}")
print(f"最大频率 (奈奎斯特频率): {frequencies[-1]}")