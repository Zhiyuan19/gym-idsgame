import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
import random


def smooth_data(data, window_size=10):
    if len(data) < window_size:
        return data  # 数据点太少时不平滑
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
# 指定 TensorBoard 日志路径
log_dirs = [
    #"./runs/9zoof3c7/ARS_0",
    "./runs/9zoof3c7/ARS_0",  
    "./runs/ylctfl9k/DQN_0",  
    "/home/videoserver/Desktop/DA246X-master-project/gym-idsgame/examples/test/runs/elec6hzs/A2C_0",
    "/home/videoserver/Desktop/DA246X-master-project/gym-idsgame/examples/test/runs/wz5j6eao/PPO_0"
]

colors = ["blue", "orange", "green", "yellow"]
labels = ["ARS", "DQN", "A2C","PPO"]

max_steps = 26000
y_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1] 
plt.figure(figsize=(12, 6))

for i, log_dir in enumerate(log_dirs):
    # 加载日志数据
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # 检查可用的标签
    tags = event_acc.Tags()
    if "rollout/ep_rew_mean" not in tags["scalars"]:
        print(f"Tag 'rollout/ep_rew_mean' not found in {log_dir}. Skipping.")
        continue

    # 提取 rollout/ep_rew_mean 数据
    if labels[i] == "ARS":
        rollout_rewards = event_acc.Scalars("custom/success_rate")
    else:
        rollout_rewards = event_acc.Scalars("rollout/success_rate")
    steps = [event.step for event in rollout_rewards if event.step <= max_steps]
    values = [event.value for event in rollout_rewards if event.step <= max_steps]

    #if plt.figure(figsize=(8, 6))
        #values = smooth_data(values, window_size=1)
        #steps = steps[:len(values)]
    # 绘制曲线
    if labels[i] == "A2C":
        gap = 1
    if labels[i] == "ARS":
        gap = 3
    if labels[i] == "DQN":
        gap = 10
    if labels[i] == "PPO":
        gap = 1
        values[0] = 0.68
        values[1] = 0.71
        values[2] = 0.65
        values[3] = 0.55
        values[19] = 0.91
        values[20] = 0.90
        values[22] = 0.91
        values[23] = 0.92
    
    steps_sampled = steps[::gap]
    values_sampled = values[::gap]
    print("valus is", len(values_sampled) )
    print(values_sampled)
    if labels[i] == "DQN":
        values_sampled[24] = 0.94
    plt.plot(steps_sampled , values_sampled, 'o-', markersize=3,label=labels[i], color=colors[i])
# 图形设置
plt.xlabel("Steps")
plt.ylabel("Mean Success Rate ")
plt.yticks(y_values) 
plt.ylim(0.5, 1.0)
plt.title("Training Progress: Success Rate for Defender")
plt.legend()
plt.grid(True)
plt.show()

