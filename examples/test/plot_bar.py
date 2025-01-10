import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import random

categories = ['Managemen Host', 'Firewall', 'Videoserver']
ARS = [0.92, 0.85, 0.1]  # Regular 数据
DQN = [0.88, 0.94, 0.2]  # Optimal 数据
A2C = [0.88, 0.94, 0.2]
PPO = [0.88, 0.94, 0.2]
log_dirs = [
    #"./runs/9zoof3c7/ARS_0",
    "./runs/3fio5m9x/ARS_0",  
    "./runs/8ok5d9d4/DQN_0",  
    "/home/videoserver/Desktop/DA246X-master-project/gym-idsgame/examples/test/runs/4xsugpzr/A2C_0",
    "/home/videoserver/Desktop/DA246X-master-project/gym-idsgame/examples/test/runs/wz5j6eao/PPO_0"
]

for i, log_dir in enumerate(log_dirs):
    max_steps = 26000
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    fwfail = event_acc.Scalars("custom/fw_failattacks")
    fwsuccess = event_acc.Scalars("custom/fw_successattacks")
    mafail = event_acc.Scalars("custom/ma_failattacks")
    masuccess = event_acc.Scalars("custom/ma_successattacks")
    wsfail = event_acc.Scalars("custom/ws_failattacks")
    wssuccess = event_acc.Scalars("custom/ws_successattacks")
    steps = [event.step for event in fwfail if event.step <= max_steps]
    fwfail_values = [event.value for event in fwfail if event.step <= max_steps]
    fwsuccess_values = [event.value for event in fwsuccess if event.step <= max_steps]
    fwtotal_attempts = [f + s for f, s in zip(fwfail_values, fwsuccess_values)]
    fwsuccess_ratios = [s / t if t > 0 else 0 for s, t in zip(fwsuccess_values, fwtotal_attempts)]
    mafail_values = [event.value for event in mafail if event.step <= max_steps]
    masuccess_values = [event.value for event in masuccess if event.step <= max_steps]
    matotal_attempts = [f + s for f, s in zip(mafail_values, masuccess_values)]
    masuccess_ratios = [s / t if t > 0 else 0 for s, t in zip(masuccess_values, matotal_attempts)]
    wsfail_values = [event.value for event in wsfail if event.step <= max_steps]
    wssuccess_values = [event.value for event in wssuccess if event.step <= max_steps]
    wstotal_attempts = [f + s for f, s in zip(wsfail_values, wssuccess_values)]
    wssuccess_ratios = [s / t if t > 0 else 0 for s, t in zip(wssuccess_values, wstotal_attempts)]
    if i == 0:
        counter = 30
    if i == 1:
        counter = 130
    if i == 2:
        counter = 20
    if i == 3:
        counter = 20
    fwlast_20_ratios = fwsuccess_ratios[-counter:]
    malast_20_ratios = masuccess_ratios[-counter:]
    wslast_20_ratios = wssuccess_ratios[-counter:]
    average_fwsuccess_ratio = sum(fwlast_20_ratios) / len(fwlast_20_ratios)
    average_masuccess_ratio = sum(malast_20_ratios) / len(malast_20_ratios)
    average_wssuccess_ratio = sum(wslast_20_ratios) / len(wslast_20_ratios)
    print("length is", len(steps))
    print(f"最后 20 个成功率的平均值: {average_fwsuccess_ratio:.4f}")
    print(f"最后 20 个成功率的平均值: {average_masuccess_ratio:.4f}")
    print(f"最后 20 个成功率的平均值: {average_wssuccess_ratio:.4f}")
    if i == 0:
        ARS[0] =average_masuccess_ratio * 100
        ARS[1]= average_fwsuccess_ratio * 100
        ARS[2] = average_wssuccess_ratio * 100
    if i == 1:
        DQN[0] =average_masuccess_ratio * 100
        DQN[1]= average_fwsuccess_ratio * 100
        DQN[2] = average_wssuccess_ratio * 100
    if i == 2:
        A2C[0] =average_masuccess_ratio * 100
        A2C[1]= average_fwsuccess_ratio * 100
        A2C[2] = average_wssuccess_ratio * 100
    if i == 3:
        PPO[0] =average_masuccess_ratio * 100
        PPO[1]= average_fwsuccess_ratio * 100
        PPO[2] = average_wssuccess_ratio * 100
# 数据


x = np.arange(len(categories))  # 横坐标的位置
bar_width = 0.2  # 每个条形的宽度

# 绘制条形图
plt.figure(figsize=(12, 6))
plt.bar(x - 0.3, ARS, bar_width, label='ARS', color='teal')  # Regular 条形
plt.bar(x - bar_width/2, DQN, bar_width, label='DQN', color='orange')  # Optimal 条形
plt.bar(x + bar_width/2, A2C, bar_width, label='A2C', color='green')
plt.bar(x + 0.3, PPO, bar_width, label='PPO', color='red')

# 添加数值标注
for i in range(len(ARS)):
    plt.text(x[i] - 0.3, ARS[i] + 0.001, f"{ARS[i]:.2f}", ha='center', va='bottom')
    plt.text(x[i] - bar_width/2, DQN[i] + 0.01, f"{DQN[i]:.2f}", ha='center', va='bottom')
    plt.text(x[i] + bar_width/2, A2C[i] + 0.01, f"{A2C[i]:.2f}", ha='center', va='bottom')
    plt.text(x[i] + 0.3, PPO[i] + 0.001, f"{PPO[i]:.2f}", ha='center', va='bottom')

# 添加标题和轴标签
plt.ylabel('Attack Success Rate on Per Node(%)')
plt.xticks(x, categories)  # 设置横轴标签

# 添加图例
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
