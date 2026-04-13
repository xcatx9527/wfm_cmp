import matplotlib.pyplot as plt
import numpy as np
from tm_data_types import read_file

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载 .wfm 文件（替换为实际路径）
file_path = r'0406-w_012.wfm'  # 您的 .wfm 文件路径
# file_path = r'./n/cpu-e-50ms_001.wfm'  # 您的 .wfm 文件路径
waveform = read_file(file_path)  # 返回 AnalogWaveform 对象

# 提取原始数据
raw_time = waveform.normalized_horizontal_values  # 库计算的时间轴 (s)
y_data = waveform.normalized_vertical_values  # 电压数据 (V)

# 从库提取采样率信息（如果可用）
# tm_data_types 通常从元数据计算 horizontal_sample_interval (s/point)
if hasattr(waveform, 'horizontal_sample_interval'):
    library_sr = 1 / waveform.horizontal_sample_interval  # 库采样率 (Hz)
    print(f"库提取采样率：{library_sr:.2e} Hz")
else:
    # 备选估算
    library_sr = 1 / (raw_time[1] - raw_time[0]) if len(raw_time) > 1 else 1
    print(f"库估算采样率：{library_sr:.2e} Hz")


print(f"时间范围：{raw_time[0]:.6f} s 到 {raw_time[-1]:.6f} s，总时长：{raw_time[-1] - raw_time[0]:.6f} s")
print(f"采样点数：{len(raw_time)}，手动采样率：{library_sr:.2e} Hz")
print(f"波形幅度范围：{np.min(y_data):.3f} V 到 {np.max(y_data):.3f} V")
print(f"时间单位：{waveform.x_axis_units}，电压单位：{waveform.y_axis_units}")


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_tek_waveform(time, data, waveform):
    """
    绘制泰克示波器波形，默认自带精细刻度
    修复：多次调用不会乱刻度
    """
    # 关键：每次都创建全新画布，不继承上一次的设置
    plt.close('all')  # 关闭之前所有图

    fig, ax = plt.subplots(figsize=(80, 6), dpi=100)  # 用全新ax
    ax.plot(time, data)

    ax.set_title('Tektronix 示波器波形 (.wfm，使用 tm_data_types，手动采样率)')
    ax.set_xlabel(f'时间 ({waveform.x_axis_units})')
    ax.set_ylabel(f'幅度 ({waveform.y_axis_units})')

    # 固定精细刻度（只对当前ax生效，不会污染下一次）
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.001))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    ax.grid(True, which='both', linestyle='--', alpha=0.6)



    plt.tight_layout()
    plt.show()

plot_tek_waveform(raw_time, y_data,waveform)


