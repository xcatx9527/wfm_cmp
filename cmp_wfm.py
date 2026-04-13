import matplotlib.pyplot as plt
import numpy as np
from tm_data_types import read_file
from scipy.interpolate import interp1d
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def safe_extract_waveform(waveform, time_offset_manual=0.0, time_start=None, time_end=None):
    """
    Safely extract single waveform data (added manual time offset parameter and time range filtering).
    :param waveform: Loaded waveform object
    :param time_offset_manual: Manually specified time offset (s) for aligning waveform start position
    :param time_start: Start time of waveform (s) - filter data greater than or equal to this time
    :param time_end: End time of waveform (s) - filter data less than or equal to this time
    :return: Calibrated time axis with offset, amplitude data (time range filtered)
    """
    if hasattr(waveform, 'waveforms'):
        waveform = list(waveform.waveforms)[0]  # Take first channel
    if not hasattr(waveform, 'normalized_horizontal_values'):
        raise ValueError("Invalid waveform object.")

    raw_time = np.asarray(waveform.normalized_horizontal_values).flatten()
    y_data = np.asarray(waveform.normalized_vertical_values).flatten()

    if len(raw_time) != len(y_data):
        min_len = min(len(raw_time), len(y_data))
        raw_time = raw_time[:min_len]
        y_data = y_data[:min_len]
        print(f"Warning: Length mismatch, truncated to {min_len} points.")

    if len(raw_time) == 0:
        raise ValueError("Waveform data is empty.")

    # Manually set sampling rate (Hz) to override library values for oscilloscope display matching
    # manual_sr = 6.25e6  # Example: 125 MS/s, adjust according to oscilloscope settings
    # manual_sr = 1.25e8  # Example: 125 MS/s, adjust according to oscilloscope settings
    manual_sr = 1 / (raw_time[1] - raw_time[0]) if len(raw_time) > 1 else 1

    # Regenerate time axis: base time axis + manual offset (fix start position misalignment)
    time_offset_base = raw_time[0]  # Original start offset
    time_base = time_offset_base + np.arange(len(y_data)) * (1 / manual_sr)
    time = time_base + time_offset_manual  # Apply manual offset

    # Time range filtering
    if time_start is not None or time_end is not None:
        # Set default ranges
        start_mask = time >= time_start if time_start is not None else np.ones_like(time, dtype=bool)
        end_mask = time <= time_end if time_end is not None else np.ones_like(time, dtype=bool)
        mask = start_mask & end_mask

        if not np.any(mask):
            raise ValueError(f"No valid data in specified time range [{time_start if time_start else '∞'}, {time_end if time_end else '∞'}] s")

        time = time[mask]
        y_data = y_data[mask]
        print(f"After time range filtering: Number of data points {len(time)}, time range {time[0]:.6f} ~ {time[-1]:.6f} s")

    # Print sampling rate and offset information
    library_sr = 1 / (raw_time[1] - raw_time[0]) if len(raw_time) > 1 else 1
    print(f"Library estimated sampling rate: {library_sr:.2e} Hz, Manual sampling rate: {manual_sr:.2e} Hz")
    print(f"Original start time: {time_offset_base:.6f} s, Manual offset: {time_offset_manual:.6f} s, Final start time: {time[0]:.6f} s")

    return time, y_data


def align_and_interpolate(time1, y1, time2, y2, target_sr=None):
    """
    Align two waveforms and interpolate to common sampling rate (preserve original logic).
    """
    sr1 = 1 / (time1[1] - time1[0]) if len(time1) > 1 else 1
    sr2 = 1 / (time2[1] - time2[0]) if len(time2) > 1 else 1
    if target_sr is None:
        target_sr = max(sr1, sr2)

    common_start = max(time1[0], time2[0])
    common_end = min(time1[-1], time2[-1])
    if common_start >= common_end:
        raise ValueError("No overlap in waveform time ranges, cannot align.")

    target_time = np.arange(common_start, common_end, 1 / target_sr)

    interp1_func = interp1d(time1, y1, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp2_func = interp1d(time2, y2, kind='linear', bounds_error=False, fill_value='extrapolate')

    y1_interp = interp1_func(target_time)
    y2_interp = interp2_func(target_time)

    return target_time, y1_interp, y2_interp


def calculate_sum_diff(y_seg1, y_seg2, amplitude_threshold=0.0):
    """
    Optimized algorithm: After filtering data points with absolute values below threshold,
    calculate sum of positive/negative amplitudes (absolute values) in window, return larger difference
    :param y_seg1: Waveform 1 window data
    :param y_seg2: Waveform 2 window data
    :param amplitude_threshold: Amplitude threshold - points with absolute values below this are excluded
    :return: Larger value of the difference between positive/negative amplitude sums after filtering
    """
    if len(y_seg1) < 2 or len(y_seg2) < 2:
        return 0.0

    y1 = np.asarray(y_seg1)
    y2 = np.asarray(y_seg2)

    # Filter data points with absolute values below threshold (core optimization)
    y1_filtered = y1[np.abs(y1) >= amplitude_threshold]
    y2_filtered = y2[np.abs(y2) >= amplitude_threshold]

    # Return 0 if no data after filtering
    if len(y1_filtered) == 0 or len(y2_filtered) == 0:
        return 0.0

    # Calculate sum of positive values and sum of absolute negative values for filtered waveform 1
    y1_pos = y1_filtered[y1_filtered > 0]
    y1_pos_sum = np.sum(y1_pos) if len(y1_pos) > 0 else 0.0

    y1_neg = y1_filtered[y1_filtered < 0]
    y1_neg_abs_sum = np.sum(np.abs(y1_neg)) if len(y1_neg) > 0 else 0.0

    # Calculate sum of positive values and sum of absolute negative values for filtered waveform 2
    y2_pos = y2_filtered[y2_filtered > 0]
    y2_pos_sum = np.sum(y2_pos) if len(y2_pos) > 0 else 0.0

    y2_neg = y2_filtered[y2_filtered < 0]
    y2_neg_abs_sum = np.sum(np.abs(y2_neg)) if len(y2_neg) > 0 else 0.0

    y1_pos_sum += 1
    y1_neg_abs_sum += 1
    y2_pos_sum += 1
    y2_neg_abs_sum += 1

    # Calculate absolute difference of positive sums + absolute difference of negative absolute sums, return larger value
    if y1_pos_sum > y2_pos_sum:
        pos_diff = abs(y1_pos_sum / y2_pos_sum)
    else:
        pos_diff = abs(y2_pos_sum / y1_pos_sum)

    if y1_neg_abs_sum > y2_neg_abs_sum:
        neg_diff = abs(y1_neg_abs_sum / y2_neg_abs_sum)
    else:
        neg_diff = abs(y2_neg_abs_sum / y1_neg_abs_sum)

    return max(pos_diff, neg_diff)


def compare_waveforms_sliding(
        file_path1, file_path2,
        w=0.001, t=0.0001, a=0.8,
        offset1=0.0, offset2=0.0,
        amplitude_threshold=0.0,
        time1_start=None, time1_end=None,  # Waveform 1 time range
        time2_start=None, time2_end=None   # Waveform 2 time range
):
    """
    Sliding window comparison algorithm (added amplitude threshold filtering and specified time segment comparison).
    :param file_path1: Waveform 1 path
    :param file_path2: Waveform 2 path
    :param w: Window width (s)
    :param t: Step size (s)
    :param a: Difference threshold (V·points)
    :param offset1: Manual time offset for waveform 1 (s)
    :param offset2: Manual time offset for waveform 2 (s)
    :param amplitude_threshold: Amplitude threshold (V) - points with absolute values below this are excluded from comparison
    :param time1_start: Start time for waveform 1 (s) - only analyze data after this time
    :param time1_end: End time for waveform 1 (s) - only analyze data before this time
    :param time2_start: Start time for waveform 2 (s) - only analyze data after this time
    :param time2_end: End time for waveform 2 (s) - only analyze data before this time
    :return: List of start times for windows with large differences, window width (for visualization rectangles)
    """
    # Load waveforms (pass manual offset and time range)
    waveform1 = read_file(file_path1)
    waveform2 = read_file(file_path2)

    full_time1, full_y1 = safe_extract_waveform(
        waveform1,
        time_offset_manual=offset1,
        time_start=time1_start,
        time_end=time1_end
    )
    full_time2, full_y2 = safe_extract_waveform(
        waveform2,
        time_offset_manual=offset2,
        time_start=time2_start,
        time_end=time2_end
    )

    # Align and interpolate (based on offset time axes)
    time, y1, y2 = align_and_interpolate(full_time1, full_y1, full_time2, full_y2)

    print(f"\n【Alignment Information】")
    print(f"Waveform 1 after offset + filtering: {full_time1[0]:.6f} ~ {full_time1[-1]:.6f} s")
    print(f"Waveform 2 after offset + filtering: {full_time2[0]:.6f} ~ {full_time2[-1]:.6f} s")
    print(f"Final aligned time range: {time[0]:.6f} s to {time[-1]:.6f} s")
    print(f"Sampling rate: {1 / (time[1] - time[0]):.2e} Hz")
    print(f"Data length: {len(time)}")
    print(f"Amplitude filter threshold: {amplitude_threshold} V (points with absolute values below this are excluded from calculation)")

    # Sliding window parameters
    sr = 1 / (time[1] - time[0])
    window_samples = int(w * sr)  # Number of points per window
    step_samples = int(t * sr)    # Number of points per step

    if window_samples < 10:
        print("Warning: Window too small (<10 points), consider increasing w.")

    large_diff_starts = []  # Start times of windows with large differences
    total_windows = (len(time) - window_samples) // step_samples + 1
    print(f"Total windows: {total_windows}")

    # Sliding window iteration
    alldiff=0
    for i in range(0, len(time) - window_samples + 1, step_samples):
        seg_time_start = time[i]
        y_seg1 = y1[i:i + window_samples]
        y_seg2 = y2[i:i + window_samples]

        # Calculate difference with amplitude threshold
        sum_diff = calculate_sum_diff(y_seg1, y_seg2, amplitude_threshold)
        alldiff+=sum_diff
        if sum_diff > a:
            large_diff_starts.append(seg_time_start)
            print(f"Large difference window {len(large_diff_starts)} start: {seg_time_start:.6f} s, Sum difference: {sum_diff:.3f} V·points")
    print(f"平均差异：{alldiff/(window_samples/step_samples)}===={window_samples}")
    return large_diff_starts, w  # Return window width for drawing rectangles




def analyze_waveforms(params):
    """
    分析并可视化两个波形文件的差异

    参数说明:
    params (dict): 包含所有分析参数的字典，键如下：
        - file_path1: 第一个波形文件路径
        - file_path2: 第二个波形文件路径
        - w: 窗口宽度 (s)
        - t: 步长 (s)
        - a: 差异阈值 (V·points)
        - amplitude_th: 振幅过滤阈值 (V)
        - offset1: 波形1的时间偏移 (s)
        - offset2: 波形2的时间偏移 (s)
        - time1_start: 波形1起始时间 (s)
        - time1_end: 波形1结束时间 (s)
        - time2_start: 波形2起始时间 (s)
        - time2_end: 波形2结束时间 (s)
    """
    try:
        # 提取参数
        file_path1 = params['file_path1']
        file_path2 = params['file_path2']
        w = params['w']
        t = params['t']
        a = params['a']
        amplitude_th = params['amplitude_th']
        offset1 = params['offset1']
        offset2 = params['offset2']
        time1_start = params['time1_start']
        time1_end = params['time1_end']
        time2_start = params['time2_start']
        time2_end = params['time2_end']

        # 对比波形，获取差异起始时间和窗口宽度
        starts, window_width = compare_waveforms_sliding(
            file_path1=file_path1,
            file_path2=file_path2,
            w=w,
            t=t,
            a=a,
            offset1=offset1,
            offset2=offset2,
            amplitude_threshold=amplitude_th,
            time1_start=time1_start,
            time1_end=time1_end,
            time2_start=time2_start,
            time2_end=time2_end
        )

        # 打印结果
        print(
            f"\n===== 分析结果 ({file_path1} vs {file_path2}) ====="
            f"\n振幅过滤阈值={amplitude_th}V, 振幅和差异绝对值 > {a} V·points 的窗口起始时间 (共 {len(starts)} 个):")
        for i, start in enumerate(starts, 1):
            print(f"  {i}: {start:.6f} s")

        # 读取波形文件用于可视化
        waveform1 = read_file(file_path1)
        waveform2 = read_file(file_path2)

        # 提取带时间范围的波形数据
        full_time1, full_y1 = safe_extract_waveform(
            waveform1,
            time_offset_manual=offset1,
            time_start=time1_start,
            time_end=time1_end
        )
        full_time2, full_y2 = safe_extract_waveform(
            waveform2,
            time_offset_manual=offset2,
            time_start=time2_start,
            time_end=time2_end
        )

        # 绘制可视化图表
        plt.figure(figsize=(20, 6),dpi=100)
        # 绘制波形曲线
        plt.plot(full_time1, full_y1,
                 label=f'Waveform 1 (Offset: {offset1:.6f}s, Time Range: {time1_start}~{time1_end}s)',
                 alpha=0.9)
        plt.plot(full_time2, full_y2,
                 label=f'Waveform 2 (Offset: {offset2:.6f}s, Time Range: {time2_start}~{time2_end}s)',
                 alpha=0.7)

        # 绘制振幅阈值线
        plt.axhline(y=amplitude_th, color='green', linestyle='--', alpha=0.5,
                    label=f'Amplitude Threshold +{amplitude_th}V')
        plt.axhline(y=-amplitude_th, color='green', linestyle='--', alpha=0.5,
                    label=f'Amplitude Threshold -{amplitude_th}V')

        # 计算y轴范围（适配矩形绘制）
        all_y = np.concatenate([full_y1, full_y2])
        y_min = np.min(all_y) - 0.1 * np.ptp(all_y)
        y_max = np.max(all_y) + 0.1 * np.ptp(all_y)

        # 绘制差异窗口矩形
        for idx, start in enumerate(starts):
            end = start + window_width
            plt.axvspan(xmin=start, xmax=end, ymin=0, ymax=1,
                        transform=plt.gca().get_xaxis_transform(),
                        color='red', alpha=0.2,
                        label='Large Difference Window' if idx == 0 else "")

        # 调整刻度样式
        ax = plt.gca()
        x_major_ticks = ax.get_xticks()
        y_major_ticks = ax.get_yticks()
        x_major_interval = x_major_ticks[1] - x_major_ticks[0] if len(x_major_ticks) > 1 else (full_time1[-1] - full_time1[0]) / 10
        y_major_interval = y_major_ticks[1] - y_major_ticks[0] if len(y_major_ticks) > 1 else (y_max - y_min) / 10

        # 设置次要刻度（每主刻度10个次刻度）
        ax.xaxis.set_minor_locator(plt.MultipleLocator(x_major_interval / 10))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(y_major_interval / 10))

        # 调整刻度长度
        ax.tick_params(which='minor', axis='x', length=8, color='#888888')
        ax.tick_params(which='minor', axis='y', length=8, color='#888888')
        ax.tick_params(which='major', axis='x', length=12, color='black')
        ax.tick_params(which='major', axis='y', length=12, color='black')

        # 网格设置
        ax.grid(True, which='major', linestyle='-', linewidth=1.0, color='#CCCCCC')
        ax.grid(True, which='minor', linestyle='-', linewidth=0.4, color='#EEEEEE')

        # 图表标题和标签
        plt.title(
            f'Waveform Comparison: {file_path1} vs {file_path2}\n'
            f'Window: {w}s, Step: {t}s, Threshold: {a} V·points, Amplitude Filter: {amplitude_th}V\n'
            f'Waveform 1 Time Range: {time1_start}~{time1_end}s | Waveform 2 Time Range: {time2_start}~{time2_end}s'
        )
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (V)')
        plt.ylim(y_min, y_max)
        plt.legend(loc='upper right')
        plt.show()

        return starts, window_width  # 返回分析结果供后续使用

    except Exception as e:
        print(f"\n分析出错: {e}")
        return None, None

# 使用示例
if __name__ == "__main__":
    # 定义d0波形的参数
    d0_params = {
        'file_path1': r'Tek000-d0.wfm',
        'file_path2': r'Tek000-errd0.wfm',
        'w': 0.01,          # 窗口宽度 10ms
        't': 0.01,          # 步长 10ms
        'a': 500,           # 差异阈值 (V·points)
        'amplitude_th': 0.01,  # 振幅过滤阈值 (V)
        'offset1': 0,       # 波形1时间偏移
        'offset2': 0.02,    # 波形2时间偏移
        'time1_start': 0.2, # 波形1起始时间
        'time1_end': 0.8,   # 波形1结束时间
        'time2_start': 0,   # 波形2起始时间
        'time2_end': 0.8    # 波形2结束时间
    }

    # 定义cpu波形的参数
    cpu_params = {
        'file_path1': r'Tek000-cpu.wfm',
        'file_path2': r'Tek000-cpuerr.wfm',
        'w': 0.01,          # 窗口宽度 10ms
        't': 0.01,          # 步长 10ms
        'a': 5,             # 差异阈值 (V·points)
        'amplitude_th': 0.01,  # 振幅过滤阈值 (V)
        'offset1': 0,       # 波形1时间偏移
        'offset2': 0.028,   # 波形2时间偏移
        'time1_start': 0.2, # 波形1起始时间
        'time1_end': 0.8,   # 波形1结束时间
        'time2_start': 0,   # 波形2起始时间
        'time2_end': 0.8    # 波形2结束时间
    }

    # 分析d0波形
    print("===== 开始分析d0波形 =====")
    d0_starts, d0_window = analyze_waveforms(d0_params)

    # 分析cpu波形
    print("\n===== 开始分析cpu波形 =====")
    cpu_starts, cpu_window = analyze_waveforms(cpu_params)


# 使用示例
# if __name__ == "__main__":
#     # 定义d0波形的参数
#     # cpu_params = {
#     #     'file_path1': r'./n/cpu-e-50ms_001.wfm',
#     #     'file_path2': r'./n/cpu-n-50ms_000.wfm',
#     #     'w': 0.001,          # 窗口宽度 10ms
#     #     't': 0.001,          # 步长 10ms
#     #     'a': 1.45,           # 差异阈值 (V·points)
#     #     'amplitude_th': 0.01,  # 振幅过滤阈值 (V)
#     #     'offset1':0,       # 波形1时间偏移
#     #     'offset2': 0,    # 波形2时间偏移
#     #     'time1_start': 0.11, # 波形1起始时间
#     #     'time1_end': 0.40,   # 波形1结束时间
#     #     'time2_start': 0.11,   # 波形2起始时间
#     #     'time2_end': 0.40    # 波形2结束时间
#     # }
#     cpu_params = {
#         'file_path1': r'./0406-w_011.wfm',
#         'file_path2': r'./0406-w_012.wfm',
#         'w': 0.001,          # 窗口宽度 10ms
#         't': 0.001,          # 步长 10ms
#         'a': 50,           # 差异阈值 (V·points)
#         'amplitude_th': 0.01,  # 振幅过滤阈值 (V)
#         'offset1':0.0067,       # 波形1时间偏移
#         'offset2': -0.0097,    # 波形2时间偏移
#         'time1_start': 0, # 波形1起始时间
#         'time1_end': 0.09,   # 波形1结束时间
#         'time2_start': 0,   # 波形2起始时间
#         'time2_end': 0.09    # 波形2结束时间
#     }
#
#     # 分析cpu波形
#     print("\n===== 开始分析cpu波形 =====")
#     cpu_starts, cpu_window = analyze_waveforms(cpu_params)