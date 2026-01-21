def mean_std(scale, target_mode):
    if scale==2:                    #  x2 超分，使用该任务统计得到的 mean/std。
        mean = (0.4485, 0.4375, 0.4045)
        std = (0.2397, 0.2290, 0.2389)
    elif scale==3:                   #  x3 超分，使用对应统计值。
        mean = (0.4485, 0.4375, 0.4045)
        std = (0.2373, 0.2265, 0.2367)
    elif scale==4:                    # x4 超分，使用对应统计值。
        mean = (0.4485, 0.4375, 0.4045)
        std = (0.2352, 0.2244, 0.2349)
    elif target_mode=='light_dn': # image normalization with statistics from HQ sets  轻量去噪任务，用 HQ 数据集统计值。
        mean = (0.4775, 0.4515, 0.4047)
        std = (0.2442, 0.2367, 0.2457)
    elif target_mode=='light_realdn': # image normalization with statistics from HQ sets  真实去噪模式，设为均值 0、标准差 1（等同不归一化）。因为“归一化”通常指把输入变成 (x - mean) / std。当 mean=0、std=1 时，这个变换就是 (x - 0) / 1 = x，输出与输入完全一样，相当于什么都没做，所以说“不归一化”。
        mean = (0.0000, 0.0000, 0.0000)
        std = (1.0000, 1.0000, 1.0000)
    elif target_mode=='light_graydn': # image normalization with statistics from HQ sets  灰度去噪，只有一个通道的均值/标准差。
        mean = (0.4539,)
        std = (0.2326,)
    elif target_mode=='light_lle':    # 低光增强任务，用对应统计值。
        mean = (0.1687, 0.1599, 0.1526)
        std = (0.1142, 0.1094, 0.1094)
    elif target_mode=='light_dr':     # 去雨任务，用对应统计值。
        mean = (0.5110, 0.5105, 0.4877)
        std = (0.2313, 0.2317, 0.2397)
    return mean, std        #返回最终选择的均值/标准差元组。因为在 Python 里写成 (0.4485, 0.4375, 0.4045) 这种“括号 + 逗号”的形式就是元组（tuple）。元组是有序且不可变的序列，用来一次性保存多个值。这里用元组保存各通道的 mean/std（RGB 三个值或灰度一个值），便于统一返回。