import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '18'


# 1.4 projection  2021-03-26_094858_FINISHED_Head
def projection():
    ## RMSE
    x = np.array([0.1, 0.7, 1.3])
    RMSE = np.array([0.0431370096, 0.0190798810, 0.0171955137])
    RMSE = np.round(RMSE * 10 ** 2, 1)
    COLOR = ['#a5cde2', 'plum', 'tomato']
    LABEL = ['Linear interpolation', 'SynCNN', 'SynNet3D']
    fig, ax = plt.subplots(figsize=(6, 768 * 6 / 1024))
    bar = ax.bar(x, RMSE, width=0.6, color=COLOR, label=LABEL)

    # 限定y轴范围；y轴坐标展示；限定x轴范围；隐藏x轴显示
    plt.ylim(1, 7)
    plt.yticks(np.arange(1, 6, 2))
    plt.xlim(-0.2, 1.6)
    ax.axes.xaxis.set_ticks([])

    # 在柱上方写对应数值
    ax.bar_label(bar, padding=1, fontsize=18)

    ylabel=plt.ylabel('RMSE ($×10^{-2}$)', fontsize=20)
    #ylabel.set_position((-1, 0.5))
    ax.yaxis.set_label_coords(-0.12, 0.5)
    plt.legend(ncol=1, fontsize=17, handletextpad=0.3, handleheight=1, labelspacing=0.000001,
               columnspacing=0.5, framealpha=0, loc=(0.01, 0.68))   #loc=(0.285, 0.67)
    # plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.03, left=0.16, right=0.99)
    # plt.show()
    plt.savefig('../proj_1_4/barchart/0_RMSE.png', format='png', dpi=300)
    plt.close()

    ## PSNR
    x = np.array([0.1, 0.7, 1.3])
    PSNR = np.array([40.8757334896, 47.9612209648, 48.8644310971])
    PSNR = np.round(PSNR, 1)
    COLOR = ['#a5cde2', 'plum', 'tomato']
    LABEL = ['Linear interpolation', 'SynCNN', 'SynNet3D']
    fig, ax = plt.subplots(figsize=(6, 768 * 6 / 1024))
    bar = ax.bar(x, PSNR, width=0.6, color=COLOR, label=LABEL)

    # 限定y轴范围；y轴坐标展示；限定x轴范围；隐藏x轴显示
    plt.ylim(41, 55)
    plt.yticks(np.arange(39, 49.5, 5))
    plt.xlim(-0.2, 1.6)
    ax.axes.xaxis.set_ticks([])

    # 在柱上方写对应数值
    ax.bar_label(bar, padding=1, fontsize=18)

    plt.ylabel('PSNR (dB)', fontsize=20)
    plt.legend(ncol=1, fontsize=17, handletextpad=0.3, handleheight=1, labelspacing=0.000001,
               columnspacing=0.5, framealpha=0, loc=(0.01, 0.68))
    # plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.03, left=0.16, right=0.99)
    # plt.show()
    plt.savefig('../proj_1_4/barchart/1_PSNR.png', format='png', dpi=300)
    plt.close()

    ## SSIM
    x = np.array([0.1, 0.7, 1.3])
    SSIM = np.array([0.9673914234, 0.9927323234, 0.9936953067])
    SSIM = np.round(SSIM * 10 ** 3, 1)
    COLOR = ['#a5cde2', 'plum', 'tomato']
    LABEL = ['Linear interpolation', 'SynCNN', 'SynNet3D']
    fig, ax = plt.subplots(figsize=(6, 768 * 6 / 1024))
    bar = ax.bar(x, SSIM, width=0.6, color=COLOR, label=LABEL)

    # 限定y轴范围；y轴坐标展示；限定x轴范围；隐藏x轴显示
    plt.ylim(950, 1020)
    plt.yticks(np.arange(950, 1010, 20))
    plt.xlim(-0.2, 1.6)
    ax.axes.xaxis.set_ticks([])

    # 在柱上方写对应数值
    ax.bar_label(bar, padding=1, fontsize=18)

    plt.ylabel('SSIM ($×10^{-3}$)', fontsize=20)
    plt.legend(ncol=1, fontsize=17, handletextpad=0.3, handleheight=1, labelspacing=0.000001,
               columnspacing=0.5, framealpha=0, loc=(0.01, 0.68))
    # plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.03, left=0.16, right=0.99)
    # plt.show()
    plt.savefig('../proj_1_4/barchart/2_SSIM.png', format='png', dpi=300)
    plt.close()

    ## FSIM
    x = np.array([0.1, 0.7, 1.3])
    FSIM = np.array([0.9162457862, 0.9874770478, 0.9909157586])
    FSIM = np.round(FSIM * 10 ** 2, 1)
    COLOR = ['#a5cde2', 'plum', 'tomato']
    LABEL = ['Linear interpolation', 'SynCNN', 'SynNet3D']
    fig, ax = plt.subplots(figsize=(6, 768 * 6 / 1024))
    bar = ax.bar(x, FSIM, width=0.6, color=COLOR, label=LABEL)

    # 限定y轴范围；y轴坐标展示；限定x轴范围；隐藏x轴显示
    plt.ylim(90, 105)
    plt.yticks(np.arange(90, 101, 4))
    plt.xlim(-0.2, 1.6)
    ax.axes.xaxis.set_ticks([])

    # 在柱上方写对应数值
    ax.bar_label(bar, padding=1, fontsize=18)

    plt.ylabel('FSIM ($×10^{-2}$)', fontsize=20)
    plt.legend(ncol=1, fontsize=17, handletextpad=0.3, handleheight=1, labelspacing=0.000001,
               columnspacing=0.5, framealpha=0, loc=(0.01, 0.68))
    # plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.03, left=0.16, right=0.99)
    # plt.show()
    plt.savefig('../proj_1_4/barchart/3_FSIM.png', format='png', dpi=300)
    plt.close()


if __name__ == "__main__":
    projection()
