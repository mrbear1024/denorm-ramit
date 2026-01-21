import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '21'


def image_transverse():
    ## RMSE
    #x = np.array([0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6])
    x = np.array([0.3, 1.3, 2.3, 3.3])
    RMSE = np.array([0.0011161603, 0.0007273301, 0.0005774822, 0.0005217358])
    RMSE = np.round(RMSE * 10 ** 4, 1)
    COLOR = ['limegreen', 'royalblue', 'blueviolet', 'darkorange']
    LABEL = ['SVCT', 'Linear interpolation', 'SynCNN', 'SynNet3D']
    fig, ax = plt.subplots(figsize=(6, 6))
    bar = ax.bar(x, RMSE, width=1, color=COLOR, label=LABEL)

    # 限定y轴范围；y轴坐标展示；限定x轴范围；隐藏x轴显示
    plt.ylim(4, 15)
    plt.yticks(np.arange(4, 15, 4))
    plt.xlim(-0.2, 3.8)
    ax.axes.xaxis.set_ticks([])

    # 在柱上方写对应数值
    ax.bar_label(bar, padding=1, fontsize=19)

    plt.ylabel('RMSE ($×10^{-4}$)', fontsize=20, labelpad=0)
    plt.legend(ncol=2, fontsize=16.5, handletextpad=0.3, handleheight=1, labelspacing=0.000001,
               columnspacing=0.5, framealpha=0, loc=(0.05, 0.795))
    # plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.022, left=0.14, right=0.99)
    # plt.show()
    plt.savefig('../img_1_4/barchart/0_RMSE.png', format='png', dpi=300)
    plt.close()

    ## PSNR
    x = np.array([0.3, 1.3, 2.3, 3.3])
    PSNR = np.array([34.1834971622, 37.9033972438, 39.9072563563, 40.7890166810])
    PSNR = np.round(PSNR, 1)

    COLOR = ['limegreen', 'royalblue', 'blueviolet', 'darkorange']
    LABEL = ['SVCT', 'Linear interpolation', 'SynCNN', 'SynNet3D']
    fig, ax = plt.subplots(figsize=(6, 6))
    bar = ax.bar(x, PSNR, width=1, color=COLOR, label=LABEL)

    # 限定y轴范围；y轴坐标展示；限定x轴范围；隐藏x轴显示
    plt.ylim(32, 45)
    plt.yticks(np.arange(32, 43, 4))
    plt.xlim(-0.2, 3.8)
    ax.axes.xaxis.set_ticks([])

    # 在柱上方写对应数值
    ax.bar_label(bar, padding=1, fontsize=19)

    plt.ylabel('PSNR (dB)', fontsize=20, labelpad=0)
    plt.legend(ncol=2, fontsize=16.5, handletextpad=0.3, handleheight=1, labelspacing=0.000001,
               columnspacing=0.5, framealpha=0, loc=(0.05, 0.795))
    # plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.025, left=0.14, right=0.99)
    # plt.show()
    plt.savefig('../img_1_4/barchart/1_PSNR.png', format='png', dpi=300)
    plt.close()

    ## SSIM
    x = np.array([0.3, 1.3, 2.3, 3.3])
    SSIM = np.array(
        [0.8191148362, 0.9417426275, 0.9644851001, 0.9690143493])
    SSIM = np.round(SSIM * 10 ** 2, 1)
    COLOR = ['limegreen', 'royalblue', 'blueviolet', 'darkorange']
    LABEL = ['SVCT', 'Linear interpolation', 'SynCNN', 'SynNet3D']
    fig, ax = plt.subplots(figsize=(6, 6))
    bar = ax.bar(x, SSIM, width=1, color=COLOR, label=LABEL)

    # 限定y轴范围；y轴坐标展示；限定x轴范围；隐藏x轴显示
    plt.ylim(75, 110)
    plt.yticks(np.arange(75, 100, 10))
    plt.xlim(-0.2, 3.8)
    ax.axes.xaxis.set_ticks([])

    # 在柱上方写对应数值
    ax.bar_label(bar, padding=1, fontsize=19)

    plt.ylabel('SSIM ($×10^{-2}$)', fontsize=20, labelpad=0)
    plt.legend(ncol=2, fontsize=16.5, handletextpad=0.3, handleheight=1, labelspacing=0.000001,
               columnspacing=0.5, framealpha=0, loc=(0.04, 0.795))
    # plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.022, left=0.15, right=0.99)
    # plt.show()
    plt.savefig('../img_1_4/barchart/2_SSIM.png', format='png', dpi=300)
    plt.close()

    ## FSIM
    x = np.array([0.3, 1.3, 2.3, 3.3])
    FSIM = np.array([0.9691038736, 0.9870710819, 0.9930042731, 0.9942396926])  # Fabricate Proposed result
    FSIM = np.round(FSIM * 10 ** 2, 1)
    COLOR = ['limegreen', 'royalblue', 'blueviolet', 'darkorange']
    LABEL = ['SVCT', 'Linear interpolation', 'SynCNN', 'SynNet3D']
    fig, ax = plt.subplots(figsize=(6, 6))
    bar = ax.bar(x, FSIM, width=1, color=COLOR, label=LABEL)

    # 限定y轴范围；y轴坐标展示；限定x轴范围；隐藏x轴显示
    plt.ylim(95, 102)
    plt.yticks(np.arange(95, 100, 2))
    plt.xlim(-0.2, 3.8)
    ax.axes.xaxis.set_ticks([])

    # 在柱上方写对应数值
    ax.bar_label(bar, padding=1, fontsize=19)

    plt.ylabel('FSIM ($×10^{-2}$)', fontsize=20, labelpad=0)
    plt.legend(ncol=2, fontsize=16.5, handletextpad=0.3, handleheight=1, labelspacing=0.000001,
               columnspacing=0.5, framealpha=0, loc=(0.04, 0.795))
    # plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.022, left=0.14, right=0.99)
    # plt.show()
    plt.savefig('../img_1_4/barchart/3_FSIM.png', format='png', dpi=300)
    plt.close()


if __name__ == "__main__":
    image_transverse()
