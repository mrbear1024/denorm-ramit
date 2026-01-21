import pickle
import matplotlib.pyplot as plt


def plot_proj():  # 1/4 sample
    with open('../proj_1_4/raw_file/Ref_217.pkl', 'rb') as f:
        Ref_proj = pickle.load(f)
    with open('../proj_1_4/raw_file/Linear_interpolation_217.pkl', 'rb') as f:
        Li_proj = pickle.load(f)
    with open('../proj_1_4/raw_file/SynCNN_217.pkl', 'rb') as f:
        SynCNN_proj = pickle.load(f)
    with open('../proj_1_4/raw_file/SynNet3D_217.pkl', 'rb') as f:
        SynNet3D_proj = pickle.load(f)

    vmin = Ref_proj.min() + (Ref_proj.max() - Ref_proj.min()) * 0.0
    vmax = Ref_proj.min() + (Ref_proj.max() - Ref_proj.min()) * 0.9
    print('vmin=%.15f' % vmin)
    print('vmax=%.15f' % vmax)

    # plot -----I reference-----
    # frame
    plt.figure(figsize=(6, 768 * 6 / 1024))  # w, h
    plt.imshow(Ref_proj, 'gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../proj_1_4/plot/0_Ref_217.png', dpi=300)
    plt.close()
    # ----------

    # plot -----II linear interpolation-----
    # frame
    plt.figure(figsize=(6, 768 * 6 / 1024))
    plt.imshow(Li_proj, 'gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../proj_1_4/plot/1_Linear_interpolation_217.png', dpi=300)
    plt.close()
    ## frame_err
    abs_error = abs(Li_proj - Ref_proj)
    vmin_er = abs_error.min() + (abs_error.max() - abs_error.min()) * 0.0
    vmax_er = abs_error.min() + (abs_error.max() - abs_error.min()) * 0.3
    print('vmin_er=%.15f' % vmin_er)
    print('vmax_er=%.15f' % vmax_er)
    plt.figure(figsize=(6, 768 * 6 / 1024))
    plt.imshow(abs_error, vmin=vmin_er, vmax=vmax_er)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../proj_1_4/plot/1_Linear_interpolation_217_er.png', dpi=300)
    plt.close()
    # ----------

    # plot -----III SynCNN-----
    ## frame
    plt.figure(figsize=(6, 768 * 6 / 1024))
    plt.imshow(SynCNN_proj, 'gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../proj_1_4/plot/2_SynCNN_217.png', dpi=300)
    plt.close()
    ## frame_err
    abs_error = abs(SynCNN_proj - Ref_proj)
    plt.figure(figsize=(6, 768 * 6 / 1024))
    plt.imshow(abs_error, vmin=vmin_er, vmax=vmax_er)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../proj_1_4/plot/2_SynCNN_217_er.png', dpi=300)
    plt.close()
    # ----------

    # plot -----IV SynNet3D-----
    ## frame
    plt.figure(figsize=(6, 768 * 6 / 1024))
    plt.imshow(SynNet3D_proj, 'gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../proj_1_4/plot/3_SynNet3D_217.png', dpi=300)
    plt.close()
    ## frame_err
    abs_error = abs(SynNet3D_proj - Ref_proj)
    plt.figure(figsize=(6, 768 * 6 / 1024))
    plt.imshow(abs_error, vmin=vmin_er, vmax=vmax_er)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../proj_1_4/plot/3_SynNet3D_217_er.png', dpi=300)
    plt.close()
    # ----------

    return None


"""
vmin=0.049337603151798
vmax=4.299103809893132
vmin_er=0.000000000000000
vmax_er=0.139432168006897
"""

if __name__ == '__main__':
    plot_proj()
