import os
import pickle
import matplotlib.pyplot as plt
import argparse
import matplotlib.patches as patches
import cv2

# FOV_mask: 2D circle
with open('./2D_FOV_mask.pkl', 'rb') as f:
    FOV = pickle.load(f)  # (512,512)


def plot_img():  # 1/4 sample
    with open('../img_1_4/raw_file/Ref_30.pkl', 'rb') as f:
        Ref_img = pickle.load(f)
    with open('../img_1_4/raw_file/SVCT_30.pkl', 'rb') as f:
        SVCT_img = pickle.load(f)
    with open('../img_1_4/raw_file/Linear_interpolation_30.pkl', 'rb') as f:
        Li_img = pickle.load(f)
    with open('../img_1_4/raw_file/SynCNN_30.pkl', 'rb') as f:
        SynCNN_img = pickle.load(f)
    with open('../img_1_4/raw_file/SynNet3D_30.pkl', 'rb') as f:
        SynNet3D_img = pickle.load(f)

    Ref_img *= FOV
    Ref_img[Ref_img < 0] = 0
    SVCT_img *= FOV
    SVCT_img[SVCT_img < 0] = 0
    Li_img *= FOV
    Li_img[Li_img < 0] = 0
    SynCNN_img *= FOV
    SynCNN_img[SynCNN_img < 0] = 0
    SynNet3D_img *= FOV
    SynNet3D_img[SynNet3D_img < 0] = 0

    vmin = Ref_img.min() + (Ref_img.max() - Ref_img.min()) * 0.0
    vmax = Ref_img.min() + (Ref_img.max() - Ref_img.min()) * 0.75
    print('vmin=%.15f' % vmin)
    print('vmax=%.15f' % vmax)

    # -----plot I reference-----
    ## slice
    plt.figure(figsize=(6, 6))
    plt.imshow(Ref_img, 'gray', vmin=vmin, vmax=vmax)
    ## plot ROI Rectangle
    rect = patches.Rectangle((121, 282), 70, 70, linestyle='dashed', linewidth=2, edgecolor='gold', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/0_Ref_30.png', dpi=300)
    plt.close()
    ## slice ROI
    plt.figure(figsize=(2.25, 2.25))
    plt.imshow(Ref_img[282:353, 121:192], 'gray', vmin=vmin, vmax=vmax)
    rect = patches.Rectangle((0, 0), 70, 70, linestyle='dashed', linewidth=2, edgecolor='gold', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/0_Ref_30_ROI.png', dpi=300)
    plt.close()
    ## slice overwrite
    img_slice = cv2.imread('../img_1_4/plot/0_Ref_30.png')
    img_ROI = cv2.imread('../img_1_4/plot/0_Ref_30_ROI.png')
    img_slice[:img_ROI.shape[0], :img_ROI.shape[1]] = img_ROI
    cv2.imwrite('../img_1_4/plot/0_Ref_30_ROI_.png', img_slice)
    # ----------

    # -----plot II SVCT-----
    ## slice
    plt.figure(figsize=(6, 6))
    plt.imshow(SVCT_img, 'gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/1_SVCT_30.png', dpi=300)
    plt.close()
    ## slice ROI
    plt.figure(figsize=(2.25, 2.25))
    plt.imshow(SVCT_img[282:353, 121:192], 'gray', vmin=vmin, vmax=vmax)
    rect = patches.Rectangle((0, 0), 70, 70, linestyle='dashed', linewidth=2, edgecolor='gold', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/1_SVCT_30_ROI.png', dpi=300)
    plt.close()
    ## slice overwrite
    img_slice = cv2.imread('../img_1_4/plot/1_SVCT_30.png')
    img_ROI = cv2.imread('../img_1_4/plot/1_SVCT_30_ROI.png')
    img_slice[:img_ROI.shape[0], :img_ROI.shape[1]] = img_ROI
    cv2.imwrite('../img_1_4/plot/1_SVCT_30_ROI_.png', img_slice)
    ## slice_err
    abs_error = abs(SVCT_img - Ref_img)
    vmin_er = abs_error.max() * 0
    vmax_er = abs_error.max() * 0.35
    print('vmin_er=%.15f' % vmin_er)
    print('vmax_er=%.15f' % vmax_er)
    plt.figure(figsize=(6, 6))
    plt.imshow(abs_error, vmin=vmin_er, vmax=vmax_er)
    ## plot ROI Rectangle
    rect = patches.Rectangle((121, 282), 70, 70, linestyle='dashed', linewidth=2, edgecolor='white', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/1_SVCT_30_er.png', dpi=300)
    plt.close()
    ## slice err ROI
    plt.figure(figsize=(2.25, 2.25))
    plt.imshow(abs_error[282:353, 121:192], vmin=vmin_er, vmax=vmax_er)
    rect = patches.Rectangle((0, 0), 70, 70, linestyle='dashed', linewidth=2, edgecolor='white', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/1_SVCT_30_er_ROI.png', dpi=300)
    plt.close()
    ## slice err overwrite
    img_slice = cv2.imread('../img_1_4/plot/1_SVCT_30_er.png')
    img_ROI = cv2.imread('../img_1_4/plot/1_SVCT_30_er_ROI.png')
    img_slice[:img_ROI.shape[0], :img_ROI.shape[1]] = img_ROI
    cv2.imwrite('../img_1_4/plot/1_SVCT_30_er_ROI_.png', img_slice)
    # ----------


    # plot III Linear interpolation
    ## slice
    plt.figure(figsize=(6, 6))
    plt.imshow(Li_img, 'gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/2_Li_30.png', dpi=300)
    plt.close()
    ## slice ROI
    plt.figure(figsize=(2.25, 2.25))
    plt.imshow(Li_img[282:353, 121:192], 'gray', vmin=vmin, vmax=vmax)
    rect = patches.Rectangle((0, 0), 70, 70, linestyle='dashed', linewidth=2, edgecolor='gold', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/2_Li_30_ROI.png', dpi=300)
    plt.close()
    ## slice overwrite
    img_slice = cv2.imread('../img_1_4/plot/2_Li_30.png')
    img_ROI = cv2.imread('../img_1_4/plot/2_Li_30_ROI.png')
    img_slice[:img_ROI.shape[0], :img_ROI.shape[1]] = img_ROI
    cv2.imwrite('../img_1_4/plot/2_Li_30_ROI_.png', img_slice)
    ## slice_err
    abs_error = abs(Li_img - Ref_img)
    plt.figure(figsize=(6, 6))
    plt.imshow(abs_error, vmin=vmin_er, vmax=vmax_er)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/2_Li_30_er.png', dpi=300)
    plt.close()
    ## slice err ROI
    plt.figure(figsize=(2.25, 2.25))
    plt.imshow(abs_error[282:353, 121:192], vmin=vmin_er, vmax=vmax_er)
    rect = patches.Rectangle((0, 0), 70, 70, linestyle='dashed', linewidth=2, edgecolor='white', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/2_Li_30_er_ROI.png', dpi=300)
    plt.close()
    ## slice err overwrite
    img_slice = cv2.imread('../img_1_4/plot/2_Li_30_er.png')
    img_ROI = cv2.imread('../img_1_4/plot/2_Li_30_er_ROI.png')
    img_slice[:img_ROI.shape[0], :img_ROI.shape[1]] = img_ROI
    cv2.imwrite('../img_1_4/plot/2_Li_30_er_ROI_.png', img_slice)
    # ----------

    # plot IV SynCNN
    ## slice
    plt.figure(figsize=(6, 6))
    plt.imshow(SynCNN_img, 'gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/3_SynCNN_30.png', dpi=300)
    plt.close()
    ## slice ROI
    plt.figure(figsize=(2.25, 2.25))
    plt.imshow(SynCNN_img[282:353, 121:192], 'gray', vmin=vmin, vmax=vmax)
    rect = patches.Rectangle((0, 0), 70, 70, linestyle='dashed', linewidth=2, edgecolor='gold', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/3_SynCNN_30_ROI.png', dpi=300)
    plt.close()
    ## slice overwrite
    img_slice = cv2.imread('../img_1_4/plot/3_SynCNN_30.png')
    img_ROI = cv2.imread('../img_1_4/plot/3_SynCNN_30_ROI.png')
    img_slice[:img_ROI.shape[0], :img_ROI.shape[1]] = img_ROI
    cv2.imwrite('../img_1_4/plot/3_SynCNN_30_ROI_.png', img_slice)
    ## slice_err
    abs_error = abs(SynCNN_img - Ref_img)
    plt.figure(figsize=(6, 6))
    plt.imshow(abs_error, vmin=vmin_er, vmax=vmax_er)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/3_SynCNN_30_er.png', dpi=300)
    plt.close()
    ## slice err ROI
    plt.figure(figsize=(2.25, 2.25))
    plt.imshow(abs_error[282:353, 121:192], vmin=vmin_er, vmax=vmax_er)
    rect = patches.Rectangle((0, 0), 70, 70, linestyle='dashed', linewidth=2, edgecolor='white', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/3_SynCNN_30_er_ROI.png', dpi=300)
    plt.close()
    ## slice err overwrite
    img_slice = cv2.imread('../img_1_4/plot/3_SynCNN_30_er.png')
    img_ROI = cv2.imread('../img_1_4/plot/3_SynCNN_30_er_ROI.png')
    img_slice[:img_ROI.shape[0], :img_ROI.shape[1]] = img_ROI
    cv2.imwrite('../img_1_4/plot/3_SynCNN_30_er_ROI_.png', img_slice)
    # ----------

    # plot V SynNet3D
    ## slice
    plt.figure(figsize=(6, 6))
    plt.imshow(SynNet3D_img, 'gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/4_SynNet3D_30.png', dpi=300)
    plt.close()
    ## slice ROI
    plt.figure(figsize=(2.25, 2.25))
    plt.imshow(SynNet3D_img[282:353, 121:192], 'gray', vmin=vmin, vmax=vmax)
    rect = patches.Rectangle((0, 0), 70, 70, linestyle='dashed', linewidth=2, edgecolor='gold', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/4_SynNet3D_30_ROI.png', dpi=300)
    plt.close()
    ## slice overwrite
    img_slice = cv2.imread('../img_1_4/plot/4_SynNet3D_30.png')
    img_ROI = cv2.imread('../img_1_4/plot/4_SynNet3D_30_ROI.png')
    img_slice[:img_ROI.shape[0], :img_ROI.shape[1]] = img_ROI
    cv2.imwrite('../img_1_4/plot/4_SynNet3D_30_ROI_.png', img_slice)
    ## slice_err
    abs_error = abs(SynNet3D_img - Ref_img)
    plt.figure(figsize=(6, 6))
    plt.imshow(abs_error, vmin=vmin_er, vmax=vmax_er)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/4_SynNet3D_30_er.png', dpi=300)
    plt.close()
    ## slice err ROI
    plt.figure(figsize=(2.25, 2.25))
    plt.imshow(abs_error[282:353, 121:192], vmin=vmin_er, vmax=vmax_er)
    rect = patches.Rectangle((0, 0), 70, 70, linestyle='dashed', linewidth=2, edgecolor='white', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig('../img_1_4/plot/4_SynNet3D_30_er_ROI.png', dpi=300)
    plt.close()
    ## slice err overwrite
    img_slice = cv2.imread('../img_1_4/plot/4_SynNet3D_30_er.png')
    img_ROI = cv2.imread('../img_1_4/plot/4_SynNet3D_30_er_ROI.png')
    img_slice[:img_ROI.shape[0], :img_ROI.shape[1]] = img_ROI
    cv2.imwrite('../img_1_4/plot/4_SynNet3D_30_er_ROI_.png', img_slice)
    # ----------

    return None

"""
vmin=0.000000000000000
vmax=0.042851172387600
vmin_er=0.000000000000000
vmax_er=0.002884379215539
"""

if __name__ == '__main__':
    plot_img()
