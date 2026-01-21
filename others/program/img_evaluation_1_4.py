import pickle
import sys
from Evaluation_metrics import Image_Quality_Evaluation

# FOV_mask: 2D circle
with open('./2D_FOV_mask.pkl', 'rb') as f:
    FOV = pickle.load(f)  # (512,512)

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

method_name = ['SVCT', 'Linear interpolation', 'SynCNN', 'SynNet3D']
method_img = [ SVCT_img, Li_img, SynCNN_img, SynNet3D_img]

for i in range(len(method_img)):
    evaluation_metrics = Image_Quality_Evaluation(Ref_img, method_img[i])
    RMSE = evaluation_metrics.RMSE()
    PSNR = evaluation_metrics.PSNR()
    SSIM = evaluation_metrics.SSIM()
    FSIM = evaluation_metrics.FSIM()
    print('%s img 1/4' % method_name[i])
    print('RMSE = %.10f' % RMSE)
    print('PSNR = %.10f' % PSNR)
    print('SSIM = %.10f' % SSIM)
    print('FSIM  = %.10f' % FSIM)
    print()

"""
SVCT img 1/4
RMSE = 0.0011161603
PSNR = 34.1834971622
SSIM = 0.8191148362
FSIM  = 0.9691038736

Linear interpolation img 1/4
RMSE = 0.0007273301
PSNR = 37.9033972438
SSIM = 0.9417426275
FSIM  = 0.9870710819

SynCNN img 1/4
RMSE = 0.0005774822
PSNR = 39.9072563563
SSIM = 0.9644851001
FSIM  = 0.9930042731

SynNet3D img 1/4
RMSE = 0.0005217358
PSNR = 40.7890166810
SSIM = 0.9690143493
FSIM  = 0.9942396926
"""
