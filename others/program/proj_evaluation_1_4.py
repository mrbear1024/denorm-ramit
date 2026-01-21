import pickle
import sys
from Evaluation_metrics import Image_Quality_Evaluation

with open('../proj_1_4/raw_file/Ref_217.pkl', 'rb') as f:
    Ref_proj = pickle.load(f)
with open('../proj_1_4/raw_file/Linear_interpolation_217.pkl', 'rb') as f:
    Li_proj = pickle.load(f)
with open('../proj_1_4/raw_file/SynCNN_217.pkl', 'rb') as f:
    SynCNN_proj = pickle.load(f)
with open('../proj_1_4/raw_file/SynNet3D_217.pkl', 'rb') as f:
    SynNet3D_proj = pickle.load(f)

method_name = ['Linear interpolation', 'SynCNN', 'SynNet3D']
method_proj = [Li_proj, SynCNN_proj, SynNet3D_proj]

for i in range(3):
    evaluation_metrics = Image_Quality_Evaluation(Ref_proj, method_proj[i])
    RMSE = evaluation_metrics.RMSE()
    PSNR = evaluation_metrics.PSNR()
    SSIM = evaluation_metrics.SSIM()
    FSIM = evaluation_metrics.FSIM()
    print('%s projection 1/4' % method_name[i])
    print('RMSE = %.10f' % RMSE)
    print('PSNR = %.10f' % PSNR)
    print('SSIM = %.10f' % SSIM)
    print('FSIM  = %.10f' % FSIM)
    print()

"""
Linear interpolation projection 1/4
RMSE = 0.0431370096
PSNR = 40.8757334896
SSIM = 0.9673914234
FSIM  = 0.9162457862

SynCNN projection 1/4
RMSE = 0.0190798810
PSNR = 47.9612209648
SSIM = 0.9927323234
FSIM  = 0.9874770478

SynNet3D projection 1/4
RMSE = 0.0171955137
PSNR = 48.8644310971
SSIM = 0.9936953067
FSIM  = 0.9909157586

"""
