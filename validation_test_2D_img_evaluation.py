import pickle
import os
import numpy as np
import sys
import argparse

sys.path.append('../others/program/')
from Evaluation_metrics import Image_Quality_Evaluation

# FOV_mask: 2D circle
with open('../others/2D_FOV_mask.pkl', 'rb') as f:
    FOV = pickle.load(f)  # (512,512)


def img_quality_evaluation(args):
    logfile = open('./result/%s/%s_2D_img_evaluation.txt' % (args.dataset.split('_')[0], args.dataset), 'w')
    print("Evaluation in Image domain:")
    logfile.write("Evaluation in Image domain:" + '\n')

    metrics = {'RMSE': [], 'PSNR': [], 'SSIM': [], 'FSIM': []}
    case_list = os.listdir('../data/1_FV_492_FDK_Reconstruction/%s' % args.dataset)
    slice_num = len(os.listdir(
        '../data/1_FV_492_FDK_Reconstruction/%s/%s' % (args.dataset, case_list[0])))  # same length for each case

    for k, case in enumerate(case_list):
        print('%d-%s' % (k, case))
        logfile.write('%d-%s' % (k, case) + '\n')

        for i in range(slice_num):
            with open('../data/1_FV_492_FDK_Reconstruction/%s/%s/%02d.pkl' % (args.dataset, case, i), 'rb') as f:
                reference = pickle.load(f)
            with open('./result/%s/model_output/%s/%02d.pkl' % (args.dataset.split('_')[0], case, i), 'rb') as f:
                predict = pickle.load(f)

            # 圈外置零，负值置零
            reference *= FOV
            reference[reference < 0] = 0
            predict *= FOV
            predict[predict < 0] = 0

            evaluation_metrics = Image_Quality_Evaluation(reference, predict)
            RMSE = evaluation_metrics.RMSE()
            PSNR = evaluation_metrics.PSNR()
            SSIM = evaluation_metrics.SSIM()
            FSIM = evaluation_metrics.FSIM()

            metrics['RMSE'].append(RMSE)
            metrics['PSNR'].append(PSNR)
            metrics['SSIM'].append(SSIM)
            metrics['FSIM'].append(FSIM)

            print('slice_%02d' % i)
            print('RMSE = %.10f' % RMSE)
            print('PSNR = %.10f' % PSNR)
            print('SSIM = %.10f' % SSIM)
            print('FSIM  = %.10f' % FSIM)
            logfile.write('slice_%02d' % i + '\n')
            logfile.write('RMSE = %.10f' % RMSE + '\n')
            logfile.write('PSNR = %.10f' % PSNR + '\n')
            logfile.write('SSIM = %.10f' % SSIM + '\n')
            logfile.write('FSIM  = %.10f' % FSIM + '\n')
            logfile.flush()

        print()
        logfile.write('\n')

    print("Statistical result")
    logfile.write('Statistical result' + '\n')

    print('RMSE %.10f±%.10f' % (np.array(metrics['RMSE']).mean(), np.array(metrics['RMSE']).std(ddof=1)))
    print('PSNR %.10f±%.10f' % (np.array(metrics['PSNR']).mean(), np.array(metrics['PSNR']).std(ddof=1)))
    print('SSIM %.10f±%.10f' % (np.array(metrics['SSIM']).mean(), np.array(metrics['SSIM']).std(ddof=1)))
    print('FSIM %.10f±%.10f' % (np.array(metrics['FSIM']).mean(), np.array(metrics['FSIM']).std(ddof=1)))
    logfile.write('RMSE %.10f±%.10f' % (np.array(metrics['RMSE']).mean(), np.array(metrics['RMSE']).std(ddof=1)) + '\n')
    logfile.write('PSNR %.10f±%.10f' % (np.array(metrics['PSNR']).mean(), np.array(metrics['PSNR']).std(ddof=1)) + '\n')
    logfile.write('SSIM %.10f±%.10f' % (np.array(metrics['SSIM']).mean(), np.array(metrics['SSIM']).std(ddof=1)) + '\n')
    logfile.write('FSIM %.10f±%.10f' % (np.array(metrics['FSIM']).mean(), np.array(metrics['FSIM']).std(ddof=1)) + '\n')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default='validation_set')
    args = parse.parse_args()

    img_quality_evaluation(args)
