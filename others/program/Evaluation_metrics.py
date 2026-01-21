import numpy as np
from math import log10
from image_similarity_measures.quality_metrics import fsim
import skimage


class Image_Quality_Evaluation():
    def __init__(self, reference, predict):
        """
        reference and predict are ndarrays of the same shape.
        """
        self.reference = reference
        self.predict = predict

    def MSE(self):
        mse = ((self.reference - self.predict) ** 2).mean()
        return mse

    def RMSE(self):
        rmse = (((self.reference - self.predict) ** 2).mean()) ** 0.5
        return rmse

    def PSNR(self):
        psnr = 10 * log10(self.reference.max() ** 2 / self.MSE())
        return psnr

    # def SSIM(self):
    #     cov = np.cov(np.stack((self.reference.ravel(), self.predict.ravel()), axis=0))
    #     mux, muy = self.reference.mean(), self.predict.mean()
    #     L = self.reference.max()
    #     ssim = (2 * mux * muy + (0.01 * L) ** 2) * (2 * cov[0, 1] + (0.03 * L) ** 2) / (
    #             mux ** 2 + muy ** 2 + (0.01 * L) ** 2) / (cov[0, 0] + cov[1, 1] + (0.03 * L) ** 2)
    #     return ssim

    def SSIM(self):
        ssim = skimage.metrics.structural_similarity(self.reference, self.predict, win_size=11,
                                                     data_range=self.reference.max() - self.reference.min())
        return ssim

    def FSIM(self):
        #print("shape=====", self.predict.shape)
        Fsim = fsim(self.reference, self.predict)
        return Fsim


if __name__ == '__main__':
    import pickle

    with open('../../data/1_FV_492_FDK_Reconstruction/validation_set/2021-03-02_135322_FINISHED_Head/40.pkl',
              'rb') as f:
        reference = pickle.load(f)
    with open('../../data/2_SV_123_FDK_Reconstruction/validation_set/2021-03-02_135322_FINISHED_Head/40.pkl',
              'rb') as f:
        predict = pickle.load(f)

    evaluation_metrics = Image_Quality_Evaluation(reference, predict)
    print('RMSE = %.10f' % evaluation_metrics.RMSE())
    print('PSNR = %.10f' % evaluation_metrics.PSNR())
    print('SSIM = %.10f' % evaluation_metrics.SSIM())
    print('FSIM = %.10f' % evaluation_metrics.FSIM())
