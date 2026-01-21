from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import torch


# No data augmentation  # 无数据增强
def make_dataset(input_path, gt_path):
    """
    input: sparse-view CBCT FDK reconstruction image (2D)
    gt: full-view CBCT FDK reconstruction image (2D)
    输入: 稀疏视角 CBCT FDK 重建图像 (2D)
    真实值: 全视角 CBCT FDK 重建图像 (2D)
    """
    imgs = []
    case_list = os.listdir(input_path)  # 按病例目录组织
    for case in case_list:
        slice_list = os.listdir('%s/%s' % (input_path, case))  # 病例内的切片文件
        for slice in slice_list:
            sv_img_2D_path = '%s/%s/%s' % (input_path, case, slice)  # 稀疏视角路径
            fv_img_2D_path = '%s/%s/%s' % (gt_path, case, slice)  # 全视角路径
            imgs.append((sv_img_2D_path, fv_img_2D_path))

    return imgs       #得到一个包含两个文件路径的元祖存储在imgs的列表中  # (input, gt) 路径对列表


class CBCTDataset(Dataset):
    def __init__(self, input_path, gt_path, input_transform=None, gt_transform=None, mode='train', crop_size = None):

        imgs = make_dataset(input_path, gt_path)  # 构建路径对列表
        self.imgs = imgs
        self.input_transform = input_transform  # input  # 输入变换
        self.gt_transform = gt_transform  # ground truth  # 真实值变换
        self.mode = mode  # training process or test process  # 训练/验证模式
        self.crop_size = crop_size  # 训练时随机裁剪尺寸

    def __getitem__(self, index):
        sv_img_2D_path, fv_img_2D_path = self.imgs[index]

        with open(sv_img_2D_path, 'rb') as f:
            sv_img_2D = pickle.load(f)  # input   float32  # 输入图像

        with open(fv_img_2D_path, 'rb') as f:
            fv_img_2D = pickle.load(f)  # gt  # 真实值图像

        sv_img_2D = self.input_transform(sv_img_2D)  # 统一变换到张量
        fv_img_2D = self.gt_transform(fv_img_2D)

        if self.mode == 'train' and self.crop_size is not None:  # 训练时随机裁剪
            _, h, w = sv_img_2D.shape
            top = torch.randint(0, h - self.crop_size + 1, (1,)).item()  # 随机裁剪起点
            left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
            sv_img_2D = sv_img_2D[:, top:top + self.crop_size, left:left+self.crop_size]
            fv_img_2D = fv_img_2D[:, top:top+self.crop_size, left:left+self.crop_size]

        if self.mode =='train':  # 训练与验证返回一致
            return sv_img_2D, fv_img_2D
        else:
            return sv_img_2D, fv_img_2D
            #return sv_img_2D, fv_img_2D, sv_img_2D_path, fv_img_2D_path  # return paths too

    def __len__(self):
        return len(self.imgs)  # 样本对数量
