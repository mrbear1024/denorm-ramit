import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ramit import RAMiT
from dataset import CBCTDataset
import numpy as np
import os
import random
from tensorboardX import SummaryWriter
import sys

sys.path.append('./others/program/')
from Evaluation_metrics import Image_Quality_Evaluation

# 是否使用cuda
#device = torch.device('cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#

# input_image_transform  # 输入图像变换
input_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# ground truth_image_transform  # 真实值图像变换
gt_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Start TensorBoard Writer  # 启动 TensorBoard 记录器
writer = SummaryWriter()


# set the random seed  # 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True  # 是否使用确定性算法
        # torch.backends.cudnn.benchmark = False  # 是否启用 cudnn benchmark


def _get_cuda_memory_stats():
    if device.type != "cuda":
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


def _format_cuda_memory_stats(prefix, stats):
    return (
        f"{prefix} cuda_mem_MB: allocated={stats['allocated_mb']:.1f}, "
        f"reserved={stats['reserved_mb']:.1f}, "
        f"max_allocated={stats['max_allocated_mb']:.1f}, "
        f"max_reserved={stats['max_reserved_mb']:.1f}"
    )


def train_model(model, criterion, optimizer, scheduler, dataloader_train, dataloader_val, args):
    num_epochs = args.epoch
    early_stopping_patience = args.early_stopping_patience
    logfile_train = open('./result/train/train_loss_log.txt', 'w')
    logfile_val = open('./result/validation/val_loss_psnr_log.txt', 'w')
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

    # The early stopping and Learning Rate Scheduling are determined with the average PSNR on the validation set.
    # 早停与学习率调度由验证集上的平均 PSNR 决定
    best_PSNR = 0
    epochs_no_improve = 0
    prev_best_model_path = None  # to store the path of the previously best model  # 保存上一次最佳模型的路径

    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logfile_train.write('Epoch {}/{}'.format(epoch, num_epochs - 1) + '\n')
        # After the model is trained for each epoch, use the validation set to verify the model performance
        # 每个 epoch 训练后，使用验证集评估模型性能
        # 1. model_train  # 训练阶段
        print('-' * 10)
        print('model_train')
        logfile_train.write('-' * 10 + '\n')
        logfile_train.write('model_train' + '\n')

        data_size_train = len(dataloader_train.dataset)
        step_train = 0
        if data_size_train % dataloader_train.batch_size != 0:
            total_step_train = data_size_train // dataloader_train.batch_size
        else:
            total_step_train = data_size_train // dataloader_train.batch_size - 1
        total_loss_train = 0

        model.train()
        for x, y in dataloader_train:
            inputs = x.to(device)  # input: SV_CBCT_img_2D  # 输入图像
            gts = y.to(device)  # gt: FV_CBCT_img_2D  # 真实值图像
            # labels = gts - inputs  # residual.   #########删除  # 残差
            # zero the parameter gradients  # 清空梯度
            #optimizer.zero_grad()
            optimizer.zero_grad(set_to_none = True)
            # forward  # 前向传播
            #outputs = model(inputs)
            #loss = criterion(outputs, gts)
            with torch.cuda.amp.autocast(enabled = use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, gts)
            #loss = criterion(outputs, labels)  # MSE loss  # 均方误差损失
            # backward  # 反向传播
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss_train += loss.item()

            print('%d/%d,train_loss:%0.15f' % (step_train, total_step_train, loss.item()))
            logfile_train.write('%d/%d,train_loss:%0.15f' % (step_train, total_step_train, loss.item()) + '\n')
            step_train += 1

        avg_loss_train = total_loss_train / step_train
        print('epoch %d mean_loss_train: %0.15f\n' % (epoch, avg_loss_train))
        logfile_train.write('epoch %d mean_loss_train:%0.15f\n' % (epoch, avg_loss_train) + '\n')
        mem_stats = _get_cuda_memory_stats()
        if mem_stats is not None:
            mem_line = _format_cuda_memory_stats("train_epoch_end", mem_stats)
            print(mem_line)
            logfile_train.write(mem_line + '\n')
        logfile_train.flush()
        writer.add_scalar('Train/Mean_Loss', avg_loss_train, epoch)  # TensorBoard  # 记录训练损失

        # 2. model_validation  # 验证阶段
        print('-' * 10)
        print('model_validation')
        logfile_val.write('Epoch {}/{}'.format(epoch, num_epochs - 1) + '\n')
        logfile_val.write('-' * 10 + '\n')
        logfile_val.write('model_validation' + '\n')

        data_size_val = len(dataloader_val.dataset)
        step_val = 0
        if data_size_val % dataloader_val.batch_size != 0:
            total_step_val = data_size_val // dataloader_val.batch_size
        else:
            total_step_val = data_size_val // dataloader_val.batch_size - 1
        total_loss_val = 0
        total_PSNR_val = 0

        use_amp_val = False

        model.eval()
        with torch.no_grad():
            for x, y in dataloader_val:
                batch_PSNR_val = 0

                inputs = x.to(device)  # input: SV_CBCT_img_2D  # 输入图像
                gts = y.to(device)  # gt: FV_CBCT_img_2D  # 真实值图像
                #########labels = gts - inputs  # residual  # 残差
                # forward  # 前向传播
                #outputs = model(inputs)
                #loss = criterion(outputs, gts)
                with torch.cuda.amp.autocast(enabled = use_amp_val):
                    outputs = model(inputs)
                    loss = criterion(outputs, gts)
                #loss = criterion(outputs, labels)
                total_loss_val += loss.item()
                # *** Calculate PSNR ***  # 计算 PSNR
                inputs = torch.squeeze(inputs).cpu().numpy()  # b,1,h,w -> b,h,w  # 去掉通道维
                outputs = torch.squeeze(outputs).cpu().numpy()
                gts = torch.squeeze(gts).cpu().numpy()
                for num in range(outputs.shape[0]):
                    input = inputs[num]
                    output = outputs[num]
                    gt = gts[num]  # FV_CBCT_img_2D

                    evaluation_metrics = Image_Quality_Evaluation(gt, input + output)
                    psnr = evaluation_metrics.PSNR()
                    batch_PSNR_val += psnr
                    total_PSNR_val += psnr

                batch_avg_PSNR_val = batch_PSNR_val / outputs.shape[0]
                # *******

                print('%d/%d,val_loss:%0.15f' % (step_val, total_step_val, loss.item()))
                logfile_val.write('%d/%d,val_loss:%0.15f' % (step_val, total_step_val, loss.item()) + '\n')
                print('%d/%d,mean_val_PSNR:%0.15f' % (step_val, total_step_val, batch_avg_PSNR_val))
                logfile_val.write('%d/%d,mean_val_PSNR:%0.15f' % (step_val, total_step_val, batch_avg_PSNR_val) + '\n')
                step_val += 1

        avg_loss_val = total_loss_val / step_val
        print('epoch %d mean_loss_val: %0.15f' % (epoch, avg_loss_val))
        logfile_val.write('epoch %d mean_loss_val:%0.15f' % (epoch, avg_loss_val) + '\n')
        avg_PSNR_val = total_PSNR_val / data_size_val
        print('epoch %d mean_PSNR_val: %0.15f\n' % (epoch, avg_PSNR_val))
        logfile_val.write('epoch %d mean_PSNR_val:%0.15f\n' % (epoch, avg_PSNR_val) + '\n')
        mem_stats = _get_cuda_memory_stats()
        if mem_stats is not None:
            mem_line = _format_cuda_memory_stats("val_epoch_end", mem_stats)
            print(mem_line)
            logfile_val.write(mem_line + '\n')
        logfile_val.flush()
        writer.add_scalar('Val/Mean_Loss', avg_loss_val, epoch)  # TensorBoard  # 记录验证损失
        writer.add_scalar('Val/Mean_PSNR', avg_PSNR_val, epoch)  # TensorBoard  # 记录验证 PSNR

        # Learning Rate Scheduling  # 学习率调度
        scheduler.step(avg_PSNR_val)

        # Early stopping  # 早停
        if avg_PSNR_val > best_PSNR:
            best_PSNR = avg_PSNR_val
            epochs_no_improve = 0

            # Delete the previous best model file  # 删除之前的最佳模型文件
            if prev_best_model_path is not None and os.path.isfile(prev_best_model_path):
                os.remove(prev_best_model_path)

            # Save the new best model and update the file path
            # 保存新的最佳模型并更新文件路径
            prev_best_model_path = './result/weight/best_model_epoch_%d.pth' % epoch
            torch.save(model.state_dict(), prev_best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print('Early stopping!')
                logfile_train.write('Early stopping!' + '\n')
                logfile_val.write('Early stopping!' + '\n')
                logfile_train.close()
                logfile_val.close()
                break

        ## ---save the weight file of the newest training epoch and delete the weight file of the previous epoch---
        ## ---保存当前 epoch 的权重文件并删除上一个 epoch 的权重文件---
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_PSNR': best_PSNR},
                   './result/weight/model_epoch_%03d.pth' % epoch)
        if epoch > 0:
            os.remove('./result/weight/model_epoch_%03d.pth' % (epoch - 1))
        ## ------

    logfile_train.close()
    logfile_val.close()

    return None


def main(args):
    set_seed(0)
    #stime = time.time()
    model = RAMiT(in_chans = 1, dim = 64, target_mode = 'light_graydn').to(device)
    criterion = nn.L1Loss()      #改成MAE
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # default lr * 0.1  # 默认学习率的 0.1
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.scheduler_patience, threshold=0)
                                  #verbose=True)  # 输出学习率变化日志

    dataset_train = CBCTDataset('../data/2.2_SV_62_FDK_Reconstruction/training_set',
                                '../data/1_FV_492_FDK_Reconstruction/training_set',
                                input_transform=input_transforms, gt_transform=gt_transforms, mode = 'train', crop_size = 256)
    dataset_val = CBCTDataset('../data/2.2_SV_62_FDK_Reconstruction/validation_set',
                              '../data/1_FV_492_FDK_Reconstruction/validation_set',
                              input_transform=input_transforms, gt_transform=gt_transforms, mode = 'val', crop_size = 256)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=2) #num_workers=10  # 数据加载线程数
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True, num_workers=2)  #num_workers=10  # 数据加载线程数
    """
    etime = time.now()

    duration = (etime - stime )
    print("prepare.duration=", duration)
    print("====================")
    
    stime = time.time()
    train_model(model, criterion, optimizer, scheduler, dataloader_train, dataloader_val, args)
    etime = time.time()
    
    print("train_model.duration=", etime - stime) 
    print("===============")
    """
    train_model(model, criterion, optimizer, scheduler, dataloader_train, dataloader_val, args)

    return None


if __name__ == '__main__':
    # # ***complexity of the model***  # 模型复杂度
    # from thop import profile
    #
    # model = Unet(1, 1)
    # input = torch.randn(1, 1, 512, 512)
    # flops, params = profile(model, inputs=(input,))
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))
    # print('GFLOPs: %.1f' % (flops / 10 ** 9))
    # print('Paramsx10^6: %.1f' % (params / 10 ** 6))
    #
    # print('Total TFLOPs: %.1f' % (flops * 93 / 10 ** 12))
    # print('Total Paramsx10^6: %.1f' % (params / 10 ** 6))
    # # ******

    # ----Train----  # 训练
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch_size', type=int, default=4)
    parse.add_argument('--epoch', type=int, default=200)    #默认为120
    parse.add_argument("--scheduler_patience", type=int, default=10)
    parse.add_argument("--early_stopping_patience", type=int, default=20)
    parse.add_argument('--gpu', type=str, default='0')
    parse.add_argument('--amp', type = int, default = 1)
    args = parse.parse_args()

    # GPU Number  # GPU 编号
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
