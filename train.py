import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
from tqdm import tqdm
import torch.distributed as dist

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
writer = None


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


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    if args.distributed:
        if not torch.cuda.is_available() and args.backend == "nccl":
            args.backend = "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.backend, init_method="env://")
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    return (not args.distributed) or args.rank == 0


def _safe_log(logfile, msg, args):
    if is_main_process(args) and logfile is not None:
        logfile.write(msg)


def _all_reduce_sum(value, args):
    if not args.distributed:
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _get_model_state_dict(model, args):
    if args.distributed:
        return model.module.state_dict()
    return model.state_dict()


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
    logfile_train = open('./result/train/train_loss_log.txt', 'w') if is_main_process(args) else None
    logfile_val = open('./result/validation/val_loss_psnr_log.txt', 'w') if is_main_process(args) else None
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

    # The early stopping and Learning Rate Scheduling are determined with the average PSNR on the validation set.
    # 早停与学习率调度由验证集上的平均 PSNR 决定
    best_PSNR = 0
    epochs_no_improve = 0
    prev_best_model_path = None  # to store the path of the previously best model  # 保存上一次最佳模型的路径

    epoch_iter = tqdm(range(num_epochs), desc="Epochs", unit="epoch", ascii=True, disable=not is_main_process(args))
    for epoch in epoch_iter:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if args.distributed and hasattr(dataloader_train, "sampler") and isinstance(dataloader_train.sampler, DistributedSampler):
            dataloader_train.sampler.set_epoch(epoch)
        if is_main_process(args):
            tqdm.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
        _safe_log(logfile_train, 'Epoch {}/{}'.format(epoch, num_epochs - 1) + '\n', args)
        # After the model is trained for each epoch, use the validation set to verify the model performance
        # 每个 epoch 训练后，使用验证集评估模型性能
        # 1. model_train  # 训练阶段
        if is_main_process(args):
            tqdm.write('-' * 10)
            tqdm.write('model_train')
        _safe_log(logfile_train, '-' * 10 + '\n', args)
        _safe_log(logfile_train, 'model_train' + '\n', args)

        step_train = 0
        total_step_train = max(len(dataloader_train) - 1, 0)
        total_loss_train = 0

        model.train()
        train_pbar = tqdm(
            dataloader_train,
            total=len(dataloader_train),
            desc='Train {}/{}'.format(epoch + 1, num_epochs),
            unit='batch',
            leave=False,
            ascii=True,
            disable=not is_main_process(args),
        )
        for x, y in train_pbar:
            inputs = x.to(device)  # input: SV_CBCT_img_2D  # 输入图像
            gts = y.to(device)  # gt: FV_CBCT_img_2D  # 真实值图像
            # labels = gts - inputs  # residual.   #########删除  # 残差
            # zero the parameter gradients  # 清空梯度
            #optimizer.zero_grad()
            optimizer.zero_grad(set_to_none = True)
            # forward  # 前向传播
            #outputs = model(inputs)
            #loss = criterion(outputs, gts)
            # torch.amp.autocast('cuda', args...)
            # with torch.cuda.amp.autocast(enabled = use_amp):
            with torch.cuda.amp.autocast('cuda', enabled = use_amp):
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

            train_pbar.set_postfix(loss='{:.6f}'.format(loss.item()))
            _safe_log(logfile_train, '%d/%d,train_loss:%0.15f' % (step_train, total_step_train, loss.item()) + '\n', args)
            step_train += 1

        sum_loss_train = torch.tensor(total_loss_train, device=device)
        sum_step_train = torch.tensor(step_train, device=device, dtype=torch.float32)
        sum_loss_train = _all_reduce_sum(sum_loss_train, args)
        sum_step_train = _all_reduce_sum(sum_step_train, args)
        avg_loss_train = (sum_loss_train / sum_step_train).item() if sum_step_train.item() > 0 else 0.0
        if is_main_process(args):
            tqdm.write('epoch %d mean_loss_train: %0.15f\n' % (epoch, avg_loss_train))
        _safe_log(logfile_train, 'epoch %d mean_loss_train:%0.15f\n' % (epoch, avg_loss_train) + '\n', args)
        mem_stats = _get_cuda_memory_stats()
        if mem_stats is not None:
            mem_line = _format_cuda_memory_stats("train_epoch_end", mem_stats)
            if is_main_process(args):
                tqdm.write(mem_line)
            _safe_log(logfile_train, mem_line + '\n', args)
        if is_main_process(args) and logfile_train is not None:
            logfile_train.flush()
        if writer is not None and is_main_process(args):
            writer.add_scalar('Train/Mean_Loss', avg_loss_train, epoch)  # TensorBoard  # 记录训练损失

        # 2. model_validation  # 验证阶段
        if is_main_process(args):
            tqdm.write('-' * 10)
            tqdm.write('model_validation')
        _safe_log(logfile_val, 'Epoch {}/{}'.format(epoch, num_epochs - 1) + '\n', args)
        _safe_log(logfile_val, '-' * 10 + '\n', args)
        _safe_log(logfile_val, 'model_validation' + '\n', args)

        data_size_val = len(dataloader_val.dataset)
        step_val = 0
        total_step_val = max(len(dataloader_val) - 1, 0)
        total_loss_val = 0
        total_PSNR_val = 0
        total_val_count = 0

        use_amp_val = False

        model.eval()
        with torch.no_grad():
            val_pbar = tqdm(
                dataloader_val,
                total=len(dataloader_val),
                desc='Val {}/{}'.format(epoch + 1, num_epochs),
                unit='batch',
                leave=False,
                ascii=True,
                disable=not is_main_process(args),
            )
            for x, y in val_pbar:
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
                total_val_count += outputs.shape[0]

                batch_avg_PSNR_val = batch_PSNR_val / outputs.shape[0]
                # *******

                val_pbar.set_postfix(
                    loss='{:.6f}'.format(loss.item()),
                    psnr='{:.3f}'.format(batch_avg_PSNR_val),
                )
                _safe_log(logfile_val, '%d/%d,val_loss:%0.15f' % (step_val, total_step_val, loss.item()) + '\n', args)
                _safe_log(logfile_val, '%d/%d,mean_val_PSNR:%0.15f' % (step_val, total_step_val, batch_avg_PSNR_val) + '\n', args)
                step_val += 1

        sum_loss_val = torch.tensor(total_loss_val, device=device)
        sum_step_val = torch.tensor(step_val, device=device, dtype=torch.float32)
        sum_psnr_val = torch.tensor(total_PSNR_val, device=device)
        sum_count_val = torch.tensor(total_val_count, device=device, dtype=torch.float32)
        sum_loss_val = _all_reduce_sum(sum_loss_val, args)
        sum_step_val = _all_reduce_sum(sum_step_val, args)
        sum_psnr_val = _all_reduce_sum(sum_psnr_val, args)
        sum_count_val = _all_reduce_sum(sum_count_val, args)
        avg_loss_val = (sum_loss_val / sum_step_val).item() if sum_step_val.item() > 0 else 0.0
        avg_PSNR_val = (sum_psnr_val / sum_count_val).item() if sum_count_val.item() > 0 else 0.0
        if is_main_process(args):
            tqdm.write('epoch %d mean_loss_val: %0.15f' % (epoch, avg_loss_val))
            tqdm.write('epoch %d mean_PSNR_val: %0.15f\n' % (epoch, avg_PSNR_val))
        _safe_log(logfile_val, 'epoch %d mean_loss_val:%0.15f' % (epoch, avg_loss_val) + '\n', args)
        _safe_log(logfile_val, 'epoch %d mean_PSNR_val:%0.15f\n' % (epoch, avg_PSNR_val) + '\n', args)
        mem_stats = _get_cuda_memory_stats()
        if mem_stats is not None:
            mem_line = _format_cuda_memory_stats("val_epoch_end", mem_stats)
            if is_main_process(args):
                tqdm.write(mem_line)
            _safe_log(logfile_val, mem_line + '\n', args)
        if is_main_process(args) and logfile_val is not None:
            logfile_val.flush()
        if writer is not None and is_main_process(args):
            writer.add_scalar('Val/Mean_Loss', avg_loss_val, epoch)  # TensorBoard  # 记录验证损失
            writer.add_scalar('Val/Mean_PSNR', avg_PSNR_val, epoch)  # TensorBoard  # 记录验证 PSNR

        # Learning Rate Scheduling  # 学习率调度
        metric_tensor = torch.tensor(avg_PSNR_val if is_main_process(args) else 0.0, device=device)
        if args.distributed:
            dist.broadcast(metric_tensor, src=0)
        scheduler.step(metric_tensor.item())

        # Early stopping  # 早停
        early_stop = False
        if is_main_process(args):
            if avg_PSNR_val > best_PSNR:
                best_PSNR = avg_PSNR_val
                epochs_no_improve = 0

                # Delete the previous best model file  # 删除之前的最佳模型文件
                if prev_best_model_path is not None and os.path.isfile(prev_best_model_path):
                    os.remove(prev_best_model_path)

                # Save the new best model and update the file path
                # 保存新的最佳模型并更新文件路径
                prev_best_model_path = './result/weight/best_model_epoch_%d.pth' % epoch
                torch.save(_get_model_state_dict(model, args), prev_best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve == early_stopping_patience:
                    tqdm.write('Early stopping!')
                    _safe_log(logfile_train, 'Early stopping!' + '\n', args)
                    _safe_log(logfile_val, 'Early stopping!' + '\n', args)
                    early_stop = True
        if args.distributed:
            early_stop_tensor = torch.tensor(1 if early_stop else 0, device=device)
            dist.broadcast(early_stop_tensor, src=0)
            early_stop = bool(early_stop_tensor.item())
        if early_stop:
            break

        ## ---save the weight file of the newest training epoch and delete the weight file of the previous epoch---
        ## ---保存当前 epoch 的权重文件并删除上一个 epoch 的权重文件---
        if is_main_process(args):
            torch.save({'model_state_dict': _get_model_state_dict(model, args),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_PSNR': best_PSNR},
                       './result/weight/model_epoch_%03d.pth' % epoch)
            if epoch > 0:
                os.remove('./result/weight/model_epoch_%03d.pth' % (epoch - 1))
        ## ------

    if logfile_train is not None:
        logfile_train.close()
    if logfile_val is not None:
        logfile_val.close()

    return None


def main(args):
    init_distributed_mode(args)
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cpu")
    set_seed(args.seed + args.rank)
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
    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        val_sampler = DistributedSampler(dataset_val, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                  sampler=train_sampler, pin_memory=True, num_workers=2) #num_workers=10  # 数据加载线程数
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
                                pin_memory=True, num_workers=2)  #num_workers=10  # 数据加载线程数
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
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None
        )
    if is_main_process(args):
        global writer
        writer = SummaryWriter()
    train_model(model, criterion, optimizer, scheduler, dataloader_train, dataloader_val, args)
    if is_main_process(args) and writer is not None:
        writer.close()
    cleanup_distributed()

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
    parse.add_argument("--seed", type=int, default=0)
    parse.add_argument("--backend", type=str, default="nccl")
    parse.add_argument("--local_rank", type=int, default=0)
    args = parse.parse_args()

    # GPU Number  # GPU 编号
    if "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
