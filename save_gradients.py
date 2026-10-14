# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 19:37:26 2021

@author: Hao Zheng
"""

import numpy as np
import torch
import os
import SimpleITK as sitk
from SE_UNet import SE_UNet
from data import load_json_file
from util import load_itk_image, save_itk
torch.manual_seed(777) # cpu
torch.cuda.manual_seed(777) #gpu
np.random.seed(777) #numpy

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred
    tflat = target
    intersection = ((iflat) * tflat).sum()   
    return 1-((2. * intersection + smooth)/((iflat).sum() + (tflat).sum() + smooth))

def Tversky_loss(pred, target):
    smooth = 1.0
    alpha = 0.05
    beta = 1-alpha
    intersection = (pred*target).sum()
    FP = (pred*(1-target)).sum()
    FN = ((1-pred)*target).sum()
    return 1-(intersection + smooth)/(intersection + alpha*FP + beta*FN + smooth)

def root_Tversky_loss(pred, target, dist):
    alpha0 = 1
    beta0 = 1
    alpha = 0.05
    beta = 1 - alpha
    weight = (0.95*dist+0.05)*alpha0*target + beta0*(1-target)*dist
    #weight = 1
    smooth = 1.0
    sigma1 = 0.0001
    sigma2 = 0.0001
    weight_i = target*sigma1 + (1-target)*sigma2
    intersection = (weight*((pred+weight_i)**0.7)*target).sum()
    intersection2 = (weight*(alpha*pred + beta*target)).sum()
    return 1-(intersection + smooth)/(intersection2 + smooth)


def process_img(data):
    data = data.astype(np.float32)
    data2 = data.copy()
    data2[data2 > 500] = 500
    data2[data2 < -1000] = -1000
    data2 = (data2 + 1000) / 1500
    data[data > 1024] = 1024
    data[data < -1024] = -1024
    data = (data + 1024) / 2048
    return data, data2

def save_gradients_tw(data_root,stage_load,file_root,savepath,file_path,layer=0,gpu='0',s1_epes=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # 关键修复: nibabel 不会自动创建父目录, 若 ./data/pred_1 不存在会抛 FileNotFoundError
    os.makedirs(savepath, exist_ok=True)  # 递归创建输出目录, 已存在则跳过
    case_net = SE_UNet(in_channel=2, n_classes=1)
    # 优化: map_location='cpu' 让权重先落到 CPU, 避免 torch.load 直接在 GPU 上分配一份临时权重副本
    weights_dict = torch.load(os.path.join(stage_load, 'SE_UNet_{}.pth'.format(s1_epes-1)),
                              map_location='cpu')
    case_net.load_state_dict(weights_dict, strict=False)
    del weights_dict  # 权重已拷贝进模型, 立即释放字典, 不再长期占内存
    # 优化: DataParallel 包装后 .cuda() 一次即可, 删除多余的 case_net.cuda() 重复调用
    case_net = torch.nn.DataParallel(case_net).cuda()
    case_net.eval()  # 推理必须 eval(): 本网络 batchnorm=False(全用 InstanceNorm3d, 模式无关),
                     # 真正的关键是 DropLayer 依赖 self.training 随机丢弃 30% 通道, 会使滑窗预测不可复现

    # load data
    file_list = load_json_file(file_path, folder='0', mode=['train', 'val'])
    file_list.sort()
    for idx in range(len(file_list)):
        name = file_list[idx]
        img0, origin, spacing, direction = load_itk_image(data_root+'/data/'+name+ 'data_cut' + '.mha')

        img0=img0-1024
        label, origin, spacing, direction = load_itk_image(data_root+'/mask/'+name+ 'mask_cut' + '.mha')
        img, img2 = process_img(img0)

        # calculate gradients
        img = img[np.newaxis, np.newaxis, ...]
        img2 = img2[np.newaxis, np.newaxis, ...]
        label = label[np.newaxis, np.newaxis, ...]
        # 优化: 双通道整卷 CT 保留在 CPU(numpy), 只把当前 128^3 小块送入 GPU,
        # 避免整卷常驻显存(可达数 GB)造成的无谓显存占用
        x = np.concatenate((img.astype(np.float32), img2.astype(np.float32)), axis=1)
        # 优化: 删除从未参与计算的 weight 读取, 以及仅用于取形状的 y / 完全未用的 w 的 GPU 拷贝

        cube_size = 128
        step = 64
        # 优化: 累加器用 float32 代替默认 float64, 内存减半; 同时删除全程未使用的 grads/grads_num
        pred = np.zeros(label.shape, dtype=np.float32)
        pred_num = np.zeros(label.shape, dtype=np.float32)
        # sliding window
        xnum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 \
            else (x.shape[ 2] - cube_size) // step + 2
        ynum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 \
            else (x.shape[3] - cube_size) // step + 2
        znum = (x.shape[4] - cube_size) // step + 1 if (x.shape[4] - cube_size) % step == 0 \
            else (x.shape[4] - cube_size) // step + 2
        for xx in range(xnum):
            xl = step * xx
            xr = step * xx + cube_size
            if xr > x.shape[2]:
                xr = x.shape[2]
                xl = x.shape[2] - cube_size
            for yy in range(ynum):
                yl = step * yy
                yr = step * yy + cube_size
                if yr > x.shape[3]:
                    yr = x.shape[3]
                    yl = x.shape[3] - cube_size
                for zz in range(znum):
                    zl = step * zz
                    zr = step * zz + cube_size
                    if zr > x.shape[4]:
                        zr = x.shape[4]
                        zl = x.shape[4] - cube_size

                    # 优化: 仅这一个 patch(1*2*128*128*128)进入显存, 用完马上释放
                    x_input = torch.from_numpy(x[:, :, xl:xr, yl:yr, zl:zr]).cuda()
                    # 关键修复: no_grad 关闭自动求导, 否则每个 patch 都会构建计算图并常驻显存
                    with torch.no_grad():
                        _, p = case_net(x_input)  # 优化: p0 从未使用, 不再保留其引用
                    pred[:, :, xl:xr, yl:yr, zl:zr] += p.cpu().numpy()
                    pred_num[:, :, xl:xr, yl:yr, zl:zr] += 1
                    del x_input, p  # 显式断开本 patch 的显存引用, 便于 allocator 复用

        pred = pred / pred_num
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        print(name, 'dice', 2 * (pred * label).sum() / (pred + label).sum())
        # np.save(os.path.join('./data/LIBBP/grads', name+'.npy'), grads[0,0])
        # 修复: pred[0] 仍是 (1,Z,Y,X) 的 4 维, SimpleITK 会把首轴当成向量分量,
        # 取 pred[0,0] 压成 3D 标量体数据, 与 prediction.py 中 np.squeeze 后的保存惯例一致
        save_itk(pred[0, 0].astype(np.uint8), origin, spacing, direction,
                 os.path.join(savepath, name + '.mha'))

        # 优化: 单个 case 结束后释放所有大数组引用并清空显存缓存, 防止多 case 循环时碎片累积导致 OOM
        del img, img2, img0, label, x, pred, pred_num
        torch.cuda.empty_cache()




if __name__ == '__main__':
    whichmodel=1#0=old
    # save_gradients_tw(whichmodel,layer=0)















