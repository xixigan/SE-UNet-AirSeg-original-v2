import numpy as np
import torch
import os
import random
import json
import SimpleITK as sitk
import scipy.ndimage as ndimage
import nibabel
from torch.utils.data import Dataset
from util import load_itk_image
from copy import deepcopy

np.random.seed(777)  # numpy

def load_json_file(file_path, folder='0', mode=['train']):
    with open(file_path, 'r') as file:
        data = json.load(file)
    file_list = []
    if int(folder) >= 0:
        for m in mode:
            file_list += data[folder][m]
    else:
        file_list = data[mode[0]]
    file_list = [f.split('.')[0] for f in file_list]
    return file_list


# def npy2niigz():
#     npy_root = './data/pred'
#     niigz_root = './data/pred2'
#     file_list = os.listdir(npy_root)
#     file_list.sort()
#     for f in file_list:
#         name = f.split('.')[0]
#         pred = np.load(os.path.join(npy_root, f))
#         pred = (pred > 0.5).astype(int)
#         pred_nii = nibabel.Nifti1Image(pred.astype(np.uint8), np.eye(4))
#         nibabel.save(pred_nii, os.path.join(os.path.join('./data/pred2', name + '.nii.gz')))
#         print(name, pred.shape)

def random_flip(data_list):
    result=deepcopy(data_list)
    flipid = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    while (flipid  == [1,  1, 1]).all():
        flipid = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    for i in range(len(result)):
        result[i] = np.ascontiguousarray(result[i][::flipid[0], ::flipid[1], ::flipid[2]])
    return result

def random_rotate(data_list):
    def rotate_left(data):
        data = data.transpose((0, 2, 1))
        data = np.ascontiguousarray(data[:, ::-1])
        return data
    def rotate_right(data):
        data = np.ascontiguousarray(data[:, ::-1])
        data = data.transpose((0, 2, 1))
        data = np.ascontiguousarray(data[:, ::-1])
        return data

    result=deepcopy(data_list)
    k=random.random()
    for i in range(len(result)):
        if  k> 0.5:
            result[i] = rotate_left(result[i])
        else:
            result[i] = rotate_right(result[i])
    return result

def random_color(data, rate=0.2):
    r1 = (random.random() - 0.5) * 2 * rate
    r2 = (random.random() - 0.5) * 2 * rate
    data = data * (1 + r2) + r1
    return data

def central_crop(sample, label, dist, crop_size):
    origin_size = sample.shape
    crop_size = np.array(crop_size)
    start = (origin_size - crop_size) // 2
    sample = sample[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
             start[2]:(start[2] + crop_size[2])]
    label = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1], start[2]:start[2] + crop_size[2]]
    dist = dist[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1], start[2]:start[2] + crop_size[2]]
    return sample, label, dist

def skeleton_sample(img, label, weight, loc, cube_size):
    """难例锚点采样(stage2)：从锚点集合 loc 随机取一点，裁剪起点在其 ±cube_size/2 内
    随机抖动后切出 cube_size^3 块。保证 patch 必然覆盖(或紧邻)"被上一阶段漏掉的骨架点"。
    其余 *_sample / *_sample_wg 函数机制完全相同，仅锚点来源与输出路数不同。
    """
    origin_size = img.shape
    crop_size = np.array([cube_size, cube_size, cube_size])
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop

def small_airway_sample(img, label, weight, loc, cube_size):
    """细小气道锚点采样(stage2)：与 skeleton_sample 同机制，锚点来自 loc_small(细支气管区域)"""
    origin_size = img.shape
    crop_size = [cube_size, cube_size, cube_size]
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    return img_crop, label_crop, weight_crop

def hard_sample(img, label, weight, loc_skeleton, loc_small, cube_size):
    """
    stage2 难例挖掘(第一层/空间挖掘)入口：在两个难区锚点集合中二选一，
    再围绕锚点随机抖动裁剪 cube_size^3 patch。
    - loc_skeleton: 上一阶段预测漏掉的骨架点(断头/末梢区域)
    - loc_small:    细小支气管锚点(定义见 AirwayHMData.crop，注意其中有疑似 bug 的注释)
    两个锚点集都为空时退回全卷随机裁剪，保证不崩溃。
    """
    origin_size = img.shape
    crop_size = np.array([cube_size, cube_size, cube_size])

    # 50% 概率选漏检骨架锚点(loc 非空才可选)，否则选细小气道锚点
    if np.random.random() > 0.5 and len(loc_skeleton[0]) > 0:
        loc = loc_skeleton
    elif len(loc_small[0]) > 0:
        loc = loc_small
    else:
        # 兜底：无任何难区锚点时退化为全卷随机裁剪
        x = np.random.randint(0, origin_size[0] - cube_size)
        y = np.random.randint(0, origin_size[1] - cube_size)
        z = np.random.randint(0, origin_size[2] - cube_size)
        return (
            img[x:x+cube_size, y:y+cube_size, z:z+cube_size],
            label[x:x+cube_size, y:y+cube_size, z:z+cube_size],
            weight[x:x+cube_size, y:y+cube_size, z:z+cube_size],
        )

    random_loc = np.random.randint(len(loc[0]))
    start = [
        np.random.randint(max(0, loc[0][random_loc] - cube_size // 2), loc[0][random_loc] + cube_size // 2),
        np.random.randint(max(0, loc[1][random_loc] - cube_size // 2), loc[1][random_loc] + cube_size // 2),
        np.random.randint(max(0, loc[2][random_loc] - cube_size // 2), loc[2][random_loc] + cube_size // 2)
    ]

    for i in range(3):
        if (start[i] + cube_size) > origin_size[i]:
            start[i] = origin_size[i] - cube_size

    img_crop = img[start[0]:start[0]+cube_size, start[1]:start[1]+cube_size, start[2]:start[2]+cube_size]
    label_crop = label[start[0]:start[0]+cube_size, start[1]:start[1]+cube_size, start[2]:start[2]+cube_size]
    weight_crop = weight[start[0]:start[0]+cube_size, start[1]:start[1]+cube_size, start[2]:start[2]+cube_size]

    return img_crop, label_crop, weight_crop

def random_sample(img, label, weight, cube_size):
    """全卷均匀随机裁剪：与难例采样互补的"常规样本"来源(概率 1-hard_ratio)"""
    origin_size = img.shape
    crop_size = [cube_size, cube_size, cube_size]
    start = [np.random.randint(0, origin_size[0] - crop_size[0]), np.random.randint(0, origin_size[1] - crop_size[1]),
             np.random.randint(0, origin_size[2] - crop_size[2])]
    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop

def skeleton_sample_wg(img, label, weight,skeleton, loc, cube_size):
    """stage3 版骨架锚点采样(_wg = with guide)：与 skeleton_sample 同机制，多输出一路 skel_crop，
    供难例遍的 atr_loss(break_loss) 使用"""
    origin_size = img.shape
    crop_size = np.array([cube_size, cube_size, cube_size])
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skel_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop,skel_crop

def break_sample_wg(img, label, weight,skeleton, loc, cube_size):
    """stage3 断点锚点采样：锚点 loc = br_skel，即 weight_br.save_weight_break 预计算的
    "气道树疑似断裂位置"坐标(np.where 索引组)，围绕断点抖动裁剪，专攻断点重建"""
    origin_size = img.shape
    crop_size = np.array([cube_size, cube_size, cube_size])
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skel_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop,skel_crop

def small_airway_sample_wg(img, label, weight,skeleton, loc, cube_size):
    """stage3 版细小气道锚点采样：同 small_airway_sample，多输出 skel_crop"""
    origin_size = img.shape
    crop_size = [cube_size, cube_size, cube_size]
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skel_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop,skel_crop

def random_sample_wg(img, label, weight,skeleton, cube_size):
    """stage3 版全卷随机裁剪：同 random_sample，多输出 skel_crop"""
    origin_size = img.shape
    crop_size = [cube_size, cube_size, cube_size]
    start = [np.random.randint(0, origin_size[0] - crop_size[0]), np.random.randint(0, origin_size[1] - crop_size[1]),
             np.random.randint(0, origin_size[2] - crop_size[2])]
    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skel_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    return img_crop, label_crop, weight_crop,skel_crop

class AirwayHMData(Dataset):
    """
    【stage2 训练数据集】两层互补难例挖掘的第一层(锚点采样)。
    每个 __getitem__ 读入一个病例整卷，crop() 内 for _ in range(self.batch_size)
    裁出 batch_size 个 128^3 patch：每块以 hard_ratio(初值0.4)概率围绕
    漏检骨架点/细小气道锚点抖动裁剪，否则全卷随机裁。
    self.batch_size 是"真功能"参数=每 item 裁几块(决定真实 batch)；
    配合 DataLoader batch_size=1 使用，返回 (B,C,Z,Y,X)，训练循环 transpose(0,1) 还原。
    额外输出 skeleton/parsing/pred 供 GUL 损失与下一轮锚点更新。
    """

    def __init__(self,
                 file_path,
                 data_root,
                 file_root,
                 pred_path,
                 batch_size,
                 cube_size,
                 aug_flag=1):
        self.data_root = data_root
        self.file_root = file_root
        self.pred_path=pred_path
        self.file_list = load_json_file(file_path, folder='0', mode=['train'])
        # 真功能参数：每个 __getitem__ 在同一病例上裁几个 patch(=真实 batch)；
        # 与 DataLoader 的 batch_size 无关，后者须保持 1(见 train.py 关键设计注释)
        self.batch_size = batch_size
        self.cube_size = cube_size
        self.aug_flag = aug_flag


        # Random : Hard = 6 : 4
        self.random_ratio = 0.6
        self.hard_ratio = 0.4

        # scheduler 
        self.decay_step = 5          
        self.decay_rate = 0.05        
        self.max_hard_ratio = 0.8     
        self.min_hard_ratio = 0.2    

    def __len__(self):
        return len(self.file_list)

    def process_img(self, img_crops):
        img2_crops = []
        for i in range(len(img_crops)):
            crop = img_crops[i]
            crop2 = crop.copy()
            crop2[crop2 > 500] = 500
            crop2[crop2 < -1000] = -1000
            crop2 = (crop2 + 1000) / 1500
            crop[crop > 1024] = 1024
            crop[crop < -1024] = -1024
            crop = (crop + 1024) / 2048
            img2_crops.append(crop2)  #
            img_crops[i] = crop
        return img_crops, img2_crops

    def crop(self, img, label, weight, pred, skeleton, parsing):
        """
        难例挖掘第一层(空间挖掘，stage2)：每个 patch 先掷骰子决定
        "瞄准难区"(概率 hard_ratio) 还是 "全卷随机"(概率 1-hard_ratio)。
        :param pred: 上一阶段(stage1)的整卷预测，用于计算"被漏掉的骨架点"
        """
        img_crops, label_crops, weight_crops = [], [], []

        # 每个气道体素到气道壁(背景)的距离；背景体素处为 0
        dis = ndimage.distance_transform_edt(label)
        # 意图：细小支气管锚点 = 骨架上距气道壁 < 2 体素的点(管径仅单像素级)
        # ⚠️ 疑似 bug：非骨架体素处 dis*skeleton 恒为 0，0<2 也为 True，
        #    导致 loc_small 实际包含全部非骨架体素，该分支的采样≈全卷随机；
        #    符合意图的写法应为 skeleton * (dis < 2) > 0，需与原作者确认后修改
        loc_small = np.where((dis * skeleton) < 2)
        # 漏检骨架锚点：骨架 ∩ (1-上一阶段预测) = stage1 没分出来的中心线点(断头/末梢)
        loc_skeleton = np.where(skeleton * (1 - pred))

        cube_size = self.cube_size

        for _ in range(self.batch_size):
            # hard_ratio 初值 0.4，由 update_scheduler 按验证指标动态调节(0.2~0.8)
            if np.random.random() < self.hard_ratio:
                # Hard mining（50% skeleton / 50% small airway）
                img_crop, label_crop, weight_crop = hard_sample(
                    img, label, weight, loc_skeleton, loc_small, cube_size
                )
            else:
                # Random crop
                img_crop, label_crop, weight_crop = random_sample(img, label, weight, cube_size)

            img_crops.append(img_crop)
            label_crops.append(label_crop)
            weight_crops.append(weight_crop)

        return img_crops, label_crops, weight_crops


    def update_scheduler(self, epoch, val_loss_random_list, val_loss_hard_list, val_td_list, val_bd_list):
        """
        【难例采样比例调度器】——不是 PyTorch 的学习率调度器(lr_scheduler)，
        而是 Dataset 内部的闭环反馈：用验证集指标动态调节下一 epoch crop() 里
        难例锚点采样的占比 hard_ratio(本类只调这一个，无 break_ratio)。

        数据来源(调用侧 train.py train2 每个 epoch 末尾)：
          validation() 返回 (TD_mean, BD_mean, loss_random, loss_hard)
          → 分别 append 进 val_td_list / val_bd_list / val_loss_random / val_loss_hard
          → 再调 train_dataset.update_scheduler(ep, ...) 把四个列表传进来。
        注意命名陷阱：val_loss_* 实际是 validation() 里算的 Dice【分数】(越大越好)，
        不是损失；diff = 全局Dice - 难区Dice，通常为正，代表难区落后全局的程度。

        输出：函数内 print 的 [Scheduler] 行只打到终端 stdout，不写入日志文件。
        """
        # 【调节闸门】两个条件缺一不可：
        #   ① epoch % decay_step == 0：stage2 每 5 个 epoch 才调一次 hard_ratio(decay_step=5)，
        #      避免逐 epoch 调节被验证指标的噪声带着跑；(stage3 版 decay_step=1，即每轮都调)
        #   ② epoch != 0：第 0 轮时 val_td_list 只有 1 个点，无"上一 epoch"可比、
        #      趋势无从谈起，且初值 0.4 本就是起点，首调无意义，故直接跳过。
        # 不满足时整个函数什么都不做，hard_ratio 原样带入下一个 epoch。
        if epoch % self.decay_step == 0 and epoch != 0:
            # 每 decay_step 个 epoch 才调一次；取最近 3 个 epoch 的分数均值抗噪
            window = min(3, len(val_loss_random_list))
            avg_random = np.mean(val_loss_random_list[-window:])
            avg_hard = np.mean(val_loss_hard_list[-window:])
            diff = avg_random - avg_hard

            # 趋势 = 本 epoch 与上一 epoch 的差值；只有 1 个历史点时无趋势可言，置 0
            if len(val_td_list) > 1:
                td_trend = val_td_list[-1] - val_td_list[-2]
                bd_trend = val_bd_list[-1] - val_bd_list[-2]
            else:
                td_trend = bd_trend = 0.0

            print(f"[Scheduler] Epoch {epoch}: diff={diff:.4f}, TD_trend={td_trend:.4f}, BD_trend={bd_trend:.4f}")

            if diff > 0.04 or td_trend < 0 or bd_trend < 0:
                # 难区明显落后全局、或树质量在退步 → 加大难例采样占比
                self.hard_ratio = min(self.max_hard_ratio, self.hard_ratio + self.decay_rate)
            elif diff < 0.02 and td_trend >= 0 and bd_trend >= 0:
                # 难区已接近全局水平且树仍在进步 → 减小难例占比，回归均衡采样
                self.hard_ratio = max(self.min_hard_ratio, self.hard_ratio - self.decay_rate)
            # elif diff > 0.05 or td_trend < -1 or bd_trend < -1:
            #     # 【死代码】diff>0.05 必然先命中第一个分支(diff>0.04)，永远走不到这里；
            #     # 且"更差却减小 hard_ratio"与上一分支语义矛盾，疑似笔误，保留未动
            #     self.hard_ratio = max(self.min_hard_ratio, self.hard_ratio - self.decay_rate)

            print(f"[Scheduler] Updated hard_ratio -> {self.hard_ratio:.2f}")

    def augment(self, data_list):
        if random.random() > 0.5:
            result = random_flip(data_list)
            data_list = result
        if random.random() > 0.5:
            result = random_rotate(data_list)
            data_list = result
        return data_list

    def __getitem__(self, item):
        name = self.file_list[item]
        img = sitk.ReadImage(self.data_root + '/data/' + name + 'data_cut' +
                             '.mha')
        img = sitk.GetArrayFromImage(img)
        img=img-1024
        label = sitk.ReadImage(self.data_root + '/mask/' + name + 'mask_cut' +
                               '.mha')
        label = sitk.GetArrayFromImage(label)

        # LIB_weight 已由 npy 改为与 CT 同几何的 .mha(lib_weight.py 产出), 统一走 load_itk_image
        weight, _, _, _ = load_itk_image(
            os.path.join(self.file_root, 'LIB_weight', name + '.mha'))
        # ===== 原 npy 读法(保留备查) =====
        # weight = np.load(
        #     os.path.join(self.file_root, 'LIB_weight', name + '.npy'))
            
        # 统一用 load_itk_image 读取 save_itk 写出的 3D 标量体数据
        # 不再需要旧 nibabel 4 维文件配套的 get_fdata()[0]
        # 显式接收 origin/spacing/direction, 与全局读取惯例一致
        pred, origin, spacing, direction = load_itk_image(
            os.path.join(self.pred_path, name + '.mha'))

        parsing = sitk.ReadImage(
            os.path.join(self.file_root, 'tree_parse',
                         name + 'mask_cut.mha'))
        parsing = sitk.GetArrayFromImage(parsing)
        # parsing = parsing.transpose(2, 1, 0)
        skeleton = sitk.ReadImage(
            os.path.join(self.file_root, 'skeleton', name + 'mask_cut.mha'))
        skeleton = sitk.GetArrayFromImage(skeleton)
        # skeleton = skeleton.transpose(2, 1, 0)

        weight = weight**(np.random.random() + 2) * label + (1 - label)
        img_crops, label_crops, weight_crops = self.crop(
            img, label, weight, pred, skeleton, parsing)
        img_crops, img2_crops = self.process_img(img_crops)
        data_list = [img_crops, img2_crops, label_crops, weight_crops]
        if self.aug_flag == 1:
            for i in range(len(data_list[0])):
                _aug = self.augment(
                    [data_list[j][i] for j in range(len(data_list))])
                for j in range(len(data_list)):
                    data_list[j][i] = _aug[j]
        img_crops, img2_crops, label_crops, weight_crops = data_list[
            0], data_list[1], data_list[2], data_list[3]

        img_crops = torch.from_numpy(np.array(img_crops))
        img2_crops = torch.from_numpy(np.array(img2_crops))
        label_crops = torch.from_numpy(np.array(label_crops))
        weight_crops = torch.from_numpy(np.array(weight_crops))

        # if img_crops.shape!=torch.Size([10, 128, 128, 128]):
        #     dsfs=3

        return img_crops, img2_crops, label_crops, weight_crops

class AirwayHMData3(Dataset):
    """
    【stage3 训练数据集】AirwayHMData 的 stage3 版，难例锚点升级为两路：
    hard_ratio(初值0.8)内再按 break_ratio(0.625)分"断点锚(br_skel)"与
    "骨架/细小气道锚"，其余全卷随机；hard_ratio/break_ratio 由
    update_scheduler 按验证 TD/BD 损失逐 epoch 在 [min,max] 内衰减调节。
    self.batch_size 同样是"真功能"参数=每 item 裁几块(决定真实 batch)，
    配合 DataLoader batch_size=1，返回 (B,C,Z,Y,X)，训练循环 transpose(0,1) 还原。
    额外输出 skel(骨架)一路供 break_loss(atr_loss) 监督。
    """
    def __init__(self, file_path, data_root, file_root,pred2_path,br_skel_path,BR_weight_path,batch_size, cube_size,aug_flag=1):
        self.data_root = data_root
        self.file_root = file_root
        self.pred2_path=pred2_path
        self.br_skel_path=br_skel_path
        self.BR_weight_path=BR_weight_path
        self.file_list = load_json_file(file_path, folder='0', mode=['train'])
        # 真功能参数：每个 __getitem__ 在同一病例上裁几个 patch(=真实 batch)；
        # 与 DataLoader 的 batch_size 无关，后者须保持 1(见 train.py 关键设计注释)
        self.batch_size = batch_size
        self.cube_size = cube_size
        self.aug_flag = aug_flag

        self.hard_ratio = 0.8      
        self.break_ratio = 0.625   
        self.min_hard_ratio = 0.5
        self.max_hard_ratio = 0.9
        self.min_break_ratio = 0.2
        self.max_break_ratio = 0.8
        self.decay_rate = 0.05
        self.decay_step = 1       

    def __len__(self):
        return len(self.file_list)

    def process_img(self, img_crops):
        img2_crops = []
        for i in range(len(img_crops)):
            crop = img_crops[i]
            crop2 = crop.copy()
            crop2[crop2 > 500] = 500
            crop2[crop2 < -1000] = -1000
            crop2 = (crop2 + 1000) / 1500
            crop[crop > 1024] = 1024
            crop[crop < -1024] = -1024
            crop = (crop + 1024) / 2048
            img2_crops.append(crop2) 
            img_crops[i] = crop
        return img_crops, img2_crops

    def crop(self, img, label, weight, pred, skeleton, parsing, br_skel):
        """
        Stage 3 crop: Dynamic sampling including Random, Hard, and Break samples.

        难例挖掘第一层(空间挖掘，stage3)：三级决策树
          随机(1-hard_ratio) → 难例 → 断点(break_ratio) / 细小气道(50%) / 漏检骨架(50%)
        :param pred:     stage2 的整卷预测(pred_2)，用于计算被 stage2 漏掉的骨架点
        :param br_skel:  weight_br.save_weight_break 预计算的断点坐标(np.where 索引组)
        """
        img_crops, label_crops, weight_crops, skel_crops = [], [], [], []

        # 三个锚点集合的定义与 stage2 相同(含 loc_small 同款疑似 bug，见 AirwayHMData.crop 注释)
        dis = ndimage.distance_transform_edt(label)
        loc_small = np.where((dis * skeleton) < 2)
        loc_skeleton = np.where(skeleton * (1 - pred))
        loc_break = br_skel 
        cube_size = self.cube_size

        for _ in range(self.batch_size):
            # ----------------- Random vs Hard+Break -----------------
            # hard_ratio 初值 0.8(范围 0.5~0.9)，由 update_scheduler 按验证指标动态调节
            if np.random.random() < self.hard_ratio:
                # ----------------- Hard mining -----------------
                # 难例分支内再掷一次骰子：break_ratio(初值 0.625) 概率专攻断点，
                # 否则 50/50 在细小气道/漏检骨架中选一种
                if np.random.random() < self.break_ratio and len(loc_break[0]) != 0:
                    # Break sample
                    img_crop, label_crop, weight_crop, skel_crop = break_sample_wg(
                        img, label, weight, skeleton, loc_break, cube_size
                    )
                else:
                    # Skeleton or Small airway
                    if np.random.random() < 0.5:
                        img_crop, label_crop, weight_crop, skel_crop = small_airway_sample_wg(
                            img, label, weight, skeleton, loc_small, cube_size
                        )
                    else:
                        img_crop, label_crop, weight_crop, skel_crop = skeleton_sample_wg(
                            img, label, weight, skeleton, loc_skeleton, cube_size
                        )
            else:
                # ----------------- Random crop -----------------
                img_crop, label_crop, weight_crop, skel_crop = random_sample_wg(
                    img, label, weight, skeleton, cube_size
                )

            img_crops.append(img_crop)
            label_crops.append(label_crop)
            weight_crops.append(weight_crop)
            skel_crops.append(skel_crop)

        return img_crops, label_crops, weight_crops, skel_crops

    def update_scheduler(self, epoch, val_loss_random_list, val_loss_hard_list, val_td_list, val_bd_list):
        """
        Adaptive adjustment of hard_ratio and break_ratio according to validation metrics.

        【难例采样比例调度器(stage3 版)】与 AirwayHMData.update_scheduler 同一机制，
        区别是本类同时调两个采样参数：
          hard_ratio  : crop() 里"难例锚点采样 vs 全卷随机采样"的切换概率
          break_ratio : 难例内部"断点锚点 vs 细小分支锚点"的切换概率
        数据来源同样是 train.py train3 每个 epoch 末尾：validation() 的
        (TD_mean, BD_mean, loss_random, loss_hard) 累积进四个列表后传入。
        val_loss_* 实为 Dice 分数(越大越好)，命名带 loss 是历史遗留。
        [Scheduler] 输出只打到终端 stdout，不写入日志文件。
        """
        # 【调节闸门】与 AirwayHMData.update_scheduler 同一逻辑，只是写成了"不满足就提前 return"
        # 的反向形式：① epoch % decay_step != 0 时跳过——本类 decay_step=1，条件恒假，
        #    即 stage3 每个 epoch 都调节(比 stage2 的每 5 轮更频繁，因 hard/break 双参数需快速响应)；
        # ② epoch == 0 时跳过——首轮 val_td_list 只有 1 个点，无趋势基线可比，
        #    且 hard_ratio=0.8/break_ratio=0.625 初值本就是起点。
        # 不满足任一条件时两个比例原样带入下一个 epoch。
        if epoch % self.decay_step != 0 or epoch == 0:
            return  # only update every decay_step

        # ------------------ Compute average losses ------------------
        # 取最近 3 个 epoch 的 Dice 分数均值抗噪；diff = 全局Dice - 难区Dice(通常为正)
        window = min(3, len(val_loss_random_list))
        avg_random = np.mean(val_loss_random_list[-window:])
        avg_hard = np.mean(val_loss_hard_list[-window:])
        diff = avg_random - avg_hard

        # Compute TD/BD trends
        # 趋势 = 本 epoch 减上一 epoch；只有 1 个历史点时置 0(不据此调节)
        if len(val_td_list) > 1:
            td_trend = val_td_list[-1] - val_td_list[-2]
            bd_trend = val_bd_list[-1] - val_bd_list[-2]
        else:
            td_trend = bd_trend = 0.0

        print(f"[Scheduler] Epoch {epoch}: diff={diff:.4f}, TD_trend={td_trend:.4f}, BD_trend={bd_trend:.4f}")

        # ------------------ Adjust hard_ratio ------------------
        if diff > 0.04 or td_trend < 0 or bd_trend < 0:
            # Hard examples are worse or tree quality decreased → increase Hard ratio
            self.hard_ratio = min(self.max_hard_ratio, self.hard_ratio + self.decay_rate)
        elif diff < 0.02 and td_trend >= 0 and bd_trend >= 0:
            # Hard close to overall and tree improving → decrease Hard ratio
            self.hard_ratio = max(self.min_hard_ratio, self.hard_ratio - self.decay_rate)
        # else: keep unchanged

        # ------------------ Adjust break_ratio within Hard ------------------
        # Simple rule: if TD/BD trend < 0, increase break_ratio to focus on break points
        # 断点(气道被砍断)直接拉低 TD/BD，故树质量退步时优先补断点锚点
        if td_trend < 0 or bd_trend < 0:
            self.break_ratio = min(self.max_break_ratio, self.break_ratio + self.decay_rate)
        elif td_trend > 0 and bd_trend > 0:
            self.break_ratio = max(self.min_break_ratio, self.break_ratio - self.decay_rate)
        # else: keep unchanged

        print(f"[Scheduler] Updated hard_ratio -> {self.hard_ratio:.2f}, break_ratio -> {self.break_ratio:.2f}")

    def augment(self, data_list):
        if random.random() > 0.5:
            result = random_flip(data_list)
            data_list = result
        if random.random() > 0.5:
            result = random_rotate(data_list)
            data_list = result
        return data_list

    def __getitem__(self, item):
        name = self.file_list[item]
        img = sitk.ReadImage(self.data_root+'/data/'+name+ 'data_cut' + '.mha')
        img = sitk.GetArrayFromImage(img)
        img=img-1024
        label = sitk.ReadImage(self.data_root+'/mask/'+name+ 'mask_cut' + '.mha')
        label = sitk.GetArrayFromImage(label)
        r=0.6
        # LIB_weight/BR_weight 已 mha 化(与 CT 同几何), 统一走 load_itk_image
        weight, _, _, _ = load_itk_image(os.path.join(self.file_root, 'LIB_weight', name + '.mha'))
        br_weight, _, _, _ = load_itk_image(os.path.join(self.BR_weight_path, name + '.mha'))

        weight = weight +r* br_weight
        # br_skel 现在存的是 uint8 二值体数据, 读出后现算断点坐标索引组
        # (crop() 的 loc_break 需要 np.where 三元组, 与产生端 weight_br.py 对齐)
        br_skel_vol, _, _, _ = load_itk_image(os.path.join(self.br_skel_path, name + '.mha'))
        br_skel = np.where(br_skel_vol == 1)
        # ===== 原 npy 读法(保留备查): LIB/BR 权重读 fp16/fp16 npy, br_skel 直接是坐标三元组 =====
        # weight = np.load(os.path.join(self.file_root, 'LIB_weight', name + '.npy'))
        # br_weight = np.load(os.path.join(self.BR_weight_path, name + '.npy'))
        # br_skel = np.load(os.path.join(self.br_skel_path, name + '.npy'))
        # 统一用 load_itk_image 读取 save_itk 写出的 3D 标量体数据,
        # 不再需要旧 nibabel 4 维文件配套的 get_fdata()[0]
        # 显式接收 origin/spacing/direction, 与全局读取惯例一致
        pred, origin, spacing, direction = load_itk_image(os.path.join(self.pred2_path, name + '.mha'))

        parsing = sitk.ReadImage(os.path.join(self.file_root, 'tree_parse', name + 'mask_cut.mha'))
        parsing = sitk.GetArrayFromImage(parsing)
        # parsing=parsing.transpose(2,1,0)
        skeleton = sitk.ReadImage(os.path.join(self.file_root, 'skeleton', name + 'mask_cut.mha'))
        skeleton = sitk.GetArrayFromImage(skeleton)
        # skeleton=skeleton.transpose(2,1,0)


        weight = weight ** (np.random.random() + 2) * label + (1 - label)
        img_crops, label_crops, weight_crops,skel_crops = self.crop(img, label, weight, pred, skeleton, parsing,br_skel)
        img_crops, img2_crops = self.process_img(img_crops)
        data_list = [img_crops, img2_crops, label_crops, weight_crops,skel_crops]
        if self.aug_flag==1:
            for i in range(len(data_list[0])):
                _aug = self.augment([data_list[j][i] for j in range(len(data_list))])
                for j in range(len(data_list)):
                    data_list[j][i] = _aug[j]
        img_crops, img2_crops, label_crops, weight_crops,skel_crops  = data_list[0], data_list[1], data_list[2], data_list[3],data_list[4]
        img_crops = torch.from_numpy(np.array(img_crops))
        img2_crops = torch.from_numpy(np.array(img2_crops))
        label_crops = torch.from_numpy(np.array(label_crops))
        weight_crops = torch.from_numpy(np.array(weight_crops))
        skel_crops = torch.from_numpy(np.array(skel_crops))

        return img_crops, img2_crops, label_crops, weight_crops,skel_crops

class OnlineHMData(Dataset):
    """
    难例挖掘第二层(损失挖掘，stage2)的读回端：
    把 save_data_online 按损失值存盘的难例 patch 读回来，供每 epoch 第二遍训练。
    文件名前缀就是损失值(float)，按损失升序排序后取末尾 rate 比例(=损失最大的那部分)。
    调用处传 rate=1.0：缓存本身已只保留最难的前 30%，故全部读回。
    """
    # 原签名带 batch_size 参数(类内从未使用)，已注释去除：
    # def __init__(self, data_root, batch_size, rate=0.33):
    def __init__(self, data_root, rate=0.33):
        self.data_root = data_root
        # 本类每 item 就是磁盘上单个难例 patch，真实 batch 完全由 DataLoader 的 batch_size 决定；
        # 原 self.batch_size = batch_size 为无用赋值，已注释去除：
        # self.batch_size = batch_size
        self.name_list = os.listdir(os.path.join(data_root, 'image'))
        # 按文件名前缀(=该 patch 的 GUL 损失)升序排序：越靠后越难
        self.name_list.sort(key=lambda x:float(x.split('_')[0]))
        # 取损失最大的末尾 rate 比例
        self.name_list = self.name_list[-int(rate * len(self.name_list)):]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        image = np.load(os.path.join(self.data_root, 'image', name))
        label = np.load(os.path.join(self.data_root, 'label', name))
        weight = np.load(os.path.join(self.data_root, 'weight', name))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        weight = torch.from_numpy(weight)

        return image, label, weight

class OnlineHMData3(Dataset):
    """
    OnlineHMData 的 stage3 版：多读一路 skel(骨架)，因为难例遍的损失含 atr_loss(break_loss)，
    必须有骨架作监督信号。排序/取末尾比例逻辑与 OnlineHMData 完全相同。
    """
    # 原签名带 batch_size 参数(同 OnlineHMData，类内从未使用)，已注释去除：
    # def __init__(self, data_root, batch_size, rate=0.33):
    def __init__(self, data_root, rate=0.33):
        self.data_root = data_root
        # 同 OnlineHMData：每 item 单个 patch，真实 batch 由 DataLoader 决定；
        # 原 self.batch_size = batch_size 为无用赋值，已注释去除：
        # self.batch_size = batch_size
        self.name_list = os.listdir(os.path.join(data_root, 'image'))
        # 按文件名前缀(=损失值)升序排序，取末尾 rate 比例(最难的部分)
        self.name_list.sort(key=lambda x:float(x.split('_')[0]))
        self.name_list = self.name_list[-int(rate * len(self.name_list)):]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        image = np.load(os.path.join(self.data_root, 'image', name))
        label = np.load(os.path.join(self.data_root, 'label', name))
        weight = np.load(os.path.join(self.data_root, 'weight', name))
        skel = np.load(os.path.join(self.data_root, 'skel', name))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        weight = torch.from_numpy(weight)
        skel = torch.from_numpy(skel)

        return image, label, weight,skel
    
class CropSegData(Dataset):
    """
    【stage1 训练数据集】纯随机裁剪版(对应训练入口 train())。

    产出四路对齐数据：img(宽窗 1024) / img2(肺窗 500) / label / weight，
    两窗在训练循环里 cat 成 2 通道输入，与网络 in_channel=2 对应。
    每个 __getitem__ 一次性返回 batch_size 个同病例 patch(形状 (B,C,Z,Y,X))，
    配合 DataLoader batch_size=1 使用。
    """
    def __init__(self, file_path, data_root,file_root, batch_size, aug_flag,ifgrad=False):
        """
        stage1 训练集(纯随机裁剪)：整卷读取 -> 随机裁 batch_size 个 128^3 patch。
        与 AirwayHMData 的区别：不做难例挖掘、不输出 skeleton/parsing，
        因为 stage1 还没有任何预测可用来定义"难区"。
        注意：DataLoader 以 batch_size=1 包一层，本类内部一次产出 batch_size 个 patch，
        所以训练循环拿到 (1,B,...) 后要 transpose(0,1) 还原成 (B,1,...)。
        :param file_root: LIB_weight 目录，加载 weight_br.py 预计算的体素加权图
        :param aug_flag:  1=启用随机翻转/旋转增强(对每个 patch 独立做)
        """
        super(CropSegData, self).__init__()
        self.file_list = load_json_file(file_path)
        self.data_root = data_root
        self.file_root = file_root
        self.batch_size = batch_size   # 每个病例每轮采出的 patch 个数(注意与 DataLoader 的 batch 无关)
        self.aug_flag = aug_flag
        self.ifgrad = ifgrad           # 遗留参数，类内从未使用

    def __len__(self):
        return len(self.file_list)     # 返回病例数(不是 patch 数)，一个 item = 一个病例的一批 patch

    def crop(self, data_list, crop_size=[128, 128, 128]):
        """
        随机中心裁剪：先随机生成 batch_size 个中心点(距边缘 >= crop_size/2)，
        再把 data_list 中每个数组按同一批中心切块。
        同一批 patch 共享中心集合的采样空间，保证 img/img2/label/weight 逐块对齐。
        """
        _data = data_list[0]
        shape = _data.shape
        # 中心点取值范围避开边缘，防止切片越界(越界切片会得到更小的块)
        random_range = [[crop_size[i]//2, shape[i]-crop_size[i]//2] for i in range(3)]
        random_center = []
        for i in range(self.batch_size):
            z = random.randint(random_range[0][0], random_range[0][1])
            y = random.randint(random_range[1][0], random_range[1][1])
            x = random.randint(random_range[2][0], random_range[2][1])
            random_center.append([z, y ,x])
        out_list = []
        for data in data_list:
            out = []
            for c in random_center:
                z, y, x = c[0], c[1], c[2]
                out.append(data[z-crop_size[0]//2 : z+crop_size[0]//2,
                                y-crop_size[1]//2 : y+crop_size[1]//2,
                                x-crop_size[2]//2 : x+crop_size[2]//2])
            out_list.append(out)
        return out_list

    def process_imgmsk(self, data, mask):
        """
        双窗宽归一化(HU 值域裁剪到 [0,1])：
        - data : 宽窗 [-1024,1024]，保留骨/软组织信息
        - data2: 肺窗 [-1000,500]，突出含气结构(气道)
        注意调用处传入前先做了 img-1024 平移(见 __getitem__)，
        与这里的截断区间组合后等效于对原始 HU 直接截断。
        mask 二值化：>0 全算气道(金标准 parsing 的多标签在这里坍缩成 0/1)。
        """
        data = data.astype(np.float32)
        data2 = data.copy()
        data2[data2 > 500] = 500
        data2[data2 < -1000] = -1000
        data2 = (data2 + 1000) / 1500
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048
        mask = (mask > 0).astype(np.int32).astype(np.float32)


        return data, data2, mask

    def augment(self, data_list):
        """随机增强：50% 概率翻转 + 50% 概率 90° 旋转。
        整个 patch 组(img/img2/label/weight)一起变换，空间对齐不破坏"""
        if random.random() > 0.5:
            result = random_flip(data_list)
            data_list = result
        if random.random() > 0.5:
            result = random_rotate(data_list)
            data_list = result
        return data_list



    def __getitem__(self, item):
        # 一个 item = 一个病例：读整卷 data/mask + 预计算的 LIB_weight，
        # 现场裁出 batch_size 个 patch 打包返回(所以返回张量首维是 patch 数)
        name = self.file_list[item]
        img = sitk.ReadImage(self.data_root+'/data/'+name+ 'data_cut' + '.mha')
        img = sitk.GetArrayFromImage(img)
        img=img-1024
        label = sitk.ReadImage(self.data_root+'/mask/'+name+ 'mask_cut' + '.mha')
        label = sitk.GetArrayFromImage(label)
        # LIB_weight 已 mha 化(与 CT 同几何), 统一走 load_itk_image
        weight, _, _, _ = load_itk_image(os.path.join(self.file_root, 'LIB_weight', name + '.mha'))
        # ===== 原 npy 读法(保留备查) =====
        # weight = np.load(os.path.join(self.file_root, 'LIB_weight', name+'.npy'))

        img, img2, label = self.process_imgmsk(img, label)
        # 加权图随 epoch 随机"锐化"：weight^(U(2,3)) 让高权重体素更突出，
        # 气道外(1-label)恒为 1，保证背景仍有基础监督
        weight = weight ** (np.random.random() + 2) * label + (1 - label)
        data_list = self.crop([img, img2, label, weight])
        if self.aug_flag==1:
            # 逐 patch 独立增强：转置取第 i 个 patch 的四路数据 -> augment -> 写回
            for i in range(len(data_list[0])):
                _aug = self.augment([data_list[j][i] for j in range(len(data_list))])
                for j in range(len(data_list)):
                    data_list[j][i] = _aug[j]
        img, img2, label, weight = data_list[0], data_list[1], data_list[2], data_list[3]
        img = torch.from_numpy(np.array(img))
        img2 = torch.from_numpy(np.array(img2))
        label = torch.from_numpy(np.array(label))
        weight = torch.from_numpy(np.array(weight))

        return img, img2, label, weight, name

class SegValCropData(Dataset):
    """
    【验证集滑窗数据集】把整卷按 cube_size^3、步长 step 切成重叠滑窗序列。
    cube=128 / step=64 => 相邻块 50% 重叠，validation() 里用"概率累加/计数平均"
    融合重叠区，消除接缝。

    与 CropSegData 的关键差异：
    - 不输出 label/weight —— 金标准由 validation() 整卷读取一次，逐 patch 比对无意义
    - 输出 pos(该块的 (xl,xr,yl,yr,zl,zr))，供累加器把预测放回整卷原位
    - __len__ 返回的是【总块数】(所有病例滑窗拼接)，非病例数；
      靠 file_list 中 name 连续重复来判定"一个病例的滑窗是否读完"
    """
    def __init__(self, file_path, data_root, batch_size, cube_size=128, step=64):
        # 【batch_size 是什么】与 DataLoader 的 batch_size 是两个独立参数，不是同一个变量。
        # 这里只用作 crop_pos() 末尾"每病例块数补齐到几的倍数"的基数；
        # DataLoader 的 batch_size 才决定每次迭代实际取几块。
        # 【为什么必须两者相等】validation() 靠 name[0] 判定病例切换、drop_last 丢尾批：
        # 相等时每个 batch 恰好落在同一病例内、且总块数整除 batch 不丢块；
        # 不相等(如补 10 的倍数却按 8 分组)会让 batch 跨病例(B 的块写进 A 的累加器、
        # 尺寸不符切片越界)并丢尾巴，指标失真。想批量推理就把两处同时改成同一个值。
        self.file_list = load_json_file(file_path, folder='0', mode=['val'])
        self.root = data_root
        self.cube_size = cube_size
        self.batch_size = batch_size
        self.step = step
        self.file_dic, self.pos_list = self.crop_pos()
        # 整卷缓存：同一病例的连续多个块只读盘一次(name 变化才重新读)
        self.last_name = ''
        self.img = None

    def __len__(self):
        return len(self.file_list)     # 总块数(病例数 x 每病例块数)

    def crop_pos(self):
        """
        预生成每个病例的全部滑窗坐标，并把 [病例名列表, 块坐标列表] 摊平成
        一维序列(item 级)。末尾 if 分支处理"除不尽"：
        整除时块数 = (L-c)/s+1；有余数时多切一块，且最后一块贴边回缩
        (起点强制 L-c，保证不越界、末端无空洞)。
        while 循环把块数补齐到 batch_size 的整数倍(复制第 0 块凑数)，
        保证 DataLoader drop_last 不会丢病例尾部。
        """
        file_dic = {}
        cube_size, step = self.cube_size, self.step
        for f in self.file_list:
            tmp = []
            img = sitk.ReadImage(self.root+'/data/'+f+ 'data_cut' + '.mha')
            x = sitk.GetArrayFromImage(img)
            x = x[np.newaxis, ...]     # 补通道维 -> (1,Z,Y,X)，下面统一按 4 维算
            # 三轴各自的块数：整除取商，有余数进一位(最后一块贴边)
            xnum = (x.shape[1] - cube_size) // step + 1 if (x.shape[1] - cube_size) % step == 0 else \
                (x.shape[1] - cube_size) // step + 2
            ynum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 else \
                (x.shape[2] - cube_size) // step + 2
            znum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 else \
                (x.shape[3] - cube_size) // step + 2
            for xx in range(xnum):
                xl = step * xx
                xr = step * xx + cube_size
                if xr > x.shape[1]:
                    xr = x.shape[1]
                    xl = x.shape[1] - cube_size
                for yy in range(ynum):
                    yl = step * yy
                    yr = step * yy + cube_size
                    if yr > x.shape[2]:
                        yr = x.shape[2]
                        yl = x.shape[2] - cube_size
                    for zz in range(znum):
                        zl = step * zz
                        zr = step * zz + cube_size
                        if zr > x.shape[3]:
                            zr = x.shape[3]
                            zl = x.shape[3] - cube_size
                        tmp.append([xl, xr, yl, yr, zl, zr])
            while (len(tmp) % self.batch_size) != 0:
                tmp.append(tmp[0])
            file_dic[f] = tmp

        file_list, pos_list = [], []
        # 摊平：每个病例名重复"它的块数"次，与块坐标一一对应
        for f in self.file_list:
            file_list += [f for i in range(len(file_dic[f]))]
            pos_list += file_dic[f]
        self.file_list = file_list
        return file_dic, pos_list

    def crop(self, data, pos):
        """按预生成的坐标切一块。注意命名：x/y/z 对应数组的前/中/后三个空间轴
        (即 Z/Y/X 切片序)，与 validation() 累加器 pred[0,:,xl:xr,yl:yr,zl:zr] 一致"""
        xl, xr, yl, yr, zl, zr = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
        data_crop = data[:, xl:xr, yl:yr, zl:zr]
        return data_crop

    def process_imgmsk(self, data):
        """双窗宽归一化(同 CropSegData，但无 mask 参数——验证不返回金标准)"""
        data = data.astype(np.float32)
        data2 = data.copy()
        data2[data2 > 500] = 500
        data2[data2 < -1000] = -1000
        data2 = (data2 + 1000) / 1500
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048

        return data, data2

    def __getitem__(self, item):
        # 返回单个滑窗块：(2,cube,cube,cube) 两通道 + 病例名 + 该块在整卷中的坐标
        name = self.file_list[item]
        if name != self.last_name:
            # 病例切换才读盘+归一化，同一病例的后续块复用 self.img(顺序遍历的前提)
            img = sitk.ReadImage(self.root+'/data/'+name+ 'data_cut' + '.mha')
            img = sitk.GetArrayFromImage(img)
            img=img-1024
            img, img2 = self.process_imgmsk(img)
            img = np.array([img, img2])   # 堆成 (2,Z,Y,X)，通道维在前便于按轴切片
            self.img = img
            self.last_name = name
        else:
            img = self.img
        img_crop = self.crop(img, self.pos_list[item])
        pos = np.array(self.pos_list[item])

        return torch.from_numpy(img_crop.astype(np.float32)), name, torch.from_numpy(pos)

class SegTestCropData(Dataset):
    """
    【测试集滑窗数据集】与 SegValCropData 几乎完全相同，仅两点差异：
      1) 数据源：mode=['test']、folder='-1'(测试集无金标准，用单独的文件列表)
      2) 消费方：test.py 用它做滑窗推理并保存预测，不参与训练/验证的指标闭环
    滑窗切块、整卷缓存、坐标输出的逻辑与验证集一致，详见 SegValCropData 注释。
    """
    def __init__(self, file_path, data_root, batch_size, cube_size=128, step=64):
        # batch_size 语义与 SegValCropData 相同：仅作物块补齐基数，
        # 必须与 DataLoader 的 batch_size 相等，原因见 SegValCropData.__init__ 注释
        self.file_list = load_json_file(file_path, folder='-1', mode=['test'])
        # self.file_list = load_json_file(file_path, folder='0', mode=['val'])
        self.root = data_root
        self.cube_size = cube_size
        self.batch_size = batch_size
        self.step = step
        self.file_dic, self.pos_list = self.crop_pos() 
        self.last_name = ''
        self.img = None

    def __len__(self):
        return len(self.file_list)     # 总块数(同验证集)

    def crop_pos(self):
        """滑窗坐标生成：逻辑与 SegValCropData.crop_pos 完全相同(整除/进一两种块数、
        末块贴边回缩、补齐到 batch_size 整数倍)，此处为复制而非复用"""
        file_dic = {}
        cube_size, step = self.cube_size, self.step
        for f in self.file_list:
            tmp = []
            img = sitk.ReadImage(self.root+'/data/'+f+ 'data_cut' + '.mha')
            x = sitk.GetArrayFromImage(img)
            x = x[np.newaxis, ...]
            xnum = (x.shape[1] - cube_size) // step + 1 if (x.shape[1] - cube_size) % step == 0 else \
                (x.shape[1] - cube_size) // step + 2
            ynum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 else \
                (x.shape[2] - cube_size) // step + 2
            znum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 else \
                (x.shape[3] - cube_size) // step + 2
            for xx in range(xnum):
                xl = step * xx
                xr = step * xx + cube_size
                if xr > x.shape[1]:
                    xr = x.shape[1]
                    xl = x.shape[1] - cube_size
                for yy in range(ynum):
                    yl = step * yy
                    yr = step * yy + cube_size
                    if yr > x.shape[2]:
                        yr = x.shape[2]
                        yl = x.shape[2] - cube_size
                    for zz in range(znum):
                        zl = step * zz
                        zr = step * zz + cube_size
                        if zr > x.shape[3]:
                            zr = x.shape[3]
                            zl = x.shape[3] - cube_size
                        tmp.append([xl, xr, yl, yr, zl, zr])
            while (len(tmp) % self.batch_size) != 0:
                tmp.append(tmp[0])
            file_dic[f] = tmp

        file_list, pos_list = [], []
        for f in self.file_list:
            file_list += [f for i in range(len(file_dic[f]))]
            pos_list += file_dic[f]
        self.file_list = file_list
        return file_dic, pos_list

    def crop(self, data, pos):
        # 按坐标切一块(同 SegValCropData.crop，x/y/z 即数组前/中/后三轴)
        xl, xr, yl, yr, zl, zr = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
        data_crop = data[:, xl:xr, yl:yr, zl:zr]
        return data_crop

    def process_imgmsk(self, data):
        # 双窗宽归一化(同验证集：宽窗 [-1024,1024] + 肺窗 [-1000,500])
        data = data.astype(np.float32)
        data2 = data.copy()
        data2[data2 > 500] = 500
        data2[data2 < -1000] = -1000
        data2 = (data2 + 1000) / 1500
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048

        return data, data2

    def __getitem__(self, item):
        # 单块滑窗 + 病例名 + 坐标(同验证集：name 变化才重新读整卷，连续块复用缓存)
        name = self.file_list[item]
        if name != self.last_name:
            img = sitk.ReadImage(self.root+'/data/'+name+ 'data_cut' + '.mha')
            img = sitk.GetArrayFromImage(img)
            img=img-1024
            img, img2 = self.process_imgmsk(img)
            img = np.array([img, img2])
            self.img = img
            self.last_name = name
        else:
            img = self.img
        img_crop = self.crop(img, self.pos_list[item])
        pos = np.array(self.pos_list[item])

        return torch.from_numpy(img_crop.astype(np.float32)), name, torch.from_numpy(pos)



if __name__ == '__main__':
    # # npy2niigz()
    # p1 = nibabel.load('./data/pred2/ATM_001_0000.nii.gz')
    # p1 = p1.get_fdata()
    # p2 = nibabel.load('./data/pred_1/ATM_001_0000.nii.gz')
    # p2 = p2.get_fdata()[0]
    # print(p1.shape, p2.shape, 2 * (p1 * p2).sum() / (p1 + p2).sum())
    print(1)
