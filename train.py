# -*- coding: utf-8 -*-
"""
SE-UNet 气道分割：三阶段级联训练主程序
====================================
整体流程见文件末尾 __main__，三个阶段依次递进：
  stage1(train)   : 只用随机裁剪 patch 学“什么是气道” -> 产出 pred_1
  stage2(train2)  : 用 pred_1 定位漏检区域做难例挖掘   -> 产出 pred_2 + BR 权重
  stage3(train3)  : 在 pred_2 基础上额外优化“断点”区域(额外输入 skel 与 br_skel 权重)
每个 epoch 内部都是“两遍训练”：先跑主 dataloader，再把本 epoch 产生的难例写入
online_savepath 磁盘缓存，然后用第二个 dataloader 把这些难例再训一遍。
"""
import numpy as np
import os
import argparse   # 命令行参数：--root_dir 指定总输出根目录
import torch
import re
import time
import cc3d
import skimage.measure as measure
import SimpleITK as sitk
import bisect
import nibabel
import shutil
from torch.utils.data import DataLoader
from SE_UNet import SE_UNet
from data import AirwayHMData, OnlineHMData,OnlineHMData3,AirwayHMData3, CropSegData, SegValCropData
from save_gradients import save_gradients_tw
from weight_br import save_s2_pred,save_weight_break
from metrics import *
from util import *
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import datetime



def double_threshold_iteration(pred, h_thresh, l_thresh):
    """
    双阈值迭代（DTI）：气道分割后处理，用于把被阈值“腰斩”的细枝重新接上。

    思想：单独用一个高阈值(0.5)二值化会漏掉置信度 0.35~0.5 的细小末梢；
    但如果直接把阈值降到 0.35，又会引入大量孤立假阳。
    折中方案：只允许“低置信度体素”从“高置信度种子”出发逐层向外生长，
    即受约束的区域生长（等价于 scipy.ndimage.binary_propagation）。

    :param pred: 滑窗平均后的概率图, 取值 [0,1], 形状 (Z,Y,X)
    :param h_thresh: 高阈值, 决定种子区(确定是气道的核心)
    :param l_thresh: 低阈值, 决定生长区(可能是气道的候选)
    :return: 二值化后的预测图, 取值 {0,1}
    """
    # 26 邻域偏移表(3x3x3 去掉中心自身)：用于判断“是否与已生长区域相邻”
    neigb = np.array(
    [[-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0], [0, 1, 0], [1, -1, 0], [1, 0, 0], [1, 1, 0], [-1, -1, -1],
    [-1, 0, -1], [-1, 1, -1], [0, -1, -1], [0, 0, -1], [0, 1, -1], [1, -1, -1], [1, 0, -1], [1, 1, -1],
    [-1, -1, 1], [-1, 0, 1], [-1, 1, 1], [0, -1, 1], [0, 0, 1], [0, 1, 1], [1, -1, 1], [1, 0, 1],[1, 1, 1]])
    h, w,z = pred.shape
    # 把 [0,1] 概率重新缩放到 [0,255]，仅为下面比较时写起来直观（不改变大小关系）
    pred = np.array(pred*255, dtype=np.float32)
    # 种子集 bin：概率 >= 高阈值的体素直接判定为气道
    bin = np.where(pred >= h_thresh*255, 255, 0).astype(np.float32)
    gbin = bin.copy()          # gbin = 当前生长结果(grown binary)，初值为种子集
    gbin_pre = gbin-1          # 上一轮快照；-1 只是造一个“必然不等”的初值，让 while 至少进入一次
    # 注意(已知缺陷)：循环体内 gbin_pre = gbin 是【引用赋值】而非拷贝，
    #   两者指向同一对象，因此 gbin_pre.all() != gbin.all() 恒为 False，
    #   整个区域生长实际只向外扩展【一层】就退出，并未真正“迭代”。
    #   若要正确迭代需改成 gbin_pre = gbin.copy()（但修好后会多出更多假阳）。
    while(gbin_pre.all() != gbin.all()):
        gbin_pre = gbin
        for i in range(h):              # 纯 Python 三重循环，遍历整卷(数千万体素)，极慢
            for j in range(w):
                for k in range(z):
                    # 仅处理“候选体素”：当前未入选 且 概率落在 [低阈值, 高阈值) 之间
                    if gbin[i][j][k] == 0 and pred[i][j][k] < h_thresh*255 and pred[i][j][k] >= l_thresh*255:
                        for n in range(0, 26):  # 查 26 邻域是否已有入选体素
                            inn = i + neigb[n, 0]
                            jnn = j + neigb[n, 1]
                            knn = k + neigb[n, 2]
                            # max(min(...)) 做边界夹持，避免索引越界（等价于边缘重复填充）
                            if gbin[max(min(inn,h-1),0)][max(min(jnn,w-1),0)][max(min(knn,z-1),0)]:
                                gbin[i][j][k] = 255   # 与种子连通 -> 吸收进来
                                break                 # 只要有一个邻居入选即可，不必查完 26 个

    return gbin/255   # 还原成 {0,1} 的二值图

def dice_loss(pred, target):
    """
    标准 Soft Dice 损失（仅 stage1 使用，见 train()）。
    Dice = 2|A∩B| / (|A|+|B|)，两个体积项相等时取 1；用 1-Dice 作为可最小化的损失。
    smooth=1 为平滑项，防止 pred 与 target 全为 0 时 0/0 得到 NaN。
    view(-1) 把整张 3D patch 拉平成一维，即“对整个 patch 算一个全局 Dice”，
    而不是逐体素平均——这是区域损失(region-based)与像素损失的关键区别。
    """
    smooth = 1.
    iflat = pred.view(-1)      # 拉平预测(已经过 sigmoid，取值 0~1)
    tflat = target.view(-1)    # 拉平标注(0/1)
    intersection = ((iflat) * tflat).sum()   # 乘积求和 = 软交集(同时为正的体素量)

    return 1 - ((2. * intersection + smooth) / ((iflat).sum() + (tflat).sum() + smooth))

def general_union_loss_lib(pred, target, weight):
    """
    加权根号 Tversky 损失（GUL, General Union Loss）——本项目的主损失函数。

    与标准 Dice 的关系：Dice 的分母是 |P|+|T|（等价 FP 与 FN 同权重）；
    Tversky 把分母拆成 alpha*FP + beta*FN，从而可以不对称地偏向少漏检或少误检。

    :param weight: 逐体素重要性权重，来自 LIB_weight(似然图) 的幂次变换，
                   在分支点/细小气道处很大、背景处为 1。
                   注意：weight 同时乘在分子与分母上，所以它改变的是
                   “哪些体素的错误更贵”，而不是整体缩放。

    偏置方向(重要)：alpha=0.2 时 FP 权重 0.2、FN 权重 beta=0.8，
    即“漏检一个真阳”的惩罚是“多预测一个假阳”的 4 倍，
    再叠加 weight 在标注内可达数十倍，会强烈鼓励模型过度预测。
    """
    smooth = 1.0  # 平滑项，防止 0/0 得到 NaN
    alpha = 0.2  # alpha=0.1 in stage1 and 0.2 in stage2  # 乘在 pred 上 -> 惩罚假阳(FP)
    beta = 1 - alpha          # 乘在 target 上 -> 惩罚假阴(FN)
    sigma1 = 0.0001
    sigma2 = 0.0001
    # weight_i: 给 target 为 0 的位置补一个极小值，避免下面 **0.7 时出现 0 的幂次不稳定
    weight_i = target * sigma1 + (1 - target) * sigma2
    # 分子：加权交集，但对 pred 做了 **0.7 的凹变换 -> 压低高置信区间的边际收益，
    #      等价于把梯度更多地分给“还没预测出来”的区域（根号 Tversky 的核心技巧）
    intersection = (weight * ((pred + weight_i) ** 0.7) * target).sum()
    # 分母：alpha*|P| + beta*|T| 的加权和（注意它不是严格的 FP/FN，而是含 TP 的体积形式）
    intersection2 = (weight * (alpha * pred + beta * target)).sum()
    return 1 - (intersection + smooth) / (intersection2 + smooth)

def atr_loss(pred, target, skel,weight):
    """
    断点/骨架损失（Break loss）——stage3 专用，用于修复气道被“截断”的位置。

    关键细节：函数内部把 target 直接覆盖成 skel，并把 pred 先与 skel 相乘，
    意味着这个损失【只看骨架上的体素】，骨架之外的一切预测完全不进入计算。
    因此它是一个纯召回项：只要求“骨架路径上必须预测为 1”，
    对骨架外多预测出的假阳没有任何惩罚（这是模型过预测的原因之一）。

    :param skel: 气道中心线骨架(0/1)，由 skeletonize_3d 得到
    :param weight: stage3 中额外融合了 BR_weight(断点处高权重)，使断点位置惩罚更重
    """
    smooth = 1.0
    target=skel               # 丢弃传入的 label，改为只在骨架上算
    pred=pred*skel            # 把预测限制在骨架范围内（骨架外一律置 0，不参与）
    intersection = (weight * pred* target).sum()    # 骨架上被正确预测的部分
    intersection2 = (weight * (pred + target)).sum()  # 体积和形式，等价 alpha=beta=0.5 的 Tversky
    return 1 - (intersection + smooth) / (intersection2 + smooth)

def save_data_online(path, image, label, weight, names, limits=1500):
    """
    在线难例挖掘（Online Hard Example Mining）的磁盘缓存写入，stage2 使用。

    机制：每个 patch 前向后单独算一次它的 GUL 损失，并把该损失值当作【文件名】
    存到磁盘（见调用处 names = str(loss) + '_' + str(iter) + '.npy'）。
    损失越大 = 该 patch 预测越差 = 越值得反复训练。
    目录内最多保留 limits 个 patch，超出时删掉损失最小(文件名数值最小)的那一个，
    相当于用一个“按损失排序的定长集合”做难例筛选（下一 epoch 由 OnlineHMData 读回再训）。

    :param limits: 缓存上限，调用处传 int(len(train_dataset)*batch_size_per_case*0.3)，
                   即只保留最难的前 30% patch
    """
    # 列出目录中已有的难例文件名，并按文件名前缀(损失值 float)升序排序 -> 最易的在前
    name_list = os.listdir(os.path.join(path, 'image'))
    name_list.sort(key=lambda x:float(x.split('_')[0]))
    # GPU tensor 转 numpy 才能落盘；detach() 断开计算图，否则会连带存下梯度
    image = image.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    weight = weight.detach().cpu().numpy()
    dice_list = [float(n.split('_')[0]) for n in name_list]   # 与 name_list 对齐的损失值(升序)，当前缓存中存在的损失值列表
    for i in range(image.shape[0]):
        if len(name_list) < limits:
            # 缓存未满：三个数组(image/label/weight)分别落盘，同名不同子目录
            np.save(os.path.join(path, 'image', names[i]), image[i]) # image[i](C,Z,Y,X)
            np.save(os.path.join(path, 'label', names[i]), label[i].astype(np.int8))  # 0/1 用 int8 省空间
            np.save(os.path.join(path, 'weight', names[i]), weight[i])
            # bisect 在升序列表中找插入位置，保持列表始终有序(相当于插入排序维护)
            index = bisect.bisect(dice_list, float(names[i].split('_')[0]))
            name_list.insert(index, names[i])
            dice_list.insert(index, float(names[i].split('_')[0]))
        else:
            # 缓存已满：先看新样本够不够“难”
            index = bisect.bisect(dice_list, float(names[i].split('_')[0]))
            if index == 0: continue   # 插入位置为 0 = 损失比全目录都小 = 不够难，直接丢弃
            # 够难则插入，并淘汰当前损失最小(name_list[0])的那个，维持上限
            name_list.insert(index, names[i])
            dice_list.insert(index, float(names[i].split('_')[0]))
            os.remove(os.path.join(path, 'image', name_list[0]))
            os.remove(os.path.join(path, 'label', name_list[0]))
            os.remove(os.path.join(path, 'weight', name_list[0]))
            name_list = name_list[1:]        # 同步截断列表头部（注意：这里裁掉的是刚插入的旧最小项）
            dice_list = dice_list[1:]
            np.save(os.path.join(path, 'image', names[i]), image[i])
            np.save(os.path.join(path, 'label', names[i]), label[i].astype(np.int8))
            np.save(os.path.join(path, 'weight', names[i]), weight[i])

def save_data_online3(path, image, label, weight, skel,names, limits=1500):
    """
    save_data_online 的 stage3 版本：唯一区别是多缓存一路 skel(骨架)。
    因为 stage3 的损失含 atr_loss，需要骨架作为额外监督信号，
    难例被读回再训时必须连骨架一起提供，否则 break_loss 无从计算。
    其余“以损失值为文件名 + bisect 维护定长难例集合”的逻辑完全相同。
    """
    name_list = os.listdir(os.path.join(path, 'image'))
    name_list.sort(key=lambda x:float(x.split('_')[0]))
    image = image.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    weight = weight.detach().cpu().numpy()
    skel = skel.detach().cpu().numpy()   # 骨架也转成 numpy 落盘
    dice_list = [float(n.split('_')[0]) for n in name_list]
    for i in range(image.shape[0]):
        if len(name_list) < limits:
            np.save(os.path.join(path, 'image', names[i]), image[i])
            np.save(os.path.join(path, 'label', names[i]), label[i].astype(np.int8))
            np.save(os.path.join(path, 'weight', names[i]), weight[i])
            np.save(os.path.join(path, 'skel', names[i]), skel[i].astype(np.int8))
            index = bisect.bisect(dice_list, float(names[i].split('_')[0]))
            name_list.insert(index, names[i])
            dice_list.insert(index, float(names[i].split('_')[0]))
        else:
            index = bisect.bisect(dice_list, float(names[i].split('_')[0]))
            if index == 0: continue
            name_list.insert(index, names[i])
            dice_list.insert(index, float(names[i].split('_')[0]))
            os.remove(os.path.join(path, 'image', name_list[0]))
            os.remove(os.path.join(path, 'label', name_list[0]))
            os.remove(os.path.join(path, 'weight', name_list[0]))
            os.remove(os.path.join(path, 'skel', name_list[0]))
            name_list = name_list[1:]
            dice_list = dice_list[1:]
            np.save(os.path.join(path, 'image', names[i]), image[i])
            np.save(os.path.join(path, 'label', names[i]), label[i].astype(np.int8))
            np.save(os.path.join(path, 'weight', names[i]), weight[i])
            np.save(os.path.join(path, 'skel', names[i]), skel[i].astype(np.int8))

def train3(data_root, model_savepath, online_savepath, log_savepath, pred2_path,br_skel_path,BR_weight_path,aug, DTI,start_model, start_epoch, file_path, file_root, gpu):
    """
    阶段 3 训练：在 stage2 结果之上专门修复“气道断点”。

    相比 stage2 多引入两路由 stage2 预测算出的监督信号（由 weight_br.py 生成）：
      BR_weight_path -> 断点处的高重要性权重，融进 weight 里放大该处惩罚
      br_skel_path   -> 断点位置坐标，供 Dataset 做“断点裁剪”定向采样
    因此 stage3 的损失比 stage2 多一项 break_loss(atr_loss)。

    :param data_root     : 预处理后数据根目录(含 data/ 与 mask/ 子目录)
    :param model_savepath: 本阶段权重输出目录, 如 ./saved_model/stage_three
    :param online_savepath: 在线难例挖掘的磁盘缓存目录(每 epoch 重建)
    :param log_savepath  : 验证汇总 CSV 日志, 如 ./LOG/log_stage_three.csv
    :param pred2_path    : stage2 的预测结果目录(作为“已做对的区域”参考)
    :param aug           : 1=开启数据增强(随机翻转), 0=关闭
    :param DTI           : 1=验证时用双阈值后处理, 0=用固定 0.5 阈值
    :param start_model   : 初始权重所在目录(上一阶段 stage_two)
    :param start_epoch   : 加载上一阶段第几个 epoch 的权重
    :param file_path     : 病例划分 json, 如 ./data/base_dict.json
    :param file_root     : 中间产物根目录(./data), 存 skeleton/tree_parse/LIB_weight 等
    :param gpu           : 逗号分隔的 GPU 编号字符串, 如 '0,1'
    """
    stage=3                 # 标记当前阶段, 传给 validation() 决定是否计算 hard 指标
    max_epoches = 50         # 总 epoch 数
    batch_size_per_case = 2   # 【注意】这是传给 Dataset 的“每个病例内部采几个 patch”，
                             # 不是 DataLoader 的 batch_size！见下方 transpose(0,1) 说明
    aug_flag = aug

    # ===== 训练超参数集中定义(便于统一调参与复现) =====
    cube_size = 128                  # 训练裁剪块边长
    dataloader_batch_size = 1        # 训练 DataLoader 的 batch(必须保持 1，真实 batch 由 batch_size_per_case 决定)
    num_workers = 0                  # DataLoader 子进程数，0=主进程读
    learning_rate = 0.0001           # AdamW 初始学习率
    lr_milestones = [10, 20, 40]     # MultiStepLR 衰减点(每 epoch 只 step 一次, 计数即真实 epoch 号)
    lr_gamma = 0.1                   # 衰减系数
    hm_limits_ratio = 0.3            # 难例缓存上限比例 = 病例数 x batch x 0.3
    hm_rate = 1.0                    # 第二遍 OnlineHMData 采样比例(1.0 = 全部读回)
    hm_dataloader_batch_size = 2     # 第二遍难例 DataLoader 的 batch
    val_cube_size = 128              # 验证滑窗块边长
    val_step = 64                    # 验证滑窗步长(重叠一半)
    val_batch_size_per_case_and_dataloader = 1   # 验证 Dataset 补齐基数(per_case) and DataLoader batch 必须相等，共用这一个变量

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu   # 必须在任何 .cuda() 之前设置才会生效

    # file_root='./data'

    # TensorBoard 日志目录：用“阶段名 + 月_日_时_分”保证每次启动是独立 run，不会混曲线
    current_date = datetime.datetime.now()
    # TensorBoard run 目录不再硬编码 './LOG/' + split('./saved_model/')，
    # 而是从 log_savepath 所在目录(即 root_dir/LOG)和 model_savepath 末级目录名(stage_x)派生，
    # 这样无论用户指定什么总输出根目录，TensorBoard 日志都会落在该根目录的 LOG 子目录下
    # 取 log_savepath(日志文件完整路径)所在目录作为 TensorBoard 日志根目录；
    # 若用户只传了裸文件名(不含目录部分)，dirname 会返回空字符串，此时回退为当前目录 '.'
    tb_dir = os.path.dirname(log_savepath) if os.path.dirname(log_savepath) != '' else '.'
    os.makedirs(tb_dir, exist_ok=True)
    log_root = os.path.join(tb_dir, os.path.basename(os.path.normpath(model_savepath)) + '_{}_{}_{}_{}'.format(current_date.month, current_date.day,
                                                                      current_date.hour,
                                                                      current_date.minute))
    writer = SummaryWriter(str(log_root))

    # 训练 loss 逐迭代落 CSV(与 TensorBoard 同源记录)：无 TensorBoard 环境时
    # 可直接用 Excel/pandas 打开画 loss 曲线；追加模式，文件不存在时先写表头
    loss_csv_path = os.path.splitext(log_savepath)[0] + '_train_losses.csv'
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, 'a') as csv_f:
            csv_f.writelines('epoch,iter,step,loss,GUL_encode,GUL_decode,break_loss\n')

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)   # 多进程下各 worker 随机种子独立（当前 num_workers=0，未生效）

    model = SE_UNet(in_channel=2, n_classes=1)   # 2 通道输入(两种窗宽归一化), 1 通道输出(前景概率)
    train_dataset = AirwayHMData3(file_path=file_path,
                                  data_root=data_root,
                                  file_root=file_root,
                                  pred2_path=pred2_path,
                                  br_skel_path=br_skel_path,
                                  BR_weight_path=BR_weight_path,
                                  batch_size=batch_size_per_case,
                                  cube_size=cube_size,
                                  aug_flag=aug_flag)
    # 【关键设计】为什么 DataLoader 的 batch_size 写死 1，而 Dataset 里却有 batch_size_per_case？
    # AirwayHMData3.__getitem__ 内部用 for _ in range(self.batch_size) 在【同一个病例】上
    # 连续采多个 patch（保证一个 batch 来自同一病例，难例采样逻辑才有意义），
    # 返回形状 (B,128,128,128)。DataLoader 再叠一维 -> (1,B,128,128,128)，
    # 随后用 transpose(0,1) 换成 (B,1,128,128,128)，从而“伪装”成 batch=B。
    # 所以：想增大真实 batch，只改上面那个 batch_size_per_case，这里的 1 必须保持不动。
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=dataloader_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,      # 0=主进程读数据，避免 Windows 多进程开销与随机态问题
                                   pin_memory=True,
                                   drop_last=True)
    # 验证集：SegValCropData 做滑窗切块(step=64 重叠一半)，一个病例会被拆成多条样本，
    # validation() 内部靠 name 变化来判断“这个病例的滑窗是否全部跑完”
    # 注意：Dataset 的 batch_size 只是"每病例块数补齐的基数"，与下面 DataLoader 的
    # batch_size 是两个独立参数；两者必须相等，否则 batch 跨病例 + drop_last 丢尾块。
    # 想一次推理多块提速，把两处同时改成同一个值(如都=8)。详见 SegValCropData.__init__ 注释。
    valid_dataset = SegValCropData(file_path,
                                   data_root,
                                   batch_size=val_batch_size_per_case_and_dataloader,
                                   cube_size=val_cube_size,
                                   step=val_step)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=val_batch_size_per_case_and_dataloader,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    # 注意：train_dataset 的 __len__ 返回的是【病例数】而非 patch 数，
    # 所以 max_step 实际是“每 epoch 的迭代次数 = 病例数”，总更新步数非常少(十几例 x 51 epoch)
    max_step = len(train_dataset) * max_epoches
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)   # AdamW: 权重衰减与梯度解耦，比 Adam 正则更干净
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=lr_milestones,
                                                        gamma=lr_gamma)  # 到达 milestones 时把学习率乘 gamma（每 epoch 只 step 一次，衰减点即真实 epoch 号）

    # 【超参数落盘】把本阶段训练+验证用到的全部超参数写成 JSON，存进权重目录，便于复现与追溯
    save_hyperparams(model_savepath, 'hyperparams_stage_three.json', {
        'stage': 3,
        'stage_说明': '训练阶段编号(3=第三阶段)',
        'model': {'arch': 'SE_UNet', 'in_channel': 2, 'n_classes': 1,
                  '说明': 'SE_UNet; in_channel=输入通道数(两种窗宽归一化), n_classes=输出通道数(前景概率)'},
        'train': {
            'max_epoches': max_epoches,
            'max_epoches_说明': '总 epoch 数',
            'batch_size_per_case': batch_size_per_case,
            'batch_size_per_case_说明': 'Dataset 内部每病例采几个 patch(真实 batch)',
            'dataloader_batch_size': dataloader_batch_size,
            'dataloader_batch_size_说明': 'DataLoader 的 batch, 必须保持 1, 真实 batch 由 batch_size_per_case 决定(见 AirwayHMData3 注释)',
            'num_workers': num_workers,
            'num_workers_说明': 'DataLoader 子进程数, 0=主进程读',
            'cube_size': cube_size,
            'cube_size_说明': '训练裁剪块边长',
            'aug': aug,
            'aug_说明': '1=开启随机翻转增强',
            'optimizer': 'AdamW',
            'optimizer_说明': '优化器类型, 初始学习率见 lr',
            'lr': learning_rate,
            'lr_说明': 'AdamW 初始学习率',
            'scheduler': 'MultiStepLR',
            'scheduler_说明': '学习率调度器, 到达 milestones 时把学习率乘 gamma',
            'milestones': lr_milestones,
            'milestones_说明': 'MultiStepLR 衰减点(epoch)',
            'gamma': lr_gamma,
            'gamma_说明': '学习率衰减系数',
            'loss': 'general_union_loss_lib(alpha=0.2) + atr_loss(break_loss, 仅骨架区域, weight 融合 BR_weight)',
            'loss_说明': '损失函数配置',
            'online_hard_mining': {
                'enabled': True,
                'cache_dir': online_savepath,
                'limits_ratio': hm_limits_ratio,
                'limits_ratio_说明': '缓存上限 = 病例数 x batch x 该比例, 只留最难的前 30%',
                'dataloader_batch_size': hm_dataloader_batch_size,
                'dataloader_batch_size_说明': '第二遍难例回放的 DataLoader 真实 batch(每个 item 是磁盘上单个难例 patch)',
                'cached_channels': ['image', 'label', 'weight', 'skel'],
                'cached_channels_说明': '难例缓存保存的通道',
            },
            'start_model': start_model,
            'start_model_说明': '热启动来源(stage2 权重目录)',
            'start_epoch': start_epoch,
            'start_epoch_说明': '热启动使用的 epoch 号',
        },
        'validation': {
            'DTI': DTI,
            'DTI_说明': '1=双阈值后处理, 0=固定 0.5 阈值',
            'cube_size': val_cube_size,
            'cube_size_说明': '验证滑窗块边长',
            'step': val_step,
            'step_说明': '验证滑窗步长(重叠一半)',
            'batch_size': val_batch_size_per_case_and_dataloader,
            'batch_size_说明': 'SegValCropData 补齐基数(per_case) and DataLoader batch, 两者必须相等(共用这一个值)',
            'fixed_threshold': 0.5,
            'fixed_threshold_说明': 'DTI=0 时使用的固定阈值',
            'dti_h_thresh': 0.5,
            'dti_h_thresh_说明': 'DTI=1 时种子阈值(见 validation())',
            'dti_l_thresh': 0.4,
            'dti_l_thresh_说明': 'DTI=1 时生长候选阈值(见 validation())',
        },
        'data': {'data_root': data_root, 'file_path': file_path, 'file_root': file_root,
                 'pred2_path': pred2_path, 'br_skel_path': br_skel_path,
                 'BR_weight_path': BR_weight_path,
                 '说明': 'data_root=数据根目录, file_path=病例列表, file_root=数据子目录; pred2_path/br_skel_path/BR_weight_path=stage2 预测、骨架与权重图路径'},
        'paths': {'model_savepath': model_savepath, 'online_savepath': online_savepath,
                  'log_savepath': log_savepath, 'tensorboard_logdir': log_root,
                  '说明': 'model_savepath=权重目录, online_savepath=难例缓存目录, log_savepath=训练日志, tensorboard_logdir=TensorBoard 目录'},
        'gpu': gpu,
        'gpu_说明': 'CUDA_VISIBLE_DEVICES 指定的 GPU 编号',
    })

    # resume
    # weights_dict = torch.load(os.path.join(model_savepath ,'SE_UNet_9.pth'))
    # 从【上一阶段】的权重热启动：stage3 用 stage2 选出的最佳 epoch 权重
    weights_dict = torch.load(start_model +
                              'SE_UNet_{}.pth'.format(start_epoch))
    model.load_state_dict(weights_dict, strict=True)   # 严格加载: 键不匹配立即报错, 避免静默漏加载
    model = torch.nn.DataParallel(model).cuda()          # 包装成多卡并行；注意存权重时要用 model.module 去掉前缀
    model.train()
    starttime = time.time()
    val_td_list=[]
    val_bd_list=[]
    val_loss_random=[]
    val_loss_hard=[]
    for ep in range(max_epoches):
        # 每 epoch 开头重建难例缓存目录：本 epoch 采到的难例只服务于本 epoch 的第二遍训练。
        # 注意 shutil.rmtree 会先删掉上一轮数据，若此时断电，本轮难例将彻底丢失且无法恢复。
        if os.path.exists(online_savepath):
            shutil.rmtree(online_savepath)   # 递归删除整个目录(含子目录内文件)
            os.mkdir(online_savepath)
            os.mkdir(online_savepath + '/image')
            os.mkdir(online_savepath + '/label')
            os.mkdir(online_savepath + '/weight')
            os.mkdir(online_savepath + '/skel')
        else:
            os.mkdir(online_savepath)
            os.mkdir(online_savepath + '/image')
            os.mkdir(online_savepath + '/label')
            os.mkdir(online_savepath + '/weight')
            os.mkdir(online_savepath + '/skel')

        # ============ 第一遍：主 dataloader 训练 ============
        # pack 顺序对应 AirwayHMData3.__getitem__ 返回的 data_list:
        #   [img_crops, img2_crops, label_crops, weight_crops, skel_crops]
        # ==== DataLoader 形状链 (train_data_loader <- AirwayHMData3, stage3 第一遍) ====
        # Dataset.__getitem__: 每个 item 读一个病例整卷, 内部裁 batch_size_per_case(记为 B) 个 patch,
        #   五路各返回 (B, cube_size, cube_size, cube_size) —— 4 维, 没有通道维
        # DataLoader(batch_size=dataloader_batch_size=1) collate 在最前再套一维:
        #   pack[j].shape = (dataloader_batch_size, B, cube_size, cube_size, cube_size)
        #   例: B=2, cube_size=128 -> pack[0].shape = (1, 2, 128, 128, 128)
        # transpose(0,1) 后: (B, 1, cube_size, cube_size, cube_size), 中间的 1 即通道维 C
        # cat 两窗后 data: (B, 2, cube_size, cube_size, cube_size), 例 (2, 2, 128, 128, 128)
        for iter, pack in enumerate(train_data_loader):
            data = pack[0].float().cuda()     # (1,B,cube,cube,cube)=(1,2,128,128,128) 窗宽1(肺窗)归一化 patch
            data2 = pack[1].float().cuda()    # 同上形状, 窗宽2(纵隔窗)归一化 patch
            label = pack[2].float().cuda()    # 同上形状, 二值标注 0/1
            weight = pack[3].float().cuda()   # 同上形状, 逐体素重要性权重(LIB_weight 的幂次)
            skel = pack[4].float().cuda()     # 同上形状, 骨架, 供 atr_loss 定位断点

            # 维度变换：(1,B,Z,Y,X) -> (B,1,Z,Y,X)，把 Dataset 内部采出的 B 个 patch
            # 从“被 DataLoader 多包的一维”里解放出来，还原成真正的 batch 维(中间的 1 即通道维 C)
            data = data.transpose(0, 1)
            data2 = data2.transpose(0, 1)
            label = label.transpose(0, 1)
            weight = weight.transpose(0, 1)
            skel = skel.transpose(0, 1)

            # 两个窗宽沿通道维拼成 2 通道输入，匹配 SE_UNet(in_channel=2)
            data = torch.cat([data, data2], dim=1)

            # 【AMP bf16】前向+损失包进 autocast：conv 走 bf16(Tensor Core)省显存提速，
            # InstanceNorm/sum/pow 等归约类算子由 autocast 自动保持 fp32；
            # A4000 为 Ampere(cc8.6)支持 bf16，指数范围与 fp32 相同，无需 GradScaler。
            # backward/optimizer.step 保持原样(master weights 与 AdamW 状态仍是 fp32)。
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # 模型是深监督结构：返回编码端与解码端两个预测头，都参与损失
                pred_en, pred_de = model(data)
                pred_en = torch.sigmoid(pred_en)   # 网络输出的是 logits，必须 sigmoid 才能当概率
                pred_de = torch.sigmoid(pred_de)

                dice_loss_en = general_union_loss_lib(pred_en, label, weight)
                dice_loss_de = general_union_loss_lib(pred_de, label, weight)

                # 断点损失：编码端与解码端各算一次再相加
                break_loss = atr_loss(pred_en, label, skel, weight) + atr_loss(
                    pred_de, label, skel, weight)
                # 总损失：解码端权重 1(主输出)，编码端 0.5(辅助深监督)，断点 0.5
                loss =  dice_loss_de * 1 + dice_loss_en * 0.5 + break_loss * 0.5  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 逐 patch 单独算一次损失，把【损失值本身当作文件名】存盘：
            #   文件名 = "<该patch的GUL损失>_<iter序号>.npy"，损失越大 = 越难 = 越值得再训
            # 这一步是 O(B) 次额外前向式计算，只为拿到排序依据，不参与反向传播
            names = [
                general_union_loss_lib(pred_de[i], label[i], weight[i])
                for i in range(data.shape[0])
            ]
            names = [str(n.item()) + '_' + str(iter) + '.npy' for n in names]
            # 写入定长难例缓存，上限 = 病例数 x batch x 0.3，即只保留最难的前 30%
            save_data_online3(online_savepath,
                             data,
                             label,
                             weight,
                             skel,
                             names,
                             limits=int(len(train_dataset) * batch_size_per_case *
                                        hm_limits_ratio))

            if iter % 10 == 0:   # 每 10 个 iter 打印一次，避免刷屏
                print('epoch:', ep, iter + ep * len(train_dataset), '/',
                      max_step, 'loss:', loss.item(), 'GUL encode:', dice_loss_en.item(),
                      'GUL decode:', dice_loss_de.item(), 'break loss:',
                      break_loss.item())

            # 写 TensorBoard：add_scalars 会把 4 个值画在同一张图的 4 条曲线上
            # 横轴用全局 iter 序号(iter + ep*病例数)，保证跨 epoch 曲线连续
            writer.add_scalars('Train', {'loss': loss.item(),'GUL_encode': dice_loss_en.item(),'GUL_decode': dice_loss_de.item(),'break_loss': break_loss.item()}, iter + ep * len(train_dataset))
            # 同一批数值同步写 CSV，一行 = 一次迭代；step 列与 TensorBoard 横轴一致
            with open(loss_csv_path, 'a') as csv_f:
                csv_f.writelines(','.join([str(ep), str(iter), str(iter + ep * len(train_dataset)),
                                           str(loss.item()), str(dice_loss_en.item()),
                                           str(dice_loss_de.item()), str(break_loss.item())]) + '\n')
        torch.cuda.empty_cache()   # 归还本 epoch 缓存的显存碎片

        # writer.close() 原先写在这里(epoch 循环内)，导致每个 epoch 重建事件文件；已挪到函数末尾统一关闭
        lr_scheduler.step()             # 学习率衰减：本 epoch 唯一一次 step(难例遍后的第 2 次 step 已移除, MultiStepLR 计数=真实 epoch)
        # '''
        # ============ 第二遍：把本 epoch 攒下的难例读回来再训一遍 ============
        # print('start online hard mining: ')
        # rate=1.0 表示读取缓存目录中【全部】难例(而非按比例抽样)（缓存本身已取了limits_ratio比例）
        hm_dataset = OnlineHMData3(data_root=online_savepath,
                                  rate=hm_rate)
        # 原调用还传了 batch_size=hm_dataloader_batch_size，Dataset 内从未使用，已注释去除：
        # hm_dataset = OnlineHMData3(data_root=online_savepath, batch_size=hm_dataloader_batch_size, rate=hm_rate)
        # OnlineHMData3 每个 item 就是磁盘上单个难例 patch(128^3)，
        # 这里 DataLoader 的 batch_size=hm_dataloader_batch_size 就是真实 batch。
        # 第二遍每 epoch 迭代次数 = 缓存难例数。
        hm_dataloader = DataLoader(dataset=hm_dataset,
                                   batch_size=hm_dataloader_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True)
        # ==== DataLoader 形状链 (hm_dataloader <- OnlineHMData3, stage3 第二遍难例) ====
        # Dataset.__getitem__: 每个 item 就是磁盘上单个难例 patch 的 np 数组(第一遍已拼好双窗):
        #   image (2, cube_size, cube_size, cube_size)
        #   label/weight/skel (1, cube_size, cube_size, cube_size)
        # DataLoader(batch_size=hm_dataloader_batch_size) 直接堆出真 batch, 故【不做 transpose(0,1)】:
        #   pack[j].shape = (hm_dataloader_batch_size, C, cube_size, cube_size, cube_size)
        #   例: hm_dataloader_batch_size=2, cube_size=128
        #       data=(2,2,128,128,128)  label/weight/skel=(2,1,128,128,128)
        for iter, pack in enumerate(hm_dataloader):
            # 难例已从磁盘读回, pack 顺序为 [image, label, weight, skel], 无 img2
            data = pack[0].float().cuda()     # (hm_dataloader_batch_size,2,cube,cube,cube)=(2,2,128,128,128)
            label = pack[1].float().cuda()    # (hm_dataloader_batch_size,1,cube,cube,cube)=(2,1,128,128,128)
            weight = pack[2].float().cuda()   # 同 label 形状
            skel = pack[3].float().cuda()     # 同 label 形状
            # 【AMP bf16】难例遍同样只包前向+损失，backward/step 原样(见第一遍注释)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_en, pred_de = model(data)
                pred_en = torch.sigmoid(pred_en)
                pred_de = torch.sigmoid(pred_de)

                break_loss = atr_loss(pred_en, label, skel, weight) + atr_loss(
                    pred_de, label, skel, weight)
                dice_loss_all = general_union_loss_lib(pred_en, label, weight) * 0.5 + \
                                general_union_loss_lib(pred_de, label, weight)
                loss = break_loss * 0.5 + dice_loss_all   # 难例遍：断点与区域损失各占一半
            optimizer.zero_grad()   # 清空上一 iter 累积的梯度(否则梯度会累加)
            loss.backward()         # 反向传播求梯度
            optimizer.step()        # 按 AdamW 规则更新参数



        # '''
        print('')
        # 本 epoch 结束后在验证集上滑窗推理并计算全套指标(TD/BD/DSC/Pre/Sen/Spe)
        TD_mean, BD_mean, loss_random, loss_hard=validation(data_root, model, valid_dataloader, ep, log_savepath, DTI,file_root,stage)
        print('')

        # 累积历史指标供调度器判断趋势：调度器需要“最近几个 epoch”的对比，故用 list 保存
        val_td_list.append(TD_mean)
        val_bd_list.append(BD_mean)
        val_loss_random.append(loss_random)   # 全体体素 Dice
        val_loss_hard.append(loss_hard)       # 剔除 stage1 已做对区域后的 Dice

        # 闭环反馈：用验证结果动态调整下一 epoch 的采样比例 hard_ratio / break_ratio
        train_dataset.update_scheduler(ep, val_loss_random, val_loss_hard, val_td_list, val_bd_list)


        torch.cuda.empty_cache()
        os.makedirs(model_savepath, exist_ok=True)   # 递归建目录，nibabel/torch 不会自动创建父目录
        # 只存权重(state_dict)。用 model.module 是为了去掉 DataParallel 加的 "module." 前缀，
        # 使保存的键与裸 SE_UNet 一致，后续严格加载(strict=True)才不会因前缀错位报错。
        # 【不含 optimizer/scheduler】，因此目前无法真正断点续训。
        torch.save(
            model.module.state_dict(),
            os.path.join(model_savepath, 'SE_UNet_' + str(ep) + '.pth'))
        # print(ep, ':', time.time() - starttime)
    writer.close()   # 训练结束统一关闭 TensorBoard writer(整个 run 只生成一套事件文件)
    sdf = 2   # 无意义的占位变量(原代码遗留)，可删除

def train2(data_root,
           model_savepath,
           online_savepath,
           log_savepath,
           pred_path,
           aug,
           DTI,
           start_model,
           start_epoch,
           file_path,
           file_root,
           gpu='0'):
    """
    阶段 2 训练：引入【加权区域损失 + 在线难例挖掘】，聚焦 stage1 漏检的区域。

    与 stage1 的两处关键差异：
      1) 损失由普通 dice_loss 换成 general_union_loss_lib(带 weight 的加权 Tversky)
      2) 每个 epoch 多跑一遍“难例缓存”训练(OnlineHMData)
    相比 stage3 则少了 break_loss 与 skel 输入。

    :param pred_path : stage1 的预测目录(pred_1)，Dataset 用它定位“漏检区”做难例裁剪
    :param start_model/start_epoch : 从 stage1 的最终权重热启动
    """
    stage=2
    max_epoches = 50
    batch_size_per_case = 2   # 同样是“Dataset 内部每病例采几个 patch”，非 DataLoader batch
    aug_flag = aug

    # ===== 训练超参数集中定义(便于统一调参与复现) =====
    cube_size = 128                  # 训练裁剪块边长
    dataloader_batch_size = 1        # 训练 DataLoader 的 batch(必须保持 1，真实 batch 由 batch_size_per_case 决定)
    num_workers = 0                  # DataLoader 子进程数，0=主进程读
    learning_rate = 0.0001           # AdamW 初始学习率
    lr_milestones = [10, 20, 40]     # MultiStepLR 衰减点(每 epoch 只 step 一次, 计数即真实 epoch 号)
    lr_gamma = 0.1                   # 衰减系数
    hm_limits_ratio = 0.3            # 难例缓存上限比例 = 病例数 x batch x 0.3
    hm_rate = 1.0                    # 第二遍 OnlineHMData 采样比例(1.0 = 全部读回)
    hm_dataloader_batch_size = 2     # 第二遍难例 DataLoader 的 batch
    val_cube_size = 128              # 验证滑窗块边长
    val_step = 64                    # 验证滑窗步长(重叠一半)
    val_batch_size_per_case_and_dataloader = 1   # 验证 Dataset 补齐基数(per_case) and DataLoader batch 必须相等，共用这一个变量

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


    current_date = datetime.datetime.now()   # TensorBoard run 目录(含时间戳，每次启动独立)
    # TensorBoard run 目录不再硬编码 './LOG/' + split('./saved_model/')，
    # 而是从 log_savepath 所在目录(即 root_dir/LOG)和 model_savepath 末级目录名(stage_x)派生，
    # 这样无论用户指定什么总输出根目录，TensorBoard 日志都会落在该根目录的 LOG 子目录下
    # 取 log_savepath(日志文件完整路径)所在目录作为 TensorBoard 日志根目录；
    # 若用户只传了裸文件名(不含目录部分)，dirname 会返回空字符串，此时回退为当前目录 '.'
    tb_dir = os.path.dirname(log_savepath) if os.path.dirname(log_savepath) != '' else '.'
    os.makedirs(tb_dir, exist_ok=True)
    log_root = os.path.join(tb_dir, os.path.basename(os.path.normpath(model_savepath)) + '_{}_{}_{}_{}'.format(current_date.month, current_date.day,
                                                                      current_date.hour,
                                                                      current_date.minute))
    writer = SummaryWriter(str(log_root))

    # 训练 loss 逐迭代落 CSV(与 TensorBoard 同源记录)：无 TensorBoard 环境时
    # 可直接用 Excel/pandas 打开画 loss 曲线；追加模式，文件不存在时先写表头
    loss_csv_path = os.path.splitext(log_savepath)[0] + '_train_losses.csv'
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, 'a') as csv_f:
            csv_f.writelines('epoch,iter,step,loss,dice_loss_encode,dice_loss_decode\n')

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    model = SE_UNet(in_channel=2, n_classes=1)

    train_dataset = AirwayHMData(file_path=file_path,
                                 data_root=data_root,
                                 file_root=file_root,
                                 pred_path=pred_path,
                                 batch_size=batch_size_per_case,
                                 cube_size=cube_size,
                                 aug_flag=aug_flag)
    # 同 stage3 的【关键设计】：Dataset 内 batch_size_per_case 决定每次裁几个同病例 patch(真实 batch)，
    # DataLoader 的 1 必须保持不动(>1 会触发 transpose/cat 的通道维 bug)，
    # drop_last=True 但 __len__=病例数、batch=1，不会丢样本。详见 AirwayHMData 类注释。
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=dataloader_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True)
    # 注意：Dataset 的 batch_size 只是"每病例块数补齐的基数"，与下面 DataLoader 的
    # batch_size 是两个独立参数；两者必须相等，否则 batch 跨病例 + drop_last 丢尾块。
    # 想一次推理多块提速，把两处同时改成同一个值(如都=8)。详见 SegValCropData.__init__ 注释。
    valid_dataset = SegValCropData(file_path,
                                   data_root,
                                   batch_size=val_batch_size_per_case_and_dataloader,
                                   cube_size=val_cube_size,
                                   step=val_step)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=val_batch_size_per_case_and_dataloader,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    max_step = len(train_dataset) * max_epoches
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=lr_milestones,
                                                        gamma=lr_gamma)

    # 【超参数落盘】把本阶段训练+验证用到的全部超参数写成 JSON，存进权重目录，便于复现与追溯
    save_hyperparams(model_savepath, 'hyperparams_stage_two.json', {
        'stage': 2,
        'stage_说明': '训练阶段编号(2=第二阶段)',
        'model': {'arch': 'SE_UNet', 'in_channel': 2, 'n_classes': 1,
                  '说明': 'SE_UNet; in_channel=输入通道数(两种窗宽归一化), n_classes=输出通道数(前景概率)'},
        'train': {
            'max_epoches': max_epoches,
            'max_epoches_说明': '总 epoch 数',
            'batch_size_per_case': batch_size_per_case,
            'batch_size_per_case_说明': 'Dataset 内部每病例采几个 patch(真实 batch)',
            'dataloader_batch_size': dataloader_batch_size,
            'dataloader_batch_size_说明': 'DataLoader 的 batch, 必须保持 1, 真实 batch 由 batch_size_per_case 决定(见 AirwayHMData 注释)',
            'num_workers': num_workers,
            'num_workers_说明': 'DataLoader 子进程数, 0=主进程读',
            'cube_size': cube_size,
            'cube_size_说明': '训练裁剪块边长',
            'aug': aug,
            'aug_说明': '1=开启随机翻转增强',
            'optimizer': 'AdamW',
            'optimizer_说明': '优化器类型, 初始学习率见 lr',
            'lr': learning_rate,
            'lr_说明': 'AdamW 初始学习率',
            'scheduler': 'MultiStepLR',
            'scheduler_说明': '学习率调度器, 到达 milestones 时把学习率乘 gamma',
            'milestones': lr_milestones,
            'milestones_说明': 'MultiStepLR 衰减点(epoch)',
            'gamma': lr_gamma,
            'gamma_说明': '学习率衰减系数',
            'loss': 'general_union_loss_lib(加权根号Tversky, alpha=0.2, beta=0.8, en+de 两路)',
            'loss_说明': '损失函数配置',
            'online_hard_mining': {
                'enabled': True,
                'cache_dir': online_savepath,
                'limits_ratio': hm_limits_ratio,
                'limits_ratio_说明': '缓存上限 = 病例数 x batch x 该比例, 只留最难的前 30%',
                'dataloader_batch_size': hm_dataloader_batch_size,
                'dataloader_batch_size_说明': '第二遍难例回放的 DataLoader 真实 batch(每个 item 是磁盘上单个难例 patch)',
            },
            'start_model': start_model,
            'start_model_说明': '热启动来源(stage1 权重目录)',
            'start_epoch': start_epoch,
            'start_epoch_说明': '热启动使用的 epoch 号',
        },
        'validation': {
            'DTI': DTI,
            'DTI_说明': '1=双阈值后处理, 0=固定 0.5 阈值',
            'cube_size': val_cube_size,
            'cube_size_说明': '验证滑窗块边长',
            'step': val_step,
            'step_说明': '验证滑窗步长(重叠一半)',
            'batch_size': val_batch_size_per_case_and_dataloader,
            'batch_size_说明': 'SegValCropData 补齐基数(per_case) and DataLoader batch, 两者必须相等(共用这一个值)',
            'fixed_threshold': 0.5,
            'fixed_threshold_说明': 'DTI=0 时使用的固定阈值',
            'dti_h_thresh': 0.5,
            'dti_h_thresh_说明': 'DTI=1 时种子阈值(见 validation())',
            'dti_l_thresh': 0.4,
            'dti_l_thresh_说明': 'DTI=1 时生长候选阈值(见 validation())',
        },
        'data': {'data_root': data_root, 'file_path': file_path, 'file_root': file_root,
                 'pred_path': pred_path,
                 '说明': 'data_root=数据根目录, file_path=病例列表, file_root=数据子目录; pred_path=stage1 预测结果路径(难例定位来源)'},
        'paths': {'model_savepath': model_savepath, 'online_savepath': online_savepath,
                  'log_savepath': log_savepath, 'tensorboard_logdir': log_root,
                  '说明': 'model_savepath=权重目录, online_savepath=难例缓存目录, log_savepath=训练日志, tensorboard_logdir=TensorBoard 目录'},
        'gpu': gpu,
        'gpu_说明': 'CUDA_VISIBLE_DEVICES 指定的 GPU 编号',
    })

    # resume
    # weights_dict = torch.load(os.path.join(model_savepath ,'SE_UNet_63.pth'))
    weights_dict = torch.load(start_model +
                              'SE_UNet_{}.pth'.format(start_epoch))
    model.load_state_dict(weights_dict, strict=True)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    starttime = time.time()
    val_td_list=[]
    val_bd_list=[]
    val_loss_random=[]
    val_loss_hard=[]
    for ep in range(max_epoches):
        if os.path.exists(online_savepath):
            shutil.rmtree(online_savepath)
            os.mkdir(online_savepath)
            os.mkdir(online_savepath + '/image')
            os.mkdir(online_savepath + '/label')
            os.mkdir(online_savepath + '/weight')
        else:
            os.mkdir(online_savepath)
            os.mkdir(online_savepath + '/image')
            os.mkdir(online_savepath + '/label')
            os.mkdir(online_savepath + '/weight')

        # ==== DataLoader 形状链 (train_data_loader <- AirwayHMData, stage2 第一遍) ====
        # Dataset.__getitem__: 每个 item 读一个病例整卷, 内部裁 batch_size_per_case(记为 B) 个 patch,
        #   四路各返回 (B, cube_size, cube_size, cube_size) —— 4 维无通道维(与 stage3 的区别: 无 skel)
        # DataLoader(batch_size=dataloader_batch_size=1) collate 在最前再套一维:
        #   pack[j].shape = (dataloader_batch_size, B, cube_size, cube_size, cube_size)
        #   例: B=2, cube_size=128 -> pack[0].shape = (1, 2, 128, 128, 128)
        # transpose(0,1) 后: (B, 1, cube_size^3); cat 两窗后 data: (B, 2, cube_size^3), 例 (2,2,128,128,128)
        for iter, pack in enumerate(train_data_loader):
            # pack 顺序对应 AirwayHMData 的 data_list: [img, img2, label, weight]（无 skel）
            data = pack[0].float().cuda()     # (1,B,cube,cube,cube)=(1,2,128,128,128) 窗宽1
            data2 = pack[1].float().cuda()    # 同上形状, 窗宽2
            label = pack[2].float().cuda()    # 同上形状, 二值标注
            weight = pack[3].float().cuda()   # 同上形状, 逐体素权重

            # 同 train3：把 Dataset 内部采出的 B 个 patch 从 (1,B,...) 还原成 (B,1,...)
            data = data.transpose(0, 1)
            data2 = data2.transpose(0, 1)
            label = label.transpose(0, 1)
            weight = weight.transpose(0, 1)
            data = torch.cat([data, data2], dim=1)   # 两窗宽拼成 2 通道

            # 【AMP bf16】前向+损失包进 autocast，归约/归一化自动保持 fp32；
            # backward/optimizer.step 不动(master weights 仍 fp32)。详见 train3 首处注释。
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_en, pred_de = model(data)
                pred_en = torch.sigmoid(pred_en)   # logits -> 概率
                pred_de = torch.sigmoid(pred_de)

                dice_loss_en = general_union_loss_lib(pred_en, label, weight)
                dice_loss_de = general_union_loss_lib(pred_de, label, weight)

                loss = dice_loss_de * 1 + dice_loss_en * 0.5   # 解码端为主，编码端辅助(深监督)
            if np.isnan(loss.item()):   # 【死代码】发现 NaN 只赋个无用变量，既不跳过也不报错，
                sdfs = 3                # NaN 仍会照常 backward 污染权重，应改为 continue 或 raise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 逐 patch 单独算一次损失，把【损失值本身当作文件名】存盘：
            #   文件名 = "<该patch的GUL损失>_<iter序号>.npy"，损失越大 = 越难 = 越值得再训
            # 这一步只是给难例排序打分，不参与反向传播
            names = [
                general_union_loss_lib(pred_de[i], label[i], weight[i])
                for i in range(data.shape[0])
            ]
            names = [str(n.item()) + '_' + str(iter) + '.npy' for n in names]
            # 写入定长难例缓存(第二层挖掘的写入端)，上限 = 病例数 x batch x 0.3，只留最难的前 30%
            save_data_online(online_savepath,
                             data,
                             label,
                             weight,
                             names,
                             limits=int(len(train_dataset) * batch_size_per_case *
                                        hm_limits_ratio))

            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset), '/',
                      max_step, 'loss:', loss.item(), 'dice loss encode:',
                      dice_loss_en.item(), 'dice loss decode:',
                      dice_loss_de.item())

            writer.add_scalars('Train', {'loss': loss.item(), 'dice_loss_encode': dice_loss_en.item(),'dice_loss_decode': dice_loss_de.item()}, iter + ep * len(train_dataset))
            # 同一批数值同步写 CSV，一行 = 一次迭代；step 列与 TensorBoard 横轴一致
            with open(loss_csv_path, 'a') as csv_f:
                csv_f.writelines(','.join([str(ep), str(iter), str(iter + ep * len(train_dataset)),
                                           str(loss.item()), str(dice_loss_en.item()),
                                           str(dice_loss_de.item())]) + '\n')
        torch.cuda.empty_cache()

        # writer.close() 原先写在这里(epoch 循环内)，导致每个 epoch 重建事件文件；已挪到函数末尾统一关闭

        lr_scheduler.step()        # 学习率衰减：本 epoch 唯一一次 step(难例遍后的第 2 次 step 已移除, MultiStepLR 计数=真实 epoch)

        # '''
        # ============ 第二遍：把本 epoch 攒下的难例读回来再训一遍 ============
        # print('start online hard mining: ')
        # rate=1.0 表示读取缓存目录中【全部】难例(缓存本身已只保留最难的前 30%)
        hm_dataset = OnlineHMData(data_root=online_savepath,
                                  rate=hm_rate)
        # 原调用还传了 batch_size=hm_dataloader_batch_size，Dataset 内从未使用，已注释去除：
        # hm_dataset = OnlineHMData(data_root=online_savepath, batch_size=hm_dataloader_batch_size, rate=hm_rate)
        # OnlineHMData 每个 item 就是磁盘上单个难例 patch(128^3)，
        # 这里 DataLoader 的 batch_size=hm_dataloader_batch_size 就是真实 batch。
        # 第二遍每 epoch 迭代次数 = 缓存难例数。
        hm_dataloader = DataLoader(dataset=hm_dataset,
                                   batch_size=hm_dataloader_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True)
        # ==== DataLoader 形状链 (hm_dataloader <- OnlineHMData, stage2 第二遍难例) ====
        # Dataset.__getitem__: 每个 item 就是磁盘上单个难例 patch 的 np 数组(第一遍已拼好双窗):
        #   image (2, cube_size, cube_size, cube_size)
        #   label/weight (1, cube_size, cube_size, cube_size)
        # DataLoader(batch_size=hm_dataloader_batch_size) 直接堆出真 batch, 故【不做 transpose(0,1)】:
        #   pack[j].shape = (hm_dataloader_batch_size, C, cube_size, cube_size, cube_size)
        #   例: hm_dataloader_batch_size=2, cube_size=128
        #       data=(2,2,128,128,128)  label/weight=(2,1,128,128,128)
        for iter, pack in enumerate(hm_dataloader):
            # 难例已从磁盘读回, pack 顺序为 [image, label, weight], 无 img2
            data = pack[0].float().cuda()     # (hm_dataloader_batch_size,2,cube,cube,cube)=(2,2,128,128,128)
            label = pack[1].float().cuda()    # (hm_dataloader_batch_size,1,cube,cube,cube)=(2,1,128,128,128)
            weight = pack[2].float().cuda()   # 同 label 形状
            # 【AMP bf16】难例遍只包前向+损失，backward/step 原样(见第一遍注释)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_en, pred_de = model(data)

                pred_en = torch.sigmoid(pred_en)
                pred_de = torch.sigmoid(pred_de)
                # 难例遍的损失与第一遍同构(编码端 0.5 + 解码端 1)，只是没有 break_loss(stage2 无该损失)
                dice_loss_all = general_union_loss_lib(pred_en, label, weight) * 0.5 + \
                                general_union_loss_lib(pred_de, label, weight)
                loss =  dice_loss_all
            optimizer.zero_grad()
            loss.backward()         # 第 2 次 backward：难例遍的独立参数更新
            optimizer.step()


        print('')
        # stage2 每个 epoch 都验证（与 stage1 只在最后一个 epoch 验证不同）
        TD_mean, BD_mean, loss_random, loss_hard=validation(data_root,model, valid_dataloader, ep, log_savepath, DTI, file_root,stage)
        print('')

        val_td_list.append(TD_mean)
        val_bd_list.append(BD_mean)
        val_loss_random.append(loss_random)
        val_loss_hard.append(loss_hard)

        train_dataset.update_scheduler(ep, val_loss_random, val_loss_hard, val_td_list, val_bd_list)

        torch.cuda.empty_cache()

        os.makedirs(model_savepath, exist_ok=True)
        torch.save(
            model.module.state_dict(),
            os.path.join(model_savepath, 'SE_UNet_' + str(ep) + '.pth'))
        # print(ep, ':', time.time() - starttime)
    writer.close()   # 训练结束统一关闭 TensorBoard writer(整个 run 只生成一套事件文件)
    sdf = 2

def train(data_root,model_savepath,
          log_savepath,
          aug,
          DTI,
          file_path,
          file_root,
          gpu='0'
          ):
    """
    阶段 1 训练：最基础的一版，只用普通 Dice 损失 + 随机裁剪，不依赖任何前序预测。

    目的：先让网络学会“气道长什么样”，产出 pred_1 供 stage2/3 做难例定位。
    与后两阶段的差异：没有 weight 加权、没有难例挖掘第二遍、没有 break_loss。
    训练轮数最多(100)，因为它是从零开始学特征。
    """
    stage=1
    max_epoches = 100
    batch_size_per_case = 2
    aug_flag = aug

    # ===== 训练超参数集中定义(便于统一调参与复现) =====
    cube_size = 128                  # 训练裁剪块边长
    dataloader_batch_size = 1        # 训练 DataLoader 的 batch(必须保持 1，真实 batch 由 batch_size_per_case 决定)
    num_workers = 0                  # DataLoader 子进程数，0=主进程读
    learning_rate = 0.0001           # AdamW 初始学习率
    lr_milestones = [60, 90]         # MultiStepLR 衰减点(stage1 轮数多，衰减点相应后移)
    lr_gamma = 0.1                   # 衰减系数
    val_cube_size = 128              # 验证滑窗块边长
    val_step = 64                    # 验证滑窗步长(重叠一半)
    val_batch_size_per_case_and_dataloader = 1   # 验证 Dataset 补齐基数(per_case) and DataLoader batch 必须相等，共用这一个变量

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


    current_date = datetime.datetime.now()
    # TensorBoard run 目录不再硬编码 './LOG/' + split('./saved_model/')，
    # 而是从 log_savepath 所在目录(即 root_dir/LOG)和 model_savepath 末级目录名(stage_x)派生，
    # 这样无论用户指定什么总输出根目录，TensorBoard 日志都会落在该根目录的 LOG 子目录下
    # 取 log_savepath(日志文件完整路径)所在目录作为 TensorBoard 日志根目录；
    # 若用户只传了裸文件名(不含目录部分)，dirname 会返回空字符串，此时回退为当前目录 '.'
    tb_dir = os.path.dirname(log_savepath) if os.path.dirname(log_savepath) != '' else '.'
    os.makedirs(tb_dir, exist_ok=True)
    log_root = os.path.join(tb_dir, os.path.basename(os.path.normpath(model_savepath)) + '_{}_{}_{}_{}'.format(current_date.month, current_date.day,
                                                                      current_date.hour,
                                                                      current_date.minute))
    writer = SummaryWriter(str(log_root))

    # 训练 loss 逐迭代落 CSV(与 TensorBoard 同源记录)：无 TensorBoard 环境时
    # 可直接用 Excel/pandas 打开画 loss 曲线；追加模式，文件不存在时先写表头
    loss_csv_path = os.path.splitext(log_savepath)[0] + '_train_losses.csv'
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, 'a') as csv_f:
            csv_f.writelines('epoch,iter,step,loss,dice_loss_encode,dice_loss_decode\n')

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    model = SE_UNet(in_channel=2, n_classes=1)

    # CropSegData 是 stage1 专用的“纯随机裁剪”数据集，不做难例采样
    train_dataset = CropSegData(file_path=file_path,
                                data_root=data_root,
                                file_root=file_root,
                                batch_size=batch_size_per_case,
                                aug_flag=aug_flag)
    # 同 stage3 的【关键设计】：Dataset 内 batch_size_per_case 决定每次裁几个同病例 patch(真实 batch)，
    # DataLoader 的 1 必须保持不动(>1 会触发 transpose/cat 的通道维 bug)；
    # __len__=病例数(5)，所以一个 epoch 第一遍 = 5 次迭代 = 5 个 patch。
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=dataloader_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True)
    # 注意：Dataset 的 batch_size 只是"每病例块数补齐的基数"，与下面 DataLoader 的
    # batch_size 是两个独立参数；两者必须相等，否则 batch 跨病例 + drop_last 丢尾块。
    # 想一次推理多块提速，把两处同时改成同一个值(如都=8)。详见 SegValCropData.__init__ 注释。
    valid_dataset = SegValCropData(file_path,
                                   data_root,
                                   batch_size=val_batch_size_per_case_and_dataloader,
                                   cube_size=val_cube_size,
                                   step=val_step)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=val_batch_size_per_case_and_dataloader,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    max_step = len(train_dataset) * max_epoches

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=lr_milestones,   # stage1 轮数多，衰减点相应后移
                                                        gamma=lr_gamma)

    # 【超参数落盘】把本阶段训练+验证用到的全部超参数写成 JSON，存进权重目录，便于复现与追溯
    save_hyperparams(model_savepath, 'hyperparams_stage_one.json', {
        'stage': 1,
        'stage_说明': '训练阶段编号(1=第一阶段)',
        'model': {'arch': 'SE_UNet', 'in_channel': 2, 'n_classes': 1,
                  '说明': 'SE_UNet; in_channel=输入通道数(两种窗宽归一化), n_classes=输出通道数(前景概率)'},
        'train': {
            'max_epoches': max_epoches,
            'max_epoches_说明': '总 epoch 数',
            'batch_size_per_case': batch_size_per_case,
            'batch_size_per_case_说明': 'Dataset 内部每病例采几个 patch(真实 batch)',
            'dataloader_batch_size': dataloader_batch_size,
            'dataloader_batch_size_说明': 'DataLoader 的 batch, 必须保持 1, 真实 batch 由 batch_size_per_case 决定(见 AirwayHMData 注释)',
            'num_workers': num_workers,
            'num_workers_说明': 'DataLoader 子进程数, 0=主进程读',
            'cube_size': cube_size,
            'cube_size_说明': 'CropSegData 内部随机裁剪的 patch 边长',
            'aug': aug,
            'aug_说明': '1=开启随机翻转增强',
            'optimizer': 'AdamW',
            'optimizer_说明': '优化器类型, 初始学习率见 lr',
            'lr': learning_rate,
            'lr_说明': 'AdamW 初始学习率',
            'scheduler': 'MultiStepLR',
            'scheduler_说明': '学习率调度器, 到达 milestones 时把学习率乘 gamma',
            'milestones': lr_milestones,
            'milestones_说明': 'MultiStepLR 衰减点(epoch)',
            'gamma': lr_gamma,
            'gamma_说明': '学习率衰减系数',
            'loss': 'dice_loss(无权重, en+de 两路等权相加)',
            'loss_说明': '损失函数配置',
        },
        'validation': {
            'DTI': DTI,
            'DTI_说明': '1=双阈值后处理, 0=固定 0.5 阈值',
            'cube_size': val_cube_size,
            'cube_size_说明': '验证滑窗块边长',
            'step': val_step,
            'step_说明': '验证滑窗步长(重叠一半)',
            'batch_size': val_batch_size_per_case_and_dataloader,
            'batch_size_说明': 'SegValCropData 补齐基数(per_case) and DataLoader batch, 两者必须相等(共用这一个值)',
            'fixed_threshold': 0.5,
            'fixed_threshold_说明': 'DTI=0 时使用的固定阈值',
            'dti_h_thresh': 0.5,
            'dti_h_thresh_说明': 'DTI=1 时种子阈值(见 validation())',
            'dti_l_thresh': 0.4,
            'dti_l_thresh_说明': 'DTI=1 时生长候选阈值(见 validation())',
        },
        'data': {'data_root': data_root, 'file_path': file_path, 'file_root': file_root,
                 '说明': 'data_root=数据根目录, file_path=病例列表, file_root=数据子目录'},
        'paths': {'model_savepath': model_savepath, 'log_savepath': log_savepath,
                  'tensorboard_logdir': log_root,
                  '说明': 'model_savepath=权重目录, log_savepath=训练日志, tensorboard_logdir=TensorBoard 目录'},
        'gpu': gpu,
        'gpu_说明': 'CUDA_VISIBLE_DEVICES 指定的 GPU 编号',
    })

    # resume
    # weights_dict = torch.load(os.path.join(model_savepath ,'SE_UNet_27.pth'))
    # model.load_state_dict(weights_dict, strict=True)
    # stage1 是从零训练(无 start_model)，故此处不加载任何权重；被注释的两行是手动续训入口
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    starttime = time.time()
    for ep in range(max_epoches):
        # ==== DataLoader 形状链 (train_data_loader <- CropSegData, stage1 唯一一遍) ====
        # Dataset.__getitem__: 每个 item 读一个病例整卷, 随机裁 batch_size_per_case(记为 B) 个 patch,
        #   返回 (img, img2, label, weight, name) 五路; 前四路各 (B, cube_size, cube_size, cube_size)
        #   —— 4 维无通道维; name 是病例名字符串(本循环未使用, 不参与形状计算)
        # DataLoader(batch_size=dataloader_batch_size=1) collate 在最前再套一维:
        #   pack[j].shape = (dataloader_batch_size, B, cube_size, cube_size, cube_size)
        #   例: B=2, cube_size=128 -> pack[0].shape = (1, 2, 128, 128, 128)
        # transpose(0,1) 后: (B, 1, cube_size^3); cat 两窗后 data: (B, 2, cube_size^3), 例 (2,2,128,128,128)
        for iter, pack in enumerate(train_data_loader):
            data = pack[0].float().cuda()     # (1,B,cube,cube,cube)=(1,2,128,128,128) 窗宽1
            data2 = pack[1].float().cuda()    # 同上形状, 窗宽2
            label = pack[2].float().cuda()    # 同上形状, 二值标注
            weight = pack[3].float().cuda()   # 同上形状, 逐体素权重

            data = data.transpose(0, 1)
            data2 = data2.transpose(0, 1)
            label = label.transpose(0, 1)
            weight = weight.transpose(0, 1)
            data = torch.cat([data, data2], dim=1)

            # 【AMP bf16】前向+损失包进 autocast，归约/归一化自动保持 fp32；
            # backward/optimizer.step 不动(master weights 仍 fp32)。详见 train3 首处注释。
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_en, pred_de = model(data)
                pred_en = torch.sigmoid(pred_en)
                pred_de = torch.sigmoid(pred_de)
                # stage1 用的是不带 weight 的普通 Dice，两条输出路径等权相加(无 0.5 深监督折扣)
                dice_loss_en = dice_loss(pred_en, label)
                dice_loss_de = dice_loss(pred_de, label)
                loss = dice_loss_de + dice_loss_en

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset), '/',
                      max_step, 'loss:', loss.item(), 'dice loss encode:',
                      dice_loss_en.item(), 'dice loss decode:',
                      dice_loss_de.item())

            # break
            writer.add_scalars('Train', {'loss': loss.item(), 'dice_loss_encode': dice_loss_en.item(),'dice_loss_decode': dice_loss_de.item()}, iter + ep * len(train_dataset))
            # 同一批数值同步写 CSV，一行 = 一次迭代；step 列与 TensorBoard 横轴一致
            with open(loss_csv_path, 'a') as csv_f:
                csv_f.writelines(','.join([str(ep), str(iter), str(iter + ep * len(train_dataset)),
                                           str(loss.item()), str(dice_loss_en.item()),
                                           str(dice_loss_de.item())]) + '\n')

        # writer.close() 原先写在这里(epoch 循环内)，导致每个 epoch 重建事件文件；已挪到函数末尾统一关闭
        lr_scheduler.step()

        print('')
        # 注意：stage1 只在【最后一个 epoch】才跑验证。
        # 因此 log_stage_one.csv 只有一条记录，无法用 valid_recall/valid 挑最佳 epoch，
        # 这也是 __main__ 里 stage1 直接用 s1_epes-1(即最后一个 epoch)权重的原因。
        if ep == max_epoches-1:
            TD_mean, BD_mean, val_loss_random, val_loss_hard=validation(data_root,model, valid_dataloader, ep, log_savepath, DTI,file_root,stage)
        print('')
        torch.cuda.empty_cache()

        os.makedirs(model_savepath, exist_ok=True)
        torch.save(
            model.module.state_dict(),
            os.path.join(model_savepath, 'SE_UNet_' + str(ep) + '.pth'))
        # print(ep, ':', time.time() - starttime)
    writer.close()   # 训练结束统一关闭 TensorBoard writer(整个 run 只生成一套事件文件)
    return max_epoches

def validation(data_root,model, valid_dataloader, epoch,log_savepath,DTI,file_root,stage):
    """:return: (TD均值, BD均值, 全体素Dice均值, 难区域Dice均值)
    """
    # 关键修复: 验证属于推理阶段, 必须切到 eval()。
    # 本网络 SE_UNet.batchnorm=False, 归一化全用 InstanceNorm3d(train/eval 行为一致),
    # 所以真正的问题出在 DropLayer: 它的 forward 里 if self.training 会随机丢弃 30% 通道,
    # 而 batch_size=1 且逐 patch 前向 => 每个 patch 拿到不同的随机掩码,
    # 滑窗平均后的预测不可复现(表现为 TD 在 57/69/99 之间无规律跳变)。
    was_training = model.training  # 记录调用前的模式, 供函数结束时原样恢复
    model.eval()                   # 验证期间关闭 DropLayer 的随机通道丢弃
    TDs,BDs,DSCs,Pres,Sens,Spes,FPRs,FNRs = [], [], [], [],[], [], [], []
    val_losses_random, val_losses_hard = [], []
    # 逐病例指标记录: 每条为 [病例名, TD, BD, DSC, Pre, Sen, Spe, FPR, FNR]，
    # 两处结算点(循环内边界结算 + 循环外末病例结算)各 append 一条，循环结束后写入 CSV 落盘
    case_records = []

    last_name = ''
    flag = False     #遗留变量名，未使用
    h_thresh=0.5
    l_thresh=0.4   #原作者
    with torch.no_grad():
        # 【病例边界结算机制】累加器 pred/pred_num 属于"当前病例"，只有当下一批数据换了病例名，
        # 才说明上一病例的滑窗块已全部收齐。因此本分支内先对【上一病例 last_name】做完整收尾：
        # 融合 -> 阈值二值化 -> 难区Dice入列 -> evaluation_case 算 8 项指标并入列，
        # 之后才读新病例 label、重置累加器。首个 batch 时 last_name='' 不会误结算(有 if last_name!='' 守卫)。
        # 注意：最后一个病例在循环结束时也不会遇到"新病例"来触发本分支，它的融合概率还躺在
        # pred/pred_num 里，须由循环外那段"原样重复"的收尾代码补结算(见循环外注释)。
        # ==== DataLoader 形状链 (valid_dataloader <- SegValCropData, stage1/2/3 共用验证) ====
        # Dataset.__getitem__: 每个 item 是整卷滑窗序列中的一个块:
        #   img_crop (2, val_cube_size, val_cube_size, val_cube_size) 双窗已堆好通道
        #   name 病例名字符串; pos (6,) = 该块在整卷中的坐标 [xl,xr,yl,yr,zl,zr]
        # DataLoader(batch_size=val_batch_size_per_case_and_dataloader=1) 在最前套一维:
        #   x=(1, 2, val_cube_size, val_cube_size, val_cube_size); name=1 元素列表; pos=(1, 6)
        #   例: val_cube_size=128 -> x.shape = (1, 2, 128, 128, 128)
        # 与训练不同: 验证每 item 就是单块标准 5D 输入, 全程无 transpose(0,1) 还原操作
        for i, (x, name, pos) in enumerate(valid_dataloader):
            name = name[0]                    # 从 1 元素列表里取回病例名字符串
            if name != last_name:
                if last_name != '':
                    pred = pred / pred_num
                    pred = np.squeeze(pred)
                    if DTI==0:
                        pred[pred >= 0.5] = 1
                        pred[pred < 0.5] = 0
                    else:
                        pred=double_threshold_iteration(pred,h_thresh,l_thresh)

                                
                    if stage!=1:
                        # 【难区掩码】1-pred1 把整卷划成两段：stage1 已判为气道的地方=0(掩掉，
                        # 不计入本轮成绩)，stage1 没判出的地方=1(含"真漏检气道"+"全部背景"，
                        # 即当前模型的增量战场)。乘上掩码后：
                        #   hard_pred  = 当前模型在掩码区新预测出的气道(捡回来的 + 乱捡的假阳)
                        #   hard_label = 掩码区里真正该捡回的气道(=stage1 的漏检集合)
                        # 于是下面两个二值 Dice(注意是分数不是损失，越大越好)分别衡量：
                        #   random 路 = 整卷总体表现；hard 路 = 只考"stage1 漏掉的找回了多少"。
                        # 两者的差 diff 供 update_scheduler 调节 hard_ratio：diff 大=难区拖后腿
                        # -> 多采难例；diff 小=难区已追上 -> 回归均衡。stage1 无上一阶段预测，故跳过。
                        hard_pred = pred * (1 - pred1)
                        hard_label= label * (1 - pred1)

                        val_losses_random.append(2 * (pred * label).sum() / (pred + label).sum())
                        val_losses_hard.append(2 * (hard_pred * hard_label).sum() / (hard_pred + hard_label).sum())

                    
                    TD,BD,DSC,Pre,Sen,Spe,FPR,FNR= evaluation_case(pred, label, last_name,file_root)
                    TDs.append(TD)
                    BDs.append(BD)
                    DSCs.append(DSC)
                    Pres.append(Pre)
                    Sens.append(Sen)
                    Spes.append(Spe)
                    FPRs.append(FPR)
                    FNRs.append(FNR)
                    # 边界结算：记录刚完成的上一病例(last_name)的逐例指标
                    case_records.append([last_name, TD, BD, DSC, Pre, Sen, Spe, FPR, FNR])

                label = sitk.ReadImage(data_root+'/mask/'+name+ 'mask_cut' + '.mha')
                label = sitk.GetArrayFromImage(label)

                if stage!=1:
                    # 统一用 load_itk_image 读取 save_itk 写出的 3D 标量体数据,
                    # 显式接收 origin/spacing/direction, 与全局读取惯例一致
                    pred1, origin, spacing, direction = load_itk_image(os.path.join(file_root, 'pred_1', name + '.mha'))

                # 每个病例的整卷累加器：pred 存各体素概率之和，pred_num 存被覆盖次数，
                # 最后相除得到重叠区平均(滑窗融合)，避免边界拼接的接缝
                pred = np.zeros(label.shape)
                pred = pred[np.newaxis, np.newaxis, ...]   # 补成 (1,1,Z,Y,X) 与网络输出对齐
                pred_num = np.zeros(pred.shape)
                last_name = name

            x = x.cuda()
            # 【AMP bf16】验证前向也走 bf16 省显存(已在 no_grad 下，无需 scaler)；
            # sigmoid 输出是 bf16，累加进整卷 numpy 之前必须 .float() 升回 fp32，
            # 否则重叠滑窗的概率求和(多数体素累 4 次)会在 8 位尾数上丢精度
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p0, p = model(x)                # 只取解码端 p 做推理，p0(编码端辅助头)忽略
                p = torch.sigmoid(p)            # logits -> 概率，阈值判断前必须做
            p = p.float().cpu().detach().numpy()    # 升 fp32 后转回 CPU 再累加，省显存
            pos = pos.numpy()               # pos[patch_idx] = 该 patch 的 (xl,xr,yl,yr,zl,zr) 切块坐标
            # 用 patch_idx 而非 i：避免遮蔽外层 dataloader 的 enumerate 计数变量 i
            for patch_idx in range(len(pos)):
                # print(pos)
                xl, xr, yl, yr, zl, zr = pos[patch_idx,0], pos[patch_idx,1], pos[patch_idx,2], pos[patch_idx,3], pos[patch_idx,4], pos[patch_idx,5]
                pred[0, :, xl:xr, yl:yr, zl:zr] += p[patch_idx]        # 把该 patch 概率加回它在整卷中的位置
                pred_num[0, :, xl:xr, yl:yr, zl:zr] += 1               # 记录覆盖次数(step=64 重叠 -> 多数体素被加 4 次)

        # 【末病例补结算】evaluation_case 在验证里出现两次不是重复调用：
        # 循环内的边界结算只覆盖第 1 ~ N-1 个病例（每个病例要等"下一个病例"出现才触发结算），
        # 第 N 个病例随循环结束而"没有下文"，边界分支永不触发它。
        # 故此处把循环内那套收尾原样再做一遍，专算最后一个病例（此时 last_name 恰停在其文件名上）：
        # 融合/squeeze 对应循环内同名两行，阈值二值化对应 DTI 分支，难区 Dice 对应 val_losses_* 入列，
        # 下方 evaluation_case 对应循环内那次调用。删掉本段则各指标列表少一项（少算最后一例），
        # 均值/方差与 DEBUG 打印的 len(TDs) 会比病例总数少 1。
        pred = pred / pred_num   # 求平均 -> 每个体素的融合概率
        pred = np.squeeze(pred)  # 去掉补出来的 (1,1) 前导维，回到 (Z,Y,X)
        if DTI==0:
            pred[pred >= 0.5] = 1     # 简单固定阈值二值化
            pred[pred < 0.5] = 0
        else:
            pred=double_threshold_iteration(pred,h_thresh,l_thresh)   # 双阈值受约束生长

        if stage!=1:
            # “难区域”指标：用 stage1 预测的补集 (1-pred1) 屏蔽掉已做对的区域，
            # 只考察 stage1 漏掉的部分，衡量级联后段的增量价值
            hard_pred = pred * (1 - pred1)
            hard_label= label * (1 - pred1)

            val_losses_random.append(2 * (pred * label).sum() / (pred + label).sum())   # 全局 Dice
            val_losses_hard.append(2 * (hard_pred * hard_label).sum() / (hard_pred + hard_label).sum())  # 难区 Dice


        TD,BD,DSC,Pre,Sen,Spe,FPR,FNR= evaluation_case(pred, label, last_name,file_root)
        TDs.append(TD)
        BDs.append(BD)
        DSCs.append(DSC)
        Pres.append(Pre)
        Sens.append(Sen)
        Spes.append(Spe)
        FPRs.append(FPR)
        FNRs.append(FNR)
        # 末病例补结算：同样记录一条逐例指标
        case_records.append([last_name, TD, BD, DSC, Pre, Sen, Spe, FPR, FNR])

        TD_mean = np.mean(TDs)
        TD_std = np.std(TDs)
        BD_mean = np.mean(BDs)
        BD_std = np.std(BDs)
        DSC_mean = np.mean(DSCs)
        DSC_std = np.std(DSCs)
        Pre_mean = np.mean(Pres)
        Pre_std = np.std(Pres)
        Sen_mean = np.mean(Sens)
        Sen_std = np.std(Sens)
        Spe_mean = np.mean(Spes)
        Spe_std = np.std(Spes)
        FPR_mean = np.mean(FPRs)
        FPR_std = np.std(FPRs)
        FNR_mean = np.mean(FNRs)
        FNR_std = np.std(FNRs)
        print('DEBUG:', len(TDs), len(val_losses_random), len(val_losses_hard), stage)
        val_loss_random = np.mean(val_losses_random) if val_losses_random else 0.0
        val_loss_hard   = np.mean(val_losses_hard)   if val_losses_hard   else 0.0
        # val_loss_random= np.mean(val_losses_random)
        # val_loss_hard= np.mean(val_losses_hard)
        
        # 控制台仍打印人类可读的一行汇总(与旧 txt 日志格式一致, 便于盯盘)
        print("TD: %0.4f (%0.4f), BD: %0.4f (%0.4f), DSC: %0.4f (%0.4f), Pre: %0.4f (%0.4f), Sen: %0.4f (%0.4f), Spe: %0.4f (%0.4f), FPR: %0.4f (%0.4f), FNR: %0.4f (%0.4f)" % (
               TD_mean, TD_std,BD_mean,BD_std,DSC_mean,DSC_std,Pre_mean,Pre_std,Sen_mean,Sen_std,Spe_mean,Spe_std,FPR_mean,FPR_std,FNR_mean,FNR_std))
        # ===== 验证汇总日志: CSV 落盘(替代旧"epoch行/指标行/空行"三行制 txt) =====
        # 每个 epoch 追加一行；文件不存在时先写表头。
        # 列顺序即 valid_recall()/valid() 的解析契约, 改动需同步两处读取端。
        # 带 epoch 列后, 读取端返回真实 epoch 号, 不再依赖"记录在列表中的下标"
        _need_header = not os.path.exists(log_savepath)
        with open(log_savepath, 'a') as file:
            if _need_header:
                file.writelines('epoch,TD_mean,TD_std,BD_mean,BD_std,DSC_mean,DSC_std,'
                                'Pre_mean,Pre_std,Sen_mean,Sen_std,Spe_mean,Spe_std,'
                                'FPR_mean,FPR_std,FNR_mean,FNR_std\n')
            file.writelines(','.join([str(epoch),
                                      '%0.4f' % TD_mean, '%0.4f' % TD_std,
                                      '%0.4f' % BD_mean, '%0.4f' % BD_std,
                                      '%0.4f' % DSC_mean, '%0.4f' % DSC_std,
                                      '%0.4f' % Pre_mean, '%0.4f' % Pre_std,
                                      '%0.4f' % Sen_mean, '%0.4f' % Sen_std,
                                      '%0.4f' % Spe_mean, '%0.4f' % Spe_std,
                                      '%0.4f' % FPR_mean, '%0.4f' % FPR_std,
                                      '%0.4f' % FNR_mean, '%0.4f' % FNR_std]) + '\n')
        # 逐病例指标落盘: 写到与日志同名不同后缀的 CSV(如 ./LOG/log_stage_two_val_case_metrics.csv)，
        # 后缀带 val 以明确这是验证集口径, 与测试集的 test_case_metrics.csv 区分开;
        # 三个阶段(train/train2/train3)与 DTI 复验(dtival)都走本函数, 只是传入的
        # log_savepath 不同(如 log_stage_two_dti.csv → log_stage_two_dti_val_case_metrics.csv),
        # 后缀统一在这里生效;
        # 追加模式、每个 epoch 每病例一行，文件不存在时先写表头；不影响既有日志解析与评分逻辑
        case_csv_path = os.path.splitext(log_savepath)[0] + '_val_case_metrics.csv'
        _need_header = not os.path.exists(case_csv_path)
        with open(case_csv_path, 'a') as csv_file:
            if _need_header:
                csv_file.writelines('epoch,case,TD,BD,DSC,Pre,Sen,Spe,FPR,FNR\n')
            for rec in case_records:
                csv_file.writelines(','.join([str(epoch)] + [str(v) for v in rec]) + '\n')
    # 恢复调用前的模式: validation() 是在每个 epoch 末尾被训练循环调用的,
    # 若不恢复, 后续 epoch 会在 eval() 下训练(DropLayer 失效), 静默改变训练语义
    if was_training:
        model.train()
    return TD_mean, BD_mean, val_loss_random, val_loss_hard

def evaluation_case(pred, label, name,file_root):
    """
    单病例指标计算（ATM22 官方口径）。

    先做一步关键后处理：只保留【最大连通域】再算指标。
    理由：气道是一棵连通的树，孤立碎块必为假阳；ATM22 比赛口径也是这么规定的。
    注意 cc3d 用 26 连通(比 6 连通宽松，对角相邻也算连上)。

    :param file_root: 中间产物目录，需含 tree_parse_val/ 与 skeleton_val/ 两个子目录
    :return: (TD, BD, DSC, Pre, Sen, Spe, FPR, FNR)
      TD  Tree Detection  : 骨架被预测覆盖的长度比例(召回向，易被过预测刷高)
      BD  Branch Detected : 分支检出率(每条分支需 >=80% 体素被覆盖才算检出)
      DSC Dice            : 体素级重叠度(最严格的分割质量指标)
      Pre Precision       : 预测中真阳占比(过预测会直接打爆这一项)
      Sen Sensitivity     : 真阳召回率
      Spe Specificity     : 真阴占比(背景巨大，看似很高但区分力弱)
      FPR False Positive Rate : 假阳率 = 100 - Spe，背景体素中被误判为气道的比例
      FNR False Negative Rate : 假阴率 = 100 - Sen，真实气道体素中被漏掉的比例
    """
    parsing = sitk.ReadImage(os.path.join(file_root, 'tree_parse_val', name + 'mask_cut.mha'))
    parsing = sitk.GetArrayFromImage(parsing)
    # parsing=parsing.transpose(2,1,0)
    if len(pred.shape) > 3:
        pred = pred[0]
    if len(label.shape) > 3:
        label = label[0]

    # cd, num = measure.label(pred, return_num=True, connectivity=2)
    cd=cc3d.connected_components(pred, connectivity=26)
    region = measure.regionprops(cd)
    num_list = [i for i in range(1, np.max(cd)+1)]
    area_list = [region[i-1].area for i in num_list]
    volume_sort = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    if volume_sort!=[]:
        large_cd =(cd==volume_sort[0]).astype(np.uint8)
    else:
        large_cd=pred.astype(np.uint8)

    skeleton = sitk.ReadImage(os.path.join(file_root, 'skeleton_val', name + 'mask_cut.mha'))
    skeleton = sitk.GetArrayFromImage(skeleton)
    # skeleton=skeleton.transpose(2,1,0)
    skeleton = (skeleton > 0)
    skeleton = skeleton.astype('uint8')

    _,_,BD=branch_detected_calculation(large_cd, parsing, skeleton)
    DSC=dice_coefficient_score_calculation(large_cd, label)
    TD=tree_length_calculation(large_cd, skeleton)
    Sen=sensitivity_calculation(large_cd, label)
    Spe=specificity_calculation(large_cd, label)
    Pre=precision_calculation(large_cd, label)
    # FPR 假阳率：此前仅作为 Spe 的中间量，现单独输出(Spe = 100 - FPR)
    FPR=false_positive_rate_calculation(large_cd, label)
    # FNR 假阴率：此前仅作为 Sen 的中间量，现单独输出(Sen = 100 - FNR)
    FNR=false_negative_rate_calculation(large_cd, label)

    print(name, "TD: %0.4f" % (TD),"BD: %0.4f" % (BD), "DSC: %0.4f" % (DSC),"Precision: %0.4f" % (Pre), "Sen: %0.4f" % (Sen),
          "Spe: %0.4f" % (Spe), "FPR: %0.4f" % (FPR), "FNR: %0.4f" % (FNR))
    return TD,BD,DSC,Pre,Sen,Spe,FPR,FNR

def valid_recall(log_path):
    """
    从验证汇总 CSV 里挑“最佳 epoch”，供 stage2 选权重做 stage3 热启动。

    打分公式(偏召回)：SCORE = (TD+BD)*0.15 + (DSC+Pre)*0.2 + Sen*0.3
    隐患：DSC+Pre 合计只占 40% 权重，当模型严重过预测(DSC 只有 3~5)时，
    TD/BD/Sen 仍能刷到 99，导致该公式会选出“最会糊”的 epoch 而非“最准”的 epoch。

    返回日志里记录的【真实 epoch 号】(CSV 的 epoch 列)；
    旧 txt 时代“返回列表下标可能与权重文件名错位”的坑已随格式切换一并废弃。
    """
    records = read_val_csv(log_path)   # 共享解析函数, 定义在 metrics.py
    # SCORE = [r['TD']*0.15 + r['BD']*0.15 + r['DSC']*0.2 + r['Pre']*0.2 + r['Sen']*0.3 for r in records]   # 原作者代码权重        
    SCORE = [r['TD']*0.3 + r['BD']*0.3 + r['DSC']*0.2 + r['Pre']*0.2 for r in records]#保持和第三阶段的公式一致
        # 备选公式: TD*0.3 + BD*0.3 + DSC*0.1 + Pre*0.1 + Sen*0.1 + Spe*0.1
    best = SCORE.index(max(SCORE))
    print('第', records[best]['epoch'], '个epoch验证效果最佳，分数为', max(SCORE))
    return records[best]['epoch']

def valid(log_path):
    """
    另一套选模型公式(偏准确)：SCORE = TD*0.3 + BD*0.3 + DSC*0.2 + Pre*0.2

    与 valid_recall 的区别：TD+BD 合计权重从 30% 提到 60%、完全去掉 Sen，
    因此更偏向“分割准确”而非“召回”，选出的 epoch 可能与 valid_recall 不同。
    两套公式并存时务必注意：同一份日志会得出不同的 best_epoch。
    (注：旧 docstring 写的 "(TD+BD+DSC+Pre)/4" 与代码实际公式不符，已按代码修正；
     test.py 的 valid() 才是四项等权口径，两边公式本就不同。)

    返回日志里记录的【真实 epoch 号】，理由同 valid_recall。
    """
    records = read_val_csv(log_path)   # 共享解析函数, 定义在 metrics.py
    SCORE = [r['TD']*0.3 + r['BD']*0.3 + r['DSC']*0.2 + r['Pre']*0.2 for r in records]#原作者代码权重
        # 备选公式: TD*0.3 + BD*0.3 + DSC*0.1 + Pre*0.1 + Sen*0.1 + Spe*0.1
    best = SCORE.index(max(SCORE))
    print('第', records[best]['epoch'], '个epoch验证效果最佳，分数为', max(SCORE))
    return records[best]['epoch']

def dtival(model_savepath, dti_log_savepath, data_root, file_path, file_root, dtiep, gpu='0', stage=0):
    """
    DTI 复验：用【双阈值后处理】口径重新评估某个已训好的 epoch，单独写一份日志。

    为什么需要它：训练日志里的指标是 DTI=0(固定 0.5 阈值)口径，
    而实际部署常用 DTI=1 后处理，两者指标差异很大，需要分开记录、互不污染。

    :param dti_log_savepath: 独立日志路径(如 log_stage_three_dti.csv)，绝不能传主日志
    :param dtiep           : 要复验的 epoch 号(即权重文件名里的数字)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # 复验结果单独落盘, 绝不写回主日志。
    # 原因: 主日志 CSV 会被 valid()/valid_recall() 逐行读取参与选优;
    # 往主日志追加一条 DTI 复验记录, 会让同一 epoch 出现两条不同口径(DTI=0/DTI=1)的行,
    # 污染 DTI=0 口径的 best_epoch 选择。
    # log_dir = os.path.dirname(log_savepath)                      # 取主日志所在目录(如 ./LOG)
    # log_name, log_ext = os.path.splitext(os.path.basename(log_savepath))  # 拆出 'log_stage_three' 与 '.csv'
    # dti_log_savepath = os.path.join(log_dir, log_name + '_dti' + log_ext)  # 拼出独立文件: log_stage_three_dti.csv
    # if log_dir:
    # #     os.makedirs(log_dir, exist_ok=True)                      # 目录不存在则创建, 避免写入时报错
    # open(dti_log_savepath, 'w').close()                          # 覆盖式清空, 保证"一次复验只留一条记录", 可反复重跑
    print('DTI 复验结果将写入独立日志(不污染主日志):', dti_log_savepath)

    model = SE_UNet(in_channel=2, n_classes=1)

    # ===== 验证超参数集中定义(便于统一调参与复现) =====
    val_cube_size = 128              # 验证滑窗块边长
    val_step = 64                    # 验证滑窗步长(重叠一半)
    val_batch_size_per_case_and_dataloader = 1   # 验证 Dataset 补齐基数(per_case) and DataLoader batch 必须相等，共用这一个变量
    num_workers = 0                  # DataLoader 子进程数，0=主进程读

    # Dataset 的 batch_size 仅是块数补齐基数，必须与 DataLoader 的 batch_size 相等(详见 SegValCropData.__init__)
    valid_dataset = SegValCropData(file_path,
                                   data_root,
                                   batch_size=val_batch_size_per_case_and_dataloader,
                                   cube_size=val_cube_size,
                                   step=val_step)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=val_batch_size_per_case_and_dataloader,
                                  shuffle=False, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)

    # dtival 是纯推理复验函数(只调用 validation, 不做反向传播), 因此模型应处于 eval()
    weights_dict = torch.load(os.path.join(model_savepath ,'SE_UNet_{}.pth'.format(dtiep)))
    model.load_state_dict(weights_dict, strict=True)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # 【超参数落盘】DTI 复验(验证)所用的全部配置单独存 JSON，与训练日志一样保持口径独立、可追溯
    save_hyperparams(model_savepath, 'hyperparams_dtival_stage_{}.json'.format(stage), {
        'task': 'DTI 复验(双阈值口径重新评估已训好的 epoch)',
        'stage': stage,
        'stage_说明': '被复验权重所属的训练阶段编号',
        'model': {'arch': 'SE_UNet', 'in_channel': 2, 'n_classes': 1,
                  'weights': os.path.join(model_savepath, 'SE_UNet_{}.pth'.format(dtiep)),
                  'epoch': dtiep,
                  '说明': 'SE_UNet; in_channel=输入通道数(两种窗宽归一化), n_classes=输出通道数(前景概率); weights/epoch=被复验的权重文件与 epoch 号'},
        'validation': {
            'DTI': 1,
            'DTI_说明': '本函数固定用 DTI=1 双阈值后处理口径',
            'cube_size': val_cube_size,
            'cube_size_说明': '验证滑窗块边长',
            'step': val_step,
            'step_说明': '验证滑窗步长(重叠一半)',
            'batch_size': val_batch_size_per_case_and_dataloader,
            'batch_size_说明': 'SegValCropData 补齐基数(per_case) and DataLoader batch, 两者必须相等(共用这一个值)',
            'dti_h_thresh': 0.5,
            'dti_h_thresh_说明': '种子阈值(见 validation())',
            'dti_l_thresh': 0.4,
            'dti_l_thresh_说明': '生长候选阈值(见 validation())',
        },
        'data': {'data_root': data_root, 'file_path': file_path, 'file_root': file_root,
                 '说明': 'data_root=数据根目录, file_path=病例列表, file_root=数据子目录'},
        'paths': {'model_savepath': model_savepath, 'dti_log_savepath': dti_log_savepath,
                  '说明': 'model_savepath=权重目录(复验 JSON 也存这里), dti_log_savepath=复验结果独立日志路径'},
        'gpu': gpu,
        'gpu_说明': 'CUDA_VISIBLE_DEVICES 指定的 GPU 编号',
    })

    starttime=time.time()
    print('')
    DTI=1
    # 注意这里传的是 dti_log_savepath 而不是 log_savepath: validation() 内部用追加模式(open(...,'a'))写日志,
    # 传独立文件才能让 DTI=1 口径的指标与 DTI=0 的训练日志物理隔离, 两套口径也不会混在同一份文件里难以区分
    TD_mean, BD_mean, val_loss_random, val_loss_hard=validation(data_root,model, valid_dataloader, dtiep, dti_log_savepath, DTI, file_root, stage)
    # print((time.time()-starttime)/60,'mins')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # 下面注释掉的是【完整三阶段流水线】，跑通一次需要依次执行：
    #   train -> save_gradients_tw(产 pred_1) -> train2 -> valid_recall 选 epoch
    #   -> save_s2_pred(产 pred_2) -> save_weight_break(产 BR_weight/br_skel) -> train3
    # 因为每阶段耗时很长，日常调试时通常只放开其中一段，故此处大部分被注释。
    # gpu_all_num = 8
    # gpu_need_num = 2
    # lm_per_gpu = 40000
    # while 1:
    #     free = np.zeros((gpu_all_num))
    #     for i in range(gpu_all_num):
    #         free[i] = get_gpu_mem_info(i)
    #     if len(np.where(free > lm_per_gpu)[0]) >= gpu_need_num:
    #         break
    # gpu = ','.join(
    #     [str(x) for x in list(np.where(free > lm_per_gpu)[0][0:gpu_need_num])])
    
    # ===== 命令行参数：用 --root_dir 指定一个"总输出根目录" =====
    # 训练/验证产生的全部工件都会落在该目录下，多个实验可各自独立归档：
    #   <root_dir>/saved_model/stage_{one,two,three}  各阶段权重(含复验 JSON)
    #   <root_dir>/LOG/                               训练 CSV 日志、逐迭代 loss、TensorBoard run、DTI 复验日志
    #   <root_dir>/data/                              流水线中间产物(pred_1/pred_2/BR_weight/br_skel/online_hardmining)
    # 输入侧(--data_root 预处理数据、--file_root 只读工件 base_dict.json/skeleton/tree_parse/LIB_weight)不受 --root_dir 影响
    parser = argparse.ArgumentParser(description='SE-UNet 三阶段级联训练')
    parser.add_argument('--root_dir', default='./run/',
                        help='总输出根目录：权重/日志/中间产物全部生成在该目录下(默认当前目录)')
    parser.add_argument('--data_root', default=r'F:\datasets\ATM22\TrainBatch1\after_yxz',
                        help='预处理后数据根目录(含 data/ 与 mask/ 子目录)')
    parser.add_argument('--file_root', default='./data',
                        help='只读工件根目录：base_dict.json、skeleton、tree_parse、LIB_weight 等')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES 的 GPU 编号，如 "0" 或 "0,1"')
    args = parser.parse_args()

    root = os.path.abspath(args.root_dir)                 # 统一转绝对路径，避免相对路径拼接歧义
    model_root = os.path.join(root, 'saved_model')       # 权重根目录
    log_dir = os.path.join(root, 'LOG')                  # 日志根目录(CSV + TensorBoard)
    mid_root = os.path.join(root, 'data')                # 流水线中间产物根目录
    for d in (model_root, log_dir, mid_root):
        os.makedirs(d, exist_ok=True)                    # 提前建好目录，后续写日志/权重不会因目录缺失报错

    gpu = args.gpu
    data_root = args.data_root
    file_root = args.file_root
    file_path = os.path.join(file_root, 'base_dict.json')   # 病例划分 json(输入，位于 file_root 下)

    #######################################TRAIN_STAGE_1##############################################################
    model_savepath = os.path.join(model_root, 'stage_one')
    log_savepath = os.path.join(log_dir, 'log_stage_one.csv')
    aug = 1 
    DTI = 1 
    s1_epes=train(data_root, model_savepath, log_savepath, aug, DTI, file_path,file_root, gpu)

    ######################################STAGE_1_PRED#################################################
    pred1_path = os.path.join(mid_root, 'pred_1')
    save_gradients_tw(data_root,stage_load=model_savepath,file_root=file_root,savepath=pred1_path,file_path=file_path,layer=0,gpu=gpu,s1_epes=s1_epes)

    #######################################TRAIN_STAGE_2################################################################
    # start_model 末尾必须带分隔符：train2 内部用 start_model + 'SE_UNet_x.pth' 字符串拼接
    start_model = os.path.join(model_root, 'stage_one') + os.sep
    start_epoch = s1_epes-1

    model_savepath = os.path.join(model_root, 'stage_two')
    online_savepath = os.path.join(mid_root, 'online_hardmining_stage_two')
    log_savepath = os.path.join(log_dir, 'log_stage_two.csv')
    aug = 1  
    DTI = 0
    train2(data_root, model_savepath, online_savepath, log_savepath,pred1_path, aug, DTI, start_model, start_epoch, file_path, file_root, gpu)

    #######################################STAGE_2_PRED/BR-wi/br_skel#####################################################
    epoch = valid_recall(log_savepath)
    whichepoch = os.path.join(model_root, 'stage_two', 'SE_UNet_{}.pth'.format(epoch))
    pred2_path = os.path.join(mid_root, 'pred_2')
    save_s2_pred(data_root, whichepoch, pred2_path, file_path,gpu)

    BR_weight_path = os.path.join(mid_root, 'BR_weight')
    br_skel_path = os.path.join(mid_root, 'br_skel')
    save_weight_break(data_root, pred2_path, BR_weight_path, br_skel_path,file_path)

    ########################################TRAIN_STAGE_3################################################################
    start_model = os.path.join(model_root, 'stage_two') + os.sep
    start_epoch = epoch

    model_savepath = os.path.join(model_root, 'stage_three')
    online_savepath = os.path.join(mid_root, 'online_hardmining_stage_three')
    log_savepath = os.path.join(log_dir, 'log_stage_three.csv')
    aug = 1
    DTI = 0
    train3(data_root,model_savepath, online_savepath, log_savepath,pred2_path,br_skel_path,BR_weight_path,aug, DTI, 
           start_model, start_epoch, file_path, file_root, gpu)
    
    ###### DTI 复验：分别用 DTI=1 口径重新评估 stage2 与 stage3 的最佳 epoch
    # 注意两行用了不同的选模型公式：stage2 用 valid_recall(偏召回)，stage3 用 valid(偏准确)
    best_epoch = valid_recall( os.path.join(log_dir, 'log_stage_two.csv'))
    dtival( os.path.join(model_root, 'stage_two'), os.path.join(log_dir, 'log_stage_two_dti.csv'),data_root,file_path,file_root,best_epoch,gpu,stage=2)

    best_epoch = valid( os.path.join(log_dir, 'log_stage_three.csv'))
    dtival(os.path.join(model_root, 'stage_three'), os.path.join(log_dir, 'log_stage_three_dti.csv'),data_root,file_path,file_root,best_epoch,gpu,stage=3)

