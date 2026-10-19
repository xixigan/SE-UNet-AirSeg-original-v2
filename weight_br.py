import numpy as np
import os
import nibabel
import SimpleITK as sitk
import torch
import cc3d
import re
import pynvml
from scipy import ndimage
from skimage.morphology import skeletonize_3d,binary_dilation
from SE_UNet import SE_UNet
from data import load_json_file
from util import load_itk_image, save_itk
from metrics import read_val_csv   # 共享的验证汇总 CSV 解析函数(全部 8 个均值列)

torch.manual_seed(777) # cpu
torch.cuda.manual_seed(777) #gpu
np.random.seed(777) #numpy

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


def save_s2_pred(data_root, whichepoch, savepath,file_path, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    case_net = SE_UNet(in_channel=2, n_classes=1)
    # 优化: map_location='cpu' 让权重先读到 CPU, 避免 torch.load 在 GPU 上直接分配一份临时权重
    weights_dict = torch.load(whichepoch, map_location='cpu')
    case_net.load_state_dict(weights_dict, strict=False)
    del weights_dict  # 权重已载入模型, 立即释放字典
    case_net.cuda()
    case_net.eval()  # 关键修复: 推理必须 eval(), train() 会让 BatchNorm 使用 batch 统计量并保留中间激活
    # load data
    file_list = load_json_file(file_path, folder='0', mode=['train', 'val'])
    file_list.sort()
    for idx in range(len(file_list)):
        name = file_list[idx]
        img0, origin, spacing, direction = load_itk_image(data_root+'/data/'+name+'data_cut'+'.mha')

        img0=img0-1024

        label, origin, spacing, direction = load_itk_image(data_root+'/mask/'+name+'mask_cut'+'.mha')
        img, img2 = process_img(img0)

        # calculate gradients
        img = img[np.newaxis, np.newaxis, ...]
        img2 = img2[np.newaxis, np.newaxis, ...]
        label = label[np.newaxis, np.newaxis, ...]
        # 优化: 双通道整卷 CT 保留在 CPU(numpy), 只把当前 128^3 小块送入 GPU,
        # 避免整卷常驻显存造成的无谓显存浪费; 同时删除仅用于取形状的 y 的 GPU 拷贝
        x = np.concatenate((img.astype(np.float32), img2.astype(np.float32)), axis=1)

        cube_size = 128
        step = 64
        # 优化: 累加器由默认 float64 改为 float32, 内存占用减半
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

                    # 优化: 只有当前 patch 进入显存, 推理结束立刻释放
                    x_input = torch.from_numpy(x[:, :, xl:xr, yl:yr, zl:zr]).cuda()
                    # 关键修复: no_grad 关闭自动求导, 否则每个 patch 的计算图都会常驻显存
                    with torch.no_grad():
                        _, p = case_net(x_input)  # 优化: p0 从未使用, 不再持有其引用
                    pred[:, :, xl:xr, yl:yr, zl:zr] += p.cpu().numpy()
                    pred_num[:, :, xl:xr, yl:yr, zl:zr] += 1
                    del x_input, p  # 断开本 patch 的显存引用, 让缓存分配器可复用该块

        pred = pred / pred_num
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        # 修复: os.mkdir 只能创建单层目录, 父目录不存在会抛 FileNotFoundError,
        # 改用 os.makedirs 递归创建多级输出目录, 已存在则跳过
        os.makedirs(savepath, exist_ok=True)

        print(name, 'dice', 2 * (pred * label).sum() / (pred + label).sum())
        # 修复: pred[0] 仍是 (1,Z,Y,X) 的 4 维, SimpleITK 会把首轴当成向量分量,
        # 取 pred[0,0] 压成 3D 标量体数据, 与 prediction.py 中 np.squeeze 后的保存惯例一致
        save_itk(pred[0, 0].astype(np.uint8), origin, spacing, direction,
                 os.path.join(savepath, name + '.mha'))

        # 优化: 释放本 case 的大数组并清空显存缓存, 避免多 case 循环时显存碎片累积
        del img, img2, img0, label, x, pred, pred_num
        torch.cuda.empty_cache()


def save_weight_break(data_root,savepath,savepath2,savepath3,file_path):
    """
    生成气道分割二阶段训练用的"断点权重图"(weight break)。

    整体思路: 用一阶段预测结果与金标准 mask 对比, 找出落在气道中心线骨架上的
    假阴性体素(漏检), 即"骨架断点"——这些位置是预测的气道树被截断的关键点,
    训练损失需要在它们附近施加 >1 的权重, 迫使网络优先修复断开的分支。

    对每个 case 计算两种权重再融合:
      w_hm  : 难例挖掘(hard mining)权重, 按"到最近断点的距离"线性衰减后取平方;
      w_br  : 孤立断段权重, 按断段到其膨胀环带的距离(上限 2);
      最终权重 = (w_br + w_hm) * lamda + (1 - lamda), 再乘以 hard_mining,
      因此只有"与断点相关联的气道体素"保留权重(>1), 其余区域为 0。

    输出(每个 case 一个 .npy):
      savepath2: 权重图 w_br (float16, 与体数据同形状), 训练时按体素加权损失;
      savepath3: 孤立断段骨架体素的坐标 np.where(br_skel==1)。

    参数:
      data_root : 数据集根目录, mask 位于 data_root/mask/<name>mask_cut.mha
      savepath  : 一阶段预测结果目录(<name>.mha)
      savepath2 : 权重图输出目录
      savepath3 : 断点坐标输出目录
      file_path : 记录 train/val case 名的 json 文件
    """
    # 修复: 用 os.makedirs 递归创建多级输出目录(os.mkdir 遇到缺失的父目录会直接报错)
    os.makedirs(savepath2, exist_ok=True)
    os.makedirs(savepath3, exist_ok=True)

    # 取全部 train + val 病例(断点权重要为训练集和验证集都生成), 按名字排序保证可复现
    file_list = load_json_file(file_path, folder='0', mode=['train', 'val'])
    file_list.sort()
    for i in range(len(file_list)):
        name = file_list[i]
        # 金标准 mask: 气道区域体数据(0/1)
        label, origin, spacing, direction = load_itk_image(data_root+'/mask/'+name+ 'mask_cut' + '.mha')
        # 统一用 load_itk_image 读取 save_itk 写出的 3D 标量体数据,
        # 显式接收 origin/spacing/direction; pred 由 mask 的同一套元数据写出,
        # 此处重复赋值与上一行取到的值相同
        pred, origin, spacing, direction = load_itk_image(os.path.join(savepath, name+'.mha'))
        # fn(false negative) = 假阴性/漏检体素: 金标准里有、预测里没有的区域
        fn = ((label.astype(np.float16) - pred)>0).astype(np.uint8) 
        # label 的 3D 骨架化: 气道树的中心线(宽度 1 体素)
        skeleton = skeletonize_3d(label)
        # fn_skel = 落在中心线上的漏检体素, 即"骨架断点":
        # 预测在这些位置把气道树截断了, 是本函数要找的核心目标
        fn_skel = fn*skeleton 

        ################################### w_hm: 难例挖掘权重(按断点距离衰减)
        # 欧氏距离变换: edt[v] = 体素 v 到最近骨架体素的距离;
        # inds = 每个体素对应的"最近骨架点"坐标(3 个通道, 依次为 Z/Y/X 轴),
        # 后面用它把骨架上的信息"回填"到整个气道区域
        edt, inds = ndimage.distance_transform_edt(1-skeleton, return_indices=True)
        # hard_mining: 把骨架断点(fn_skel)沿 inds 回填到气道区域——
        # 每个气道体素取它最近中心线点的断点值, 再乘 label 只保留气道内部;
        # 值=1 表示"该体素所属的最近中心线位置发生了断裂"
        hard_mining = fn_skel[inds[0,...], inds[1,...], inds[2,...]] * label 
        # loc: 与断点相关联的气道体素集合(hard_mining > 0)
        loc = (hard_mining>0).astype(np.uint8)
        # f: 这些关联体素到最近骨架点的距离; 骨架本身置 0
        # (骨架上的权重由 hard_mining 项直接给出, 不需要距离衰减)
        f = loc * edt 
        f = f * (1. - skeleton)
        maxf = np.amax(f)   # 断点影响范围内的最大距离, 用作衰减归一化分母
        if np.max(maxf)==0:
            # 该 case 没有任何骨架断点(预测完整覆盖了中心线):
            # 权重图和断段全 0, 直接落盘后处理下一个 case
            # 注意: 此分支 savepath3 存的是全 0 体数据, 与正常分支存的坐标数组形状不一致
            w_br=np.zeros(label.shape,dtype=np.float16)
            br_skel=np.zeros(label.shape)
            np.save(os.path.join(savepath2, name+'.npy'), w_br)
            np.save(os.path.join(savepath3, name+'.npy'), br_skel)
            # 打印完整信息: 病例名 + 无断点说明 + 权重范围(原独立的 min/max 打印已合并至此)
            print('[save_weight_break] name:', name,
                  '| 该 case 无骨架断点, 跳过权重计算,',
                  '已保存权重图, 权重 min:', float(np.min(w_br)), '| 权重 max:', float(np.max(w_br)))
            continue
        # D: 线性衰减系数——骨架上 f=0 → D=1, 距离达到 maxf 时 D=0; 只在 loc 内有效
        D = -((1./(maxf)) * f) + 1
        D = D * loc

        # w_hm = hard_mining² × D²: 离断点越近、且确实属于断裂区域, 权重越大
        w_hm = (hard_mining**2)*(D**2)
        w_hm = w_hm.astype(np.float16)

        ################################### w_br: 孤立断段权重
        # 把骨架断点 fn_skel 按 26 连通分成若干连通域, 每个域是一段候选"断裂线段"
        cd=cc3d.connected_components(fn_skel, connectivity=26)
        br_skel=np.zeros(cd.shape)
        # 注意: 这里的循环变量 i 与外层 case 循环同名; Python 的 for 每轮都会从
        # range 迭代器重新给 i 赋值, 所以不会破坏外层遍历, 但阅读时容易混淆
        for i in range(1,np.max(cd)+1):
            t=cd==i
            t=t.astype(np.int8)
            neighbor_filter = ndimage.generate_binary_structure(3, 3)  # 3×3×3 全 1 邻域核
            # 骨架与邻域核卷积: 结果 = 该点 3×3×3 邻域内(含自身)的骨架体素数;
            # 乘 t 只看当前断段内部的体素
            skeleton_filtered = ndimage.convolve(skeleton, neighbor_filter) * t
            # 若断段中存在邻域骨架计数 == 2 的体素(自身 + 恰好 1 个骨架邻居),
            # 说明该段与其余骨架直接相连, 不是真正孤立的断裂, 跳过
            if np.sum(skeleton_filtered==2):
                continue
            # print(i)
            br_skel+=t   # 保留"孤立断段": 断裂且不与剩余骨架相连的线段
        # 把孤立断段(br_skel)沿最近骨架点索引回填到气道区域, 得到断段关联体素集合
        br_label = br_skel[inds[0,...], inds[1,...], inds[2,...]] * label 
        # X = 断段区域的膨胀环带(binary_dilation(br_label) - br_label, 即紧贴断段外围的一圈);
        # distance_transform_edt(1-X) 给出每个体素到最近 X 体素的距离;
        # 乘 br_label 后只保留断段内部体素的取值(骨架断段宽 1 体素, 值≈1)
        edt, inds = ndimage.distance_transform_edt(1-(binary_dilation(br_label)-br_label), return_indices=True)
        w_br=br_label*edt
        w_br[w_br>=2]=2   # 距离权重封顶为 2

        w_br = w_br.astype(np.float16)
        lamda=0.7
        # 融合两种权重: (w_br + w_hm)*0.7 + 0.3, 全图先抬到 ≥0.3
        # 理论上界(与具体 case 无关): w_br 封顶=2, w_hm=hard_mining²(0/1)×D²(≤1)≤1,
        # 故 max=(2+1)*0.7+0.3=2.4; 只要断段核心体素同时取到 w_br=2、w_hm=1 就打满
        w_br=(w_br+w_hm)*lamda+1-lamda
        # 最后乘 hard_mining: 只保留"与断点相关联的气道体素"上的权重,
        # 背景与正常气道区域归 0, 训练时损失在这些体素上被放大
        w_br=w_br*hard_mining

        # 权重图落盘: <case名>.npy, 与体数据同形状
        np.save(os.path.join(savepath2, name+'.npy'), w_br)
        # 断点坐标落盘: np.where 返回 (Z轴坐标数组, Y轴坐标数组, X轴坐标数组) 三元组
        np.save(os.path.join(savepath3, name+'.npy'),np.where(br_skel==1))
        # 打印完整信息: 病例名 + 有断点说明 + 权重范围(原独立的 min/max 打印已合并至此)
        # 注意 min/max 是公式的结构性常数(min=0: 背景被 hard_mining 清零; max=2.4 封顶),
        # 各 case 相同属正常现象, case 间真正差异体现在非零体素的数量与分布上。
        # 为何显示 2.40234375 而不是 2.4: w_br 是 float16, fp16 在 2.x 区间精度为
        # 2^-9=0.001953125, 存不下 2.4, 实际存的是最近可表示值 2.40234375;
        # float() 把该 fp16 真值精确展开成 float64 打印, 旧版 print(numpy 标量)则显示
        # numpy 的最短表示 2.402 —— 两者是同一个数, 磁盘数据不变。
        print('[save_weight_break] name:', name,
              '| 该 case 存在骨架断点, 已完成权重计算,',
              '已保存权重图, 权重 min:', float(np.min(w_br)), '| 权重 max:', float(np.max(w_br)))



def valid_recall(log_path):
    """
    从验证汇总 CSV 日志挑"最佳 epoch"(偏召回口径)。

    打分公式(与 train.py valid_recall 一致)：SCORE = (TD+BD)*0.15 + (DSC+Pre)*0.2 + Sen*0.3
    CSV 契约(与 train.py validation() 写入端一致)：首行表头 epoch,TD_mean,...,FNR_std，
    每个 epoch 一行；返回日志里记录的【真实 epoch 号】(epoch 列)，
    旧 txt 时代"返回列表下标可能错位"的坑已随格式切换一并废弃。
    """
    records = read_val_csv(log_path)   # 共享解析函数, 定义在 metrics.py; 全部 8 个均值已解析
    SCORE = [(r['TD'] + r['BD']) * 0.15 + (r['DSC'] + r['Pre']) * 0.2 + r['Sen'] * 0.3
             for r in records]
    best = SCORE.index(max(SCORE))
    print('第', records[best]['epoch'], '个epoch验证效果最佳，分数为', max(SCORE))
    return records[best]['epoch']


def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """

    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return free


if __name__ == '__main__':
    ###################################### GPU setup
    gpu_all_num = 12
    gpu_need_num = 4
    lm_per_gpu = 22000
    while 1:
        free = np.zeros((gpu_all_num))
        for i in range(gpu_all_num):
            free[i] = get_gpu_mem_info(i)
        if len(np.where(free > lm_per_gpu)[0]) >= gpu_need_num:
            break
    gpu = ','.join(
        [str(x) for x in list(np.where(free > lm_per_gpu)[0][0:gpu_need_num])])



    log_path = './LOG/log_stage_two.csv'
    epoch = valid_recall(log_path)
    whichepoch = './saved_model/stage_two/SE_UNet_{}.pth'.format(epoch)
    savepath='./data/pred_2'
    file_root='./data'
    data_root='/mnt/yby/AFTER_PREPROCESS'
    save_s2_pred(data_root,whichepoch, savepath, file_root, gpu)


    savepath2='./data/BR_weight'
    savepath3='./data/br_skel'
    save_weight_break(data_root,savepath, savepath2, savepath3, file_root)
