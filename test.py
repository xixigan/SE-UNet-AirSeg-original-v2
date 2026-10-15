# -*- coding: utf-8 -*-
"""
SE-UNet 气道分割：独立测试入口
==========================
与 train.py 的区别：
  train.py 的 validation()  : 训练过程中每个 epoch 末跑一次，用于选模型
  test.py  的 test()        : 训练全部结束后，用选定的 best_epoch 权重在测试集上跑一次
流程：读 log_stage_three.txt 选最佳 epoch -> 加载 stage_three 权重 -> 滑窗推理
     -> DTI 后处理 -> 边缘剪裁 -> 取最大连通域 -> 算指标 + 存 mha + 画箱线图
"""
import numpy as np
import torch
import os
import re
import cc3d
import matplotlib.pyplot as plt
import skimage.measure as measure
import SimpleITK as sitk
from matplotlib.pyplot import MultipleLocator
from torch.utils.data import DataLoader
from SE_UNet import SE_UNet
from data import SegTestCropData
from metrics import *
from util import *



def double_threshold_iteration(pred, h_thresh, l_thresh):
    """
    双阈值迭代（DTI）：气道分割后处理，与 train.py 中的同名函数完全重复（项级复用）。

    为何需要它：气道从粗的主干逐渐变细到末梢，末梢体素的预测置信度往往只有 0.3~0.5。
    单一阈值 0.5 会把这些末梢“腰斩”（TD/BD 下降）；阈值直接降到 0.35 又会引入大片
    孤立假阳（Pre 下降）。DTI 的思路是取两者之长：
      高阈值 -> 确定可靠的“种子”
      低阈值 -> 候选区，但只有与种子连通才允许被吸收（受约束区域生长）

    :param pred: 滑窗平均后的概率图, 取值 [0,1], 形状 (Z,Y,X)
    :param h_thresh: 高阈值(种子), 本项目用 0.5
    :param l_thresh: 低阈值(生长候选), test.py 用 0.35, train.py 用 0.4
    :return: 二值化预测图, 取值 {0,1}
    """
    # 26 邻域偏移表(3x3x3 去掉中心自身)：判断当前体素是否与已生长区域相邻
    neigb = np.array(
    [[-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0], [0, 1, 0], [1, -1, 0], [1, 0, 0], [1, 1, 0], [-1, -1, -1],
    [-1, 0, -1], [-1, 1, -1], [0, -1, -1], [0, 0, -1], [0, 1, -1], [1, -1, -1], [1, 0, -1], [1, 1, -1],
    [-1, -1, 1], [-1, 0, 1], [-1, 1, 1], [0, -1, 1], [0, 0, 1], [0, 1, 1], [1, -1, 1], [1, 0, 1],[1, 1, 1]])
    h, w,z = pred.shape
    pred = np.array(pred*255, dtype=np.float32)   # 概率缩放到 0~255，仅为下面比较直观
    bin = np.where(pred >= h_thresh*255, 255, 0).astype(np.float32)  # 种子集：高置信度体素
    gbin = bin.copy()      # gbin = grown binary，当前生长结果，初值为种子
    gbin_pre = gbin-1      # 上一轮快照；-1 造一个“必然不等”的初值，保证 while 至少进入一次
    # 已知缺陷：gbin_pre = gbin 是【引用赋值】而非拷贝，两者指向同一对象，
    #   导致 gbin_pre.all() != gbin.all() 恒为 False，区域生长实际只扩展【一层】就退出。
    #   正确写法应为 gbin_pre = gbin.copy()（但修好后会多出更多假阳，需配合重调阈值）。
    while(gbin_pre.all() != gbin.all()):
        gbin_pre = gbin
        for i in range(h):        # 纯 Python 三重循环遍历整卷(数千万体素)，极慢，应向量化
            for j in range(w):
                for k in range(z):
                    # 仅处理候选体素：未入选 且 概率落在 [低阈值, 高阈值) 区间
                    if gbin[i][j][k] == 0 and pred[i][j][k] < h_thresh*255 and pred[i][j][k] >= l_thresh*255:
                        for n in range(0, 26):   # 逐个查 26 邻域
                            inn = i + neigb[n, 0]
                            jnn = j + neigb[n, 1]
                            knn = k + neigb[n, 2]
                            # max(min(...)) 夹持索引避免越界（等价于边缘重复填充）
                            if gbin[max(min(inn,h-1),0)][max(min(jnn,w-1),0)][max(min(knn,z-1),0)]:
                                gbin[i][j][k] = 255   # 与已入选区域相邻 -> 吸收进来
                                break                 # 有一个邻居入选即可，不必查完 26 个

    return gbin/255   # 还原为 {0,1} 二值图

def valid(log_path):
    """
    从训练日志挑最佳 epoch。SCORE = (TD+BD+DSC+Pre)/4，四项等权平均。

    与 train.py 的 valid_recall() 公式不同(那个偏召回)，故同一份日志可能选出不同 epoch。
    日志约定每 3 行一条记录(epoch行/指标行/空行)，所以用 l%3==1 抽指标行。
    【风险】返回的是“记录列表下标”而非日志里写的真实 epoch 号；
    若日志被额外追加过记录，下标就会错位，拼出从未训练过的权重名并抛 FileNotFoundError。
    """
    with open(log_path, 'r') as file:
        lines=file.readlines()
    m_lines=[]
    for l in range(len(lines)):
        if l%3==1:
            m_lines.append(lines[l])
    TD=[]
    BD=[]
    DSC=[]
    Pre=[]
    Sen=[]
    Spe=[]
    SCORE=[]
    for l in m_lines:
        TD.append(float(re.findall(r'TD: (.*?) ', l)[0]))
        BD.append(float(re.findall(r'BD: (.*?) ', l)[0]))
        DSC.append(float(re.findall(r'DSC: (.*?) ', l)[0]))
        Pre.append(float(re.findall(r'Pre: (.*?) ', l)[0]))
        Sen.append(float(re.findall(r'Sen: (.*?) ', l)[0]))
        Spe.append(float(re.findall(r'Spe: (.*?) ', l)[0]))
    for l in range(len(TD)):
        SCORE.append((TD[l]+BD[l]+DSC[l]+Pre[l])/4)#原作者代码权重
        # SCORE.append(TD[l]*0.3 + BD[l]*0.3 + DSC[l]*0.1 + Pre[l]*0.1 + Sen[l]*0.1 + Spe[l]*0.1)


    print('第',SCORE.index(max(SCORE)),'个epoch验证效果最佳，分数为',max(SCORE))
    return SCORE.index(max(SCORE))

def test(data_root, model_stage, best_epoch, testlog_savepath, result_savepath,
         DTI, file_path, file_root, gpu):
    """
    测试主流程：加载指定 epoch 权重，对测试集逐病例滑窗推理、后处理、算指标并落盘。

    完整链路：滑窗累加平均 -> DTI 或固定阈值二值化 -> 边缘 15% 裁剪 -> 取最大连通域
             -> 写 mha(沿用原 CT 的几何信息) -> 计算六项指标 -> 打印/存日志/画箱线图

    :param model_stage : 权重目录，如 ./saved_model/stage_three（其末级名会用作结果子目录名）
    :param best_epoch  : 要测试的 epoch 号（由 valid() 选出）
    :param DTI         : 1=双阈值受约束生长; 0=固定 0.5 阈值
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.makedirs(result_savepath, exist_ok=True)   # 递归创建输出目录，避免写文件时报错

    # ===== 推理超参数集中定义(便于统一调参与复现) =====
    cube_size = 128                  # 滑窗推理块边长
    step = 64                        # 滑窗步长(重叠一半)
    batch_size_per_case_and_dataloader = 1   # Dataset 补齐基数(per_case) and DataLoader batch 必须相等，共用这一个变量
    num_workers = 0                  # DataLoader 子进程数，0=主进程读

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)   # 定义但未被使用(当前 num_workers=0)，属遗留代码

    model = SE_UNet(in_channel=2, n_classes=1)

    # 注意：Dataset 的 batch_size 只是"每病例块数补齐的基数"，与下面 DataLoader 的
    # batch_size 是两个独立参数；两者必须相等，否则 batch 跨病例 + drop_last 丢尾块。
    # 想一次推理多块提速，把两处同时改成同一个值(如都=8)。详见 SegValCropData.__init__ 注释。
    test_dataset = SegTestCropData(file_path,
                                   data_root,
                                   batch_size=batch_size_per_case_and_dataloader,
                                   cube_size=cube_size,
                                   step=step)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size_per_case_and_dataloader,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    weights_dict = torch.load(model_stage +
                              '/SE_UNet_{}.pth'.format(best_epoch))
    model.load_state_dict(weights_dict, strict=False)   # strict=False 容忍少量键不匹配
    model = torch.nn.DataParallel(model).cuda()
    # 关键修复: 测试是纯推理, 必须 eval()。
    # train() 下 DropLayer 会按 self.training 随机丢弃 30% 通道,
    # 而 batch_size=1 逐 patch 前向 => 每个 patch 掩码不同, 滑窗平均结果不可复现
    model.eval()

    # 八个指标的累积列表(按病例)，最后求均值/标准差用于汇总行与箱线图
    TDs, BDs, DSCs, Pres, Sens, Spes, FPRs, FNRs = [], [], [], [], [], [], [], []
    # 逐病例指标记录: 每条为 [病例名, TD, BD, DSC, Pre, Sen, Spe, FPR, FNR]，
    # 两处结算点(循环内边界结算 + 循环外末病例结算)各 append 一条，评估结束后写入 CSV 落盘
    case_records = []
    last_name = ''
    folder_name = model_stage.split('/')[-1]          # 如 'stage_three'，作为结果子目录
    os.makedirs(result_savepath +'/' +folder_name, exist_ok=True)
    flag = False                                      # 遗留未使用变量
    h_thresh = 0.5                                    # DTI 高阈值(种子)
    l_thresh = 0.35                                   # DTI 低阈值(生长候选)，比 train.py 的 0.4 更宽松

    # 【超参数落盘】把本次测试所用的全部超参数与关键配置写成 JSON，存进结果目录，便于复现与追溯
    save_hyperparams(os.path.join(result_savepath, folder_name), 'hyperparams_test.json', {
        'task': '测试(滑窗推理+后处理+指标计算)',
        'model': {'arch': 'SE_UNet', 'in_channel': 2, 'n_classes': 1,
                  'model_stage': model_stage,
                  'weights': model_stage + '/SE_UNet_{}.pth'.format(best_epoch),
                  'epoch': best_epoch,
                  '说明': 'SE_UNet; in_channel=输入通道数(两种窗宽归一化), n_classes=输出通道数(前景概率); model_stage=权重所属阶段目录, weights/epoch=测试所用权重与 epoch 号'},
        'inference': {
            'cube_size': cube_size,
            'cube_size_说明': '滑窗推理块边长',
            'step': step,
            'step_说明': '滑窗步长(重叠一半)',
            'dataset_batch_size': batch_size_per_case_and_dataloader,
            'dataset_batch_size_说明': 'Dataset 补齐基数(per_case) and DataLoader batch 共用同一个变量, 两者必须相等',
            'dataloader_batch_size': batch_size_per_case_and_dataloader,
            'dataloader_batch_size_说明': 'DataLoader 的 batch, 与 dataset_batch_size 共用同一个变量',
            'num_workers': num_workers,
            'num_workers_说明': 'DataLoader 子进程数, 0=主进程读',
        },
        'postprocess': {
            'DTI': DTI,
            'DTI_说明': '1=双阈值受约束生长; 0=固定 0.5 阈值',
            'fixed_threshold': 0.5,
            'fixed_threshold_说明': 'DTI=0 时使用的固定阈值',
            'dti_h_thresh': h_thresh,
            'dti_h_thresh_说明': 'DTI=1 时种子阈值',
            'dti_l_thresh': l_thresh,
            'dti_l_thresh_说明': 'DTI=1 时生长候选阈值',
            'edge_crop_ratio': 0.15,
            'edge_crop_ratio_说明': '边缘 15% 直接置 0',
            'keep_largest_component': True,
            'keep_largest_component_说明': '只保留 26 连通最大连通域',
        },
        'data': {'data_root': data_root, 'file_path': file_path, 'file_root': file_root,
                 '说明': 'data_root=数据根目录, file_path=病例列表, file_root=数据子目录'},
        'paths': {'result_savepath': result_savepath, 'testlog_savepath': testlog_savepath,
                  '说明': 'result_savepath=结果输出根目录, testlog_savepath=测试日志路径'},
        'gpu': gpu,
        'gpu_说明': 'CUDA_VISIBLE_DEVICES 指定的 GPU 编号',
    })
    with torch.no_grad():                             # 推理不需要梯度，省显存
        for i, (x, name, pos) in enumerate(test_dataloader):  
            name = name[0]   # DataLoader 把字符串包成 list，取回标量
            # 靠 name 变化判定“上一个病例的滑窗已全部累加完”，此时才做后处理与评估
            if name != last_name:
                if last_name != '':
                    print(last_name)
                    pred = pred / pred_num       # 重叠区求平均 -> 融合概率图
                    pred = pred[0, 0]            # 去掉 (1,1) 前导维 -> (Z,Y,X)
                    if DTI == 0:
                        pred[pred >= 0.5] = 1     # 固定阈值二值化
                        pred[pred < 0.5] = 0
                    else:
                        pred = double_threshold_iteration(pred, h_thresh, l_thresh)   # 双阈值受约束生长

                    # 边缘裁剪：把最外 15% 直接置 0。
                    # 依据：预处理已按气道范围裁剪过，边缘基本不可能是气道，
                    # 而滑窗在边缘处缺少上下文、假阳最多，故用先验强行切掉。
                    pred[0:int(0.15 * pred.shape[0]), :, :] = 0
                    pred[int(0.85 * pred.shape[0]):, :, :] = 0
                    pred[:, 0:int(0.15 * pred.shape[1]), :] = 0
                    pred[:, int(0.85 * pred.shape[1]):, :] = 0

                    pred = maximum_3d(pred)      # 只保留最大连通域(气道是连通的树，碎块必为假阳)
                    # 用原 CT 的几何信息重建 SimpleITK 图像，保证体素间距/原点/方向不错位，
                    # 否则下游 tree_parsing 按 spacing 换算毫米坐标会得到错误结果
                    pred_img = sitk.GetImageFromArray(pred.astype(np.byte))
                    pred_img.SetOrigin(img.GetOrigin())
                    pred_img.SetDirection(img.GetDirection())
                    pred_img.SetSpacing(img.GetSpacing())
                    sitk.WriteImage(
                        pred_img,
                        os.path.join(result_savepath, folder_name,
                                     last_name + '.mha'))

                    TD, BD, DSC, Pre, Sen, Spe, FPR, FNR = evaluation_case_test(
                        pred, label, last_name, testlog_savepath, file_root)
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
                # ---- 新病例开始：读入 CT 与标注，重置该病例的累加器 ----
                img = sitk.ReadImage(data_root + '/data/' + name + 'data_cut' +
                                     '.mha')
                arr = sitk.GetArrayFromImage(img)
                label = sitk.ReadImage(data_root + '/mask/' + name +
                                       'mask_cut' + '.mha')
                label = sitk.GetArrayFromImage(label)
                pred = np.zeros(arr.shape)                    # 概率累加器
                pred = pred[np.newaxis, np.newaxis, ...]      # 补成 (1,1,Z,Y,X)
                pred_num = np.zeros(pred.shape)               # 覆盖次数累加器
                last_name = name
            #
            x = x.cuda()
            p0, p = model(x)                 # 只取解码端 p；p0 是编码端辅助头，推理不用
            p = torch.sigmoid(p)             # logits -> 概率(阈值判断前必须做)
            p = p.cpu().detach().numpy()     # 转回 CPU 累加，省显存
            pos = pos.numpy()                # 每个 patch 的 (xl,xr,yl,yr,zl,zr) 切块坐标
            # 用 patch_idx 而非 i 作为内层循环变量：原写法复用了外层 dataloader 的 i,
            # 会把外层循环计数覆盖掉(虽因外层是 enumerate、每轮重新赋值而不会漏样本,
            # 但极易误读, 且一旦外层改成 while 或需要 i 做日志就会出错)
            for patch_idx in range(len(pos)):
                # print(pos)
                xl, xr, yl, yr, zl, zr = pos[patch_idx, 0], pos[patch_idx, 1], pos[patch_idx, 2], pos[
                    patch_idx, 3], pos[patch_idx, 4], pos[patch_idx, 5]
                pred[0, :, xl:xr, yl:yr, zl:zr] += p[patch_idx]        # 概率加回整卷对应位置
                pred_num[0, :, xl:xr, yl:yr, zl:zr] += 1               # step=64 重叠 -> 多数体素计 4 次
        print(last_name)
        pred = pred / pred_num
        pred = pred[0, 0]
        if DTI == 0:
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
        else:
            pred = double_threshold_iteration(pred, h_thresh, l_thresh)
        
        pred[0:int(0.15 * pred.shape[0]), :, :] = 0
        pred[int(0.85 * pred.shape[0]):, :, :] = 0
        pred[:, 0:int(0.15 * pred.shape[1]), :] = 0
        pred[:, int(0.85 * pred.shape[1]):, :] = 0

        pred = maximum_3d(pred)
        pred_img = sitk.GetImageFromArray(pred.astype(np.byte))
        pred_img.SetOrigin(img.GetOrigin())
        pred_img.SetDirection(img.GetDirection())
        pred_img.SetSpacing(img.GetSpacing())
        sitk.WriteImage(
            pred_img,
            os.path.join(result_savepath, folder_name, last_name + '.mha'))

        TD, BD, DSC, Pre, Sen, Spe, FPR, FNR = evaluation_case_test(
            pred, label, last_name, testlog_savepath, file_root)
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
        # FPR/FNR 追加在行尾，不影响既有日志解析(按 "指标名: 值 " 逐项正则抓取)
        print(
            "TD: %0.4f (%0.4f), BD: %0.4f (%0.4f), DSC: %0.4f (%0.4f), Pre: %0.4f (%0.4f), Sen: %0.4f (%0.4f), Spe: %0.4f (%0.4f), FPR: %0.4f (%0.4f), FNR: %0.4f (%0.4f)"
            % (TD_mean, TD_std, BD_mean, BD_std, DSC_mean, DSC_std, Pre_mean,
               Pre_std, Sen_mean, Sen_std, Spe_mean, Spe_std, FPR_mean, FPR_std, FNR_mean, FNR_std))
        line = "TD: %0.4f (%0.4f), BD: %0.4f (%0.4f), DSC: %0.4f (%0.4f), Pre: %0.4f (%0.4f), Sen: %0.4f (%0.4f), Spe: %0.4f (%0.4f), FPR: %0.4f (%0.4f), FNR: %0.4f (%0.4f)" % (
            TD_mean, TD_std, BD_mean, BD_std, DSC_mean, DSC_std, Pre_mean,
            Pre_std, Sen_mean, Sen_std, Spe_mean, Spe_std, FPR_mean, FPR_std, FNR_mean, FNR_std)
        with open(testlog_savepath, 'a') as file:
            file.writelines([line + '\n'])
        # 逐病例指标落盘: 写到与测试日志同名不同后缀的 CSV(如 ./LOG/testlog_stage_three_case_metrics.csv)，
        # 每病例一行；epoch 列记录本次测试所用的 best_epoch；文件不存在时先写表头
        case_csv_path = os.path.splitext(testlog_savepath)[0] + '_case_metrics.csv'
        _need_header = not os.path.exists(case_csv_path)
        with open(case_csv_path, 'a') as csv_file:
            if _need_header:
                csv_file.writelines('epoch,case,TD,BD,DSC,Pre,Sen,Spe,FPR,FNR\n')
            for rec in case_records:
                csv_file.writelines(','.join([str(best_epoch)] + [str(v) for v in rec]) + '\n')
        metrics = [TDs, BDs, DSCs, Pres, Sens, Spes, FPRs, FNRs]
        plt.figure(figsize=(10, 10))
        b = plt.boxplot(metrics,
                        meanline=True,
                        showmeans=True,
                        labels=['TD', 'BD', 'DSC', 'Pre', 'Sen', 'Spe', 'FPR', 'FNR'],
                        patch_artist=True)
        for c in b["boxes"]:
            c.set(color=random_color())
        plt.grid(linestyle='-.')
        plt.title('Metrics of ' + folder_name, fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        y_major_locator = MultipleLocator(10)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(0, 105)
        plt.savefig('Metrics of ' + folder_name + '.png')
        plt.show()
        # np.save('metrics of '+whichstage+'.npy', metrics)

def evaluation_case_test(pred, label, name,testlog_savepath,file_root):
    """
    单病例指标计算（测试版）。

    与 train.py 的 evaluation_case() 逻辑【完全相同】，唯一区别是读取的中间产物目录：
      本函数读 tree_parse_test/ 与 skeleton_test/   （测试集）
      前者读 tree_parse_val/ 与 skeleton_val/       （验证集）
    同样先取 26 连通的最大连通域再算指标，理由见 train.py 中同名函数的注释。
    """
    parsing = sitk.ReadImage(os.path.join(file_root, 'tree_parse_test', name + 'mask_cut.mha'))
    parsing = sitk.GetArrayFromImage(parsing)
    # parsing=parsing.transpose(2,1,0)
    if len(pred.shape) > 3:
        pred = pred[0]
    if len(label.shape) > 3:
        label = label[0]

    cd=cc3d.connected_components(pred, connectivity=26)
    region = measure.regionprops(cd)
    num_list = [i for i in range(1, np.max(cd)+1)]
    area_list = [region[i-1].area for i in num_list]
    volume_sort = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    if volume_sort!=[]:
        large_cd =(cd==volume_sort[0]).astype(np.uint8)
    else:
        large_cd=pred.astype(np.uint8)

    skeleton = sitk.ReadImage(os.path.join(file_root, 'skeleton_test', name + 'mask_cut.mha'))
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

if __name__ == '__main__':
    gpu_all_num = 1
    gpu_need_num = 1
    lm_per_gpu = 10000
    while 1:
        free = np.zeros((gpu_all_num))
        for i in range(gpu_all_num):
            free[i] = get_gpu_mem_info(i)
        if len(np.where(free > lm_per_gpu)[0]) >= gpu_need_num:
            break
    gpu = ','.join(
        [str(x) for x in list(np.where(free > lm_per_gpu)[0][0:gpu_need_num])])

    log_path = './LOG/log_stage_three.txt'
    best_epoch = valid(log_path)
    model_stage = './saved_model/stage_three'

    data_root = r'F:\datasets\ATM22\TrainBatch1\after'
    file_path = './data/test.json'
    file_root = './data'
    result_savepath = './test_result'
    testlog_savepath = './LOG/test_log_stage_three.txt'
    DTI = 1

    test(data_root,model_stage, best_epoch, testlog_savepath, result_savepath, DTI,
         file_path, file_root,gpu)
