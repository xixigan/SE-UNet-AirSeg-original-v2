# -*- coding: utf-8 -*-

'''
Program :   The evaluation functions for the ATM 22 Challenge, including the TD / BD / DSC / Precision
Author  :   Minghui Zhang, Institute of Medical Robotics, Shanghai Jiao Tong University.
File    :   evaluation_atm22.py
Date    :   2022/02/02 16:19
Version :   V1.0
'''

import numpy as np


def branch_detected_calculation(pred, label_parsing, label_skeleton, thresh=0.5):
    """
    BD (Branch Detected, 分支检出率)：按【每条分支】单独判定是否被检出，
    再统计检出分支占总分支的比例。与 TD/DSC 的"全体素"口径不同，
    BD 关心的是"树的结构完整性"——主干对了但少一条侧枝，BD 会掉而 DSC 几乎不掉。

    :param pred           : 预测掩码(0/1)，此处传入的是 evaluation_case 筛过的最大连通域
    :param label_parsing  : 金标准"分支编号图"，每个体素的值 = 它所属分支的 id(1,2,3,...)，0 为背景
    :param label_skeleton : 金标准骨架(0/1)，气道中心线
    :param thresh         : 单条分支的"检出门槛"，被覆盖比例达到 0.5 才算这条分支检出
    :return: (总分支数, 检出分支数, 检出率百分比) —— 调用方只取第三个作为 BD
    """
    # 骨架 ∩ 分支编号：只保留落在中心线上的体素，并带上它属于哪条分支的 id。
    # 结果是一个"稀疏"标签图：绝大多数体素为 0，只有骨架上的点带分支 id。
    # 分支编号有可能是很多条，他不是uint8，所以切记最开始保存就要用 int32 类型
    label_branch = label_skeleton * label_parsing
    label_branch_flat = label_branch.flatten()
    # bincount(x) 返回长度 max(x)+1 的计数数组，第 i 位 = 值 i 出现的次数；
    # [1:] 丢掉第 0 位(背景体素数)，于是第 k 位(从 0 数)对应"分支标签 k+1 的骨架体素数"。
    # 【关键】bincount 会为 1..最大id 之间的【每一个】整数都留一个位置，
    # 若分支 id 不连续(如 1,2,3,7,9)或某分支与骨架无交集，该位置就是 0 ——
    # 这正是后续 0/0 产生 NaN 的根源
    label_branch_bincount = np.bincount(label_branch_flat)[1:] # 每条分支的各自体素数
    # 注意：这里用数组长度当"分支总数"，实为"最大分支 id"，
    # id 断档时该值偏大，会把不存在的分支也算进分母，使 BD 被低估
    total_branch_num = label_branch_bincount.shape[0]
    # 再与预测相乘：只留下"骨架上、且被预测覆盖到"的体素，仍带分支 id
    pred_branch = label_branch * pred
    pred_branch_flat = pred_branch.flatten()
    # 同样统计每条分支被覆盖的体素数；astype(int32) 是防止 pred 为布尔/浮点类型
    pred_branch_bincount = np.bincount(pred_branch_flat.astype(np.int32))[1:]
    # 若预测里最大分支 id 小于金标准(pred 完全没碰到编号最大的那些分支)，
    # bincount 返回的数组会更短，这里补 0 对齐长度，保证能与分母逐位相除
    if total_branch_num != pred_branch_bincount.shape[0]:
        lack_num = total_branch_num - pred_branch_bincount.shape[0]
        pred_branch_bincount = np.concatenate((pred_branch_bincount, np.zeros(lack_num)))
    # 逐分支覆盖率 = 该分支被预测到的骨架体素数 / 该分支骨架体素总数
    # 【隐患】分母为 0 的空分支 -> 0/0 = NaN，触发 RuntimeWarning: invalid value encountered in divide
    branch_ratio_array = pred_branch_bincount / label_branch_bincount
    # 按门槛判定：>=0.8 记 1(该分支检出)，否则记 0。
    # NaN >= 0.8 结果为 False -> NaN 被【静默】当作"未检出"，所以代码不会中断
    branch_ratio_array = np.where(branch_ratio_array >= thresh, 1, 0)
    detected_branch_num = np.count_nonzero(branch_ratio_array)
    detected_branch_ratio = round((detected_branch_num * 100) / total_branch_num, 2)
    return total_branch_num, detected_branch_num, detected_branch_ratio #BD


def dice_coefficient_score_calculation(pred, label, smooth=1e-5):
    """
    DSC (Dice Similarity Coefficient, Dice 相似系数)：体素级重叠度，
    等价于 F1 分数，是分割任务最严格的综合指标(同时受漏检和误检惩罚)。
    公式：DSC = 2|P∩T| / (|P|+|T|)
    """
    pred = pred.flatten()
    label = label.flatten()
    # 两个二值数组逐位相乘再求和 = 交集体素数，即 TP(真阳)
    intersection = np.sum(pred * label)
    # 分母 |P|+|T| = TP+FP + TP+FN，故 DSC = 2TP/(2TP+FP+FN)，
    # 与精确率/召回率的调和平均等价；smooth 防止两边全空时 0/0
    dice_coefficient_score = round(((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(label) + smooth)) * 100, 2)
    return dice_coefficient_score #DSC


def tree_length_calculation(pred, label_skeleton, smooth=1e-5):
    """
    TD (Tree Detection, 树检测率)：金标准【骨架】被预测覆盖的长度比例。
    公式：TD = |P∩S| / |S| × 100，S 为骨架体素。
    注意分母是骨架体素数(不是全体气道)，所以它衡量的是"中心线有没有被找着"，
    属于召回向指标；把气道预测得很粗、覆盖大片背景也刷不高它，
    但过度预测导致骨架全被包含时 TD 会接近 100，故需与 Pre 配合看。
    """
    pred = pred.flatten()
    label_skeleton = label_skeleton.flatten()
    # 分子 = 预测与骨架的交集体素数(TP 在骨架上的部分)；分母 = 骨架总体素数
    tree_length = round((np.sum(pred * label_skeleton) + smooth) / (np.sum(label_skeleton) + smooth) * 100, 2)
    return tree_length #TD


def false_positive_rate_calculation(pred, label, smooth=1e-5):
    """
    FPR (False Positive Rate, 假阳率/误报率)：全部【背景】体素中被误判为气道的比例。
    公式：FPR = FP / (FP+TN) = FP / 背景总体素数。
    现作为独立指标随 TD/BD/DSC/Pre/Sen/Spe 一起输出(Spe = 100 - FPR)。
    """
    pred = pred.flatten()
    label = label.flatten()
    # pred - pred*label：预测为 1 但金标准为 0 的体素 = FP(假阳)
    fp = np.sum(pred - pred * label) + smooth
    # 分母 sum(1-label) = 全部背景体素数 = FP + TN
    # 【注意】气道 CT 中背景占比 >99%，分母极大 -> FPR 极小且几乎不动，Spe 恒近 100
    fpr = round(fp * 100 / (np.sum((1.0 - label)) + smooth), 3)
    return fpr


def false_negative_rate_calculation(pred, label, smooth=1e-5):
    """
    FNR (False Negative Rate, 假阴率/漏检率)：全部【真实气道】体素中被漏掉的比例。
    公式：FNR = FN / (FN+TP) = FN / 金标准总体素数 = 100 - Sen。
    现作为独立指标随 TD/BD/DSC/Pre/Sen/Spe/FPR 一起输出(Sen = 100 - FNR)。
    """
    pred = pred.flatten()
    label = label.flatten()
    # label - pred*label：金标准为 1 但预测为 0 的体素 = FN(假阳/漏检)
    fn = np.sum(label - pred * label) + smooth
    # 分母 sum(label) = 真实气道总体素数 = FN + TP
    fnr = round(fn * 100 / (np.sum(label) + smooth), 3)
    return fnr


def sensitivity_calculation(pred, label):
    """
    Sen (Sensitivity, 敏感度)：即召回率 Recall = TP/(TP+FN)，"真实气道被找出多少"。
    此处用 100 - FNR 的等价形式实现(医学统计惯例)。
    惩罚的是【漏检】：砍断气道、丢失远端分支会直接拉低 Sen。
    """
    sensitivity = round(100 - false_negative_rate_calculation(pred, label), 3)
    return sensitivity #Sen


def specificity_calculation(pred, label):
    """
    Spe (Specificity, 特异度)：= TN/(TN+FP) = 100 - FPR，"背景被正确排除的比例"。
    因背景体素基数巨大，Spe 恒在 99.9x%，区分力很弱，
    故 ATM22 的 SCORE 加权公式中【不】使用 Spe，约束过预测的任务交给 Pre。
    """
    specificity = round(100 - false_positive_rate_calculation(pred, label), 3)
    return specificity #Spe


def precision_calculation(pred, label, smooth=1e-5):
    """
    Pre (Precision, 精确率/查准率)：= TP/(TP+FP)，"预测出的体素有多少是真的"。
    与 Sen 互补且互相牵制：Sen 分母是金标准(怕漏)，Pre 分母是预测(怕误)。
    分母 sum(pred) 是预测总体积，对假阳【敏感】，因此它是本任务中
    唯一能有效惩罚过度预测的指标(而非 Spe)。
    """
    pred = pred.flatten()
    label = label.flatten()
    # 两个二值数组逐位相乘求和 = 交集体素数 = TP
    tp = np.sum(pred * label) + smooth
    # 分母 sum(pred) = 预测总体素数 = TP + FP
    precision = round(tp * 100 / (np.sum(pred) + smooth), 3)
    return precision #Pre
