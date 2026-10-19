# -*- coding: utf-8 -*-
"""
SE-UNet 气道分割：原始文件夹(ZYX 未转换)端到端测试评估脚本
==========================================================
适用输入：一个文件夹(或 data/mask 两个文件夹)，里面是原始 CT 与金标准 mask，
         文件名如 张三.nii.gz，数组顺序为 ZYX(即 convert_zyx_to_yxz.py 的"转之前")。

脚本把项目原有的 4 步链路合并成一次运行，无需手动跑中间脚本：
  1) ZYX -> YXZ 轴置换      —— 复用 convert_zyx_to_yxz.transpose_zyx_image，
                              与训练数据的制作方式完全一致(含 spacing/direction 元数据)
  2) CT 肺野提取 + 裁剪      —— preprocessing.savenpy(mode='prepro') 的内存等价实现；
                              mask 用同一 box 裁剪，等价 preprocessing.cutmask
  3) 滑窗推理 + 后处理       —— 与 test.py 的 test() 完全一致：
                              img-1024 -> 双窗宽归一化 -> 128^3 滑窗(step=64)累加平均
                              -> DTI(0.5/0.35) 或固定阈值 0.5 -> 边缘 15% 裁剪 -> 最大连通域
  4) 指标计算               —— 体素级: DSC / Pre / Sen / Spe / FPR / FNR (metrics.py)
                              树级:   TD / BD —— 对金标准 mask 现场做 skeletonize_3d +
                              airway_parse(等价 ske_and_parse.tree_parse_test)，每例必算

输出：
  <output_dir>/pred/<case>.mha       预测掩码(裁剪后 YXZ 几何，可与转换后的 CT 直接叠加查看)
  <output_dir>/gt_cut/<case>.mha     裁剪后的金标准(仅 --save_gt 时输出，便于目检对齐)
  <output_dir>/test_case_metrics.csv      逐病例明细: "epoch,case,TD,BD,DSC,Pre,Sen,Spe,FPR,FNR"
                                     (格式对齐 LOG/test_log_stage_three_case_metrics.csv)
  <output_dir>/test_summary.txt      汇总行: "TD: 均值 (标准差), BD: ..." 每次运行追加一行
                                     (格式对齐 LOG/test_log_stage_three.txt)

用法示例：
  # A. 一个文件夹内含 data/ 与 mask/ 两个子文件夹(最常见)
  python test_folder.py --input_dir F:/raw_dataset --weights ./saved_model/stage_three/SE_UNet_43.pth
  # B. CT 与 mask 分开在两个文件夹
  python test_folder.py --data_dir F:/ct --mask_dir F:/mask --weights ./saved_model/stage_three/SE_UNet_43.pth
  # C. 不给权重，由训练日志自动挑最佳 epoch(与 test.py 相同规则: (TD+BD+DSC+Pre)/4)
  python test_folder.py --input_dir F:/raw_dataset --log ./LOG/log_stage_three.txt --model_dir ./saved_model/stage_three
  # D. 常用可选项
  python test_folder.py --input_dir ... --weights ... --gpu 1 --DTI 0 --save_gt --output_dir ./test_result/my_test
"""
import argparse
import os
import re
import time
import warnings

import numpy as np
import SimpleITK as sitk
import torch
from skimage.morphology import skeletonize_3d

# 项目内已有实现，全部复用，保证与训练/测试管线逐行等价
from SE_UNet import SE_UNet
from util import maximum_3d, large_connected_domain26, th_2t, get_l
from preprocessing import crop_origin_yxz
from convert_zyx_to_yxz import transpose_zyx_image, case_key, assert_same_geometry
from test import double_threshold_iteration, valid              # DTI 后处理 + 读日志选最佳 epoch
from metrics import (dice_coefficient_score_calculation,
                     precision_calculation, sensitivity_calculation,
                     specificity_calculation, false_positive_rate_calculation,
                     false_negative_rate_calculation, tree_length_calculation,
                     branch_detected_calculation)

warnings.filterwarnings('ignore')   # 屏蔽 skeletonize/cc3d 的弃用警告等噪音


# ---------------------------------------------------------------- 病例配对 ----------------------------------------------------------------
def collect_nifti(folder):
    """递归收集文件夹内所有 .nii / .nii.gz 文件，返回 {case_key: path}。
    case_key 会去掉文件名末尾的 _data/_mask/_ct/_label 等后缀，
    因此 '张三_data.nii.gz' 与 '张三_mask.nii.gz' 会归到同一个 key '张三'。"""
    indexed = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and (path.name.lower().endswith('.nii')
                               or path.name.lower().endswith('.nii.gz')):
            key = case_key(path)
            if key in indexed:
                raise ValueError('重复病例 key %r: %s 与 %s' % (key, indexed[key], path))
            indexed[key] = path
    return indexed


def find_pairs(args):
    """根据命令行参数确定 CT/mask 配对，返回 [(case, ct_path, mask_path), ...]。
    支持三种目录组织：
      1) --data_dir/--mask_dir 分开指定
      2) --input_dir 下有 data/ 和 mask/ 两个子文件夹
      3) --input_dir 单层文件夹内用 _data/_mask 后缀区分"""
    from pathlib import Path
    if args.data_dir and args.mask_dir:
        ct_dir, mask_dir = Path(args.data_dir), Path(args.mask_dir)
    else:
        root = Path(args.input_dir)
        if (root / 'data').is_dir() and (root / 'mask').is_dir():
            ct_dir, mask_dir = root / 'data', root / 'mask'
        else:
            # 单层文件夹：按后缀把文件分成 CT 组和 mask 组
            all_files = collect_nifti(root)
            ct_map, mask_map = {}, {}
            for path in all_files.values():
                name = path.name
                stem = re.sub(r'(?i)\.nii(\.gz)?$', '', name)
                if re.search(r'(?i)[_-]?(mask|label)$', stem):
                    mask_map[case_key(path)] = path
                elif re.search(r'(?i)[_-]?(data|ct)$', stem):
                    ct_map[case_key(path)] = path
                else:
                    raise ValueError('文件夹 %s 既无 data/mask 子目录，文件 %s 也无 '
                                     'data/mask 后缀，无法区分 CT 与标注。'
                                     '请改用 --data_dir/--mask_dir 分开指定。' % (root, name))
            ct_dir, mask_dir = None, None
            common = sorted(set(ct_map) & set(mask_map))
            if not common:
                raise ValueError('在 %s 中未找到任何 *_data/*_mask 配对' % root)
            return [(k, ct_map[k], mask_map[k]) for k in common]
    ct_map, mask_map = collect_nifti(ct_dir), collect_nifti(mask_dir)
    common = sorted(set(ct_map) & set(mask_map))
    only_ct = sorted(set(ct_map) - set(mask_map))
    only_mask = sorted(set(mask_map) - set(ct_map))
    if only_ct:
        print('[警告] 以下病例缺 mask，跳过:', only_ct)
    if only_mask:
        print('[警告] 以下病例缺 CT，跳过:', only_mask)
    if not common:
        raise ValueError('data 与 mask 文件夹没有配对上任何病例(检查命名后缀)')
    return [(k, ct_map[k], mask_map[k]) for k in common]


# ---------------------------------------------------------------- 预处理(内存版) ----------------------------------------------------------------
def preprocess_case(ct_path, mask_path):
    """ZYX 原始 nii.gz -> (data_cut, label_cut, 几何信息)。
    完整复刻 preprocessing.py 的 savenpy(mode='prepro') + cutmask 两步，
    只是不落盘中间文件，CT 与 mask 共用同一个裁剪 box。"""
    ct_img = sitk.ReadImage(str(ct_path))
    mask_img = sitk.ReadImage(str(mask_path))
    assert_same_geometry(ct_img, mask_img, str(ct_path))   # size/origin/spacing/direction 必须一致

    # ---- 步骤 1: ZYX -> YXZ(与训练集制作时同一函数，spacing/direction 随轴置换) ----
    ct_img = transpose_zyx_image(ct_img)
    mask_img = transpose_zyx_image(mask_img)
    origin, spacing, direction = (ct_img.GetOrigin(), ct_img.GetSpacing(),
                                  ct_img.GetDirection())

    # ---- 步骤 2a: CT 肺野提取 + 裁剪(逐行对应 savenpy) ----
    case_pixels = sitk.GetArrayFromImage(ct_img) + 1024    # 原始 HU -> 0~3071 平移空间
    cmin = np.min(case_pixels)
    hist = np.histogram(case_pixels.flatten(), 300)
    th = -800
    if cmin <= th:   # 仅当数据最小值异常低时触发(与 savenpy 的肺底噪声钳制一致)
        hhh1 = hist[1][np.where(hist[1] >= th)[0][0]:]
        hhh0 = hist[0][np.where(hist[1] >= th)[0][0]:]
        maxloc = np.where(hhh0 == np.max(hhh0))
        firstPeak = hhh1[maxloc[0][0]]
        measureDists = np.zeros([300], np.float32)
        for k in range(hhh0.shape[0]):
            measureDists[k] = pow(hhh1[k + 1] - firstPeak, 2) * hhh0[k]
        maxloc2 = np.where(measureDists == np.max(measureDists))
        secondPeak = hhh1[maxloc2[0][0]]
        aaa = firstPeak if secondPeak >= firstPeak else secondPeak
        case_pixels[np.where(case_pixels <= th)] = aaa

    T = th_2t(case_pixels)                 # 直方图双峰法求肺/软组织分割阈值
    L = get_l(case_pixels, T)              # 逐切片 2D 连通域提取左右肺
    L1 = maximum_3d(L)                     # 最大连通域 = 一侧肺(或双肺融合)
    L2 = maximum_3d(L ^ L1)                # 剩余部分里的第二大 = 另一侧肺
    L = L1 + L2
    xx, yy, zz = np.where(L)
    box = np.array([[np.min(xx), np.max(xx)],
                    [np.min(yy), np.max(yy)],
                    [np.min(zz), np.max(zz)]])
    margin = 5                             # 肺边界外扩 5 体素，防止贴边切掉气道
    box = np.vstack([np.max([[0, 0, 0], box[:, 0] - margin], 0),
                     np.min([np.array(L.shape), box[:, 1] + margin], axis=0).T]).T
    data_cut = case_pixels[box[0, 0]:box[0, 1],
                           box[1, 0]:box[1, 1],
                           box[2, 0]:box[2, 1]]
    crop_origin = crop_origin_yxz(origin, spacing, direction, box[:, 0])  # 裁剪后的物理原点

    # ---- 步骤 2b: mask 同 box 裁剪(对应 cutmask：先取最大连通域再裁) ----
    mask_arr = sitk.GetArrayFromImage(mask_img)
    mask_arr = (mask_arr > 0).astype(np.uint8)              # 兼容 0/255 或 0/1 标注
    mask_arr = large_connected_domain26(mask_arr)           # 26 连通最大域 + 填洞
    label_cut = mask_arr[box[0, 0]:box[0, 1],
                         box[1, 0]:box[1, 1],
                         box[2, 0]:box[2, 1]]
    return data_cut, label_cut, crop_origin, spacing, direction


# ---------------------------------------------------------------- 推理 ----------------------------------------------------------------
def process_imgmsk(img):
    """双窗宽归一化，与 SegTestCropData.process_imgmsk(data.py) 逐行等价。
    调用前必须先做 img-1024 平移(与训练/测试管线保持一致)。"""
    data = img.astype(np.float32)
    data2 = data.copy()
    data2[data2 > 500] = 500          # 通道2：肺窗 [-1000, 500]
    data2[data2 < -1000] = -1000
    data2 = (data2 + 1000) / 1500
    data[data > 1024] = 1024          # 通道1：宽窗 [-1024, 1024]
    data[data < -1024] = -1024
    data = (data + 1024) / 2048
    return data, data2


def crop_positions(shape, cube_size, step):
    """滑窗坐标生成，与 SegTestCropData.crop_pos(data.py) 等价(去掉补齐逻辑)。
    轴序 (Z,Y,X)；每维块数 = 整除则 1 块贴边、否则多 1 块；末块回缩贴边，保证完整覆盖。"""
    pos_list = []
    for dim in range(3):
        size = shape[dim]
        if size < cube_size:
            raise ValueError(
                '体数据某维尺寸 %d < cube_size %d，无法滑窗(项目管线假设输入已按气道裁剪过)。'
                % (size, cube_size))
        n = (size - cube_size) // step + 1 if (size - cube_size) % step == 0 \
            else (size - cube_size) // step + 2
        spans = []
        for k in range(n):
            lo = step * k
            hi = step * k + cube_size
            if hi > size:
                hi = size
                lo = size - cube_size
            spans.append((lo, hi))
        pos_list.append(spans)
    blocks = []
    for xl, xr in pos_list[0]:
        for yl, yr in pos_list[1]:
            for zl, zr in pos_list[2]:
                blocks.append((xl, xr, yl, yr, zl, zr))
    return blocks


def infer_one_volume(model, data_cut, cube_size, step, DTI, h_thresh, l_thresh, edge_crop):
    """对裁剪后的 data_cut(HU+1024 空间)做滑窗推理 + 后处理，返回二值预测 (Z,Y,X)。
    推理与后处理链与 test.py 的 test() 完全一致，只是输入改为内存数组。"""
    img = data_cut - 1024                        # 还原 HU，使下面窗宽阈值语义成立
    ch1, ch2 = process_imgmsk(img)               # 双窗宽归一化 -> 两通道
    two_channel = np.array([ch1, ch2])           # (2,Z,Y,X)

    pred = np.zeros((1, 1) + data_cut.shape, dtype=np.float32)      # 概率累加器
    pred_num = np.zeros((1, 1) + data_cut.shape, dtype=np.float32)  # 覆盖次数累加器
    with torch.no_grad():
        for (xl, xr, yl, yr, zl, zr) in crop_positions(data_cut.shape, cube_size, step):
            block = two_channel[:, xl:xr, yl:yr, zl:zr].astype(np.float32)
            x = torch.from_numpy(block)[np.newaxis].cuda()          # (1,2,128,128,128)
            _, p = model(x)                                         # p0 为编码端辅助头，不用
            p = torch.sigmoid(p).cpu().detach().numpy()[0]          # logits -> 概率
            pred[0, :, xl:xr, yl:yr, zl:zr] += p
            pred_num[0, :, xl:xr, yl:yr, zl:zr] += 1

    pred = (pred / pred_num)[0, 0]               # 重叠区平均 -> 融合概率图 (Z,Y,X)
    if DTI == 0:
        pred = (pred >= 0.5).astype(np.float32)  # 固定阈值二值化
    else:
        pred = double_threshold_iteration(pred, h_thresh, l_thresh)  # 双阈值受约束生长

    if edge_crop:   # 边缘 15% 置 0：裁剪后边缘滑窗缺上下文、假阳最多(与 test.py 相同先验)
        pred[0:int(0.15 * pred.shape[0]), :, :] = 0
        pred[int(0.85 * pred.shape[0]):, :, :] = 0
        pred[:, 0:int(0.15 * pred.shape[1]), :] = 0
        pred[:, int(0.85 * pred.shape[1]):, :] = 0

    pred = maximum_3d(pred)                      # 只保留最大连通域(气道树必连通)
    return pred.astype(np.uint8)


# ---------------------------------------------------------------- 指标 ----------------------------------------------------------------
def compute_metrics(pred, label):
    """单病例指标。TD/BD 每例必算：现场对金标准做骨架化/分支解析
    (等价 ske_and_parse.tree_parse_test 的产物，但只针对本病例临时生成)。
    返回顺序对齐 LOG/test_log_stage_three.txt: TD,BD,DSC,Pre,Sen,Spe,FPR,FNR"""
    m = {}
    try:
        # 延迟导入：airway_parse 依赖 pyvista/stl 等重组件，只在算树指标时才加载
        from ske_and_parse import airway_parse
        skeleton = skeletonize_3d(label).astype(np.uint8)        # 金标准中心线骨架
        parsing = airway_parse(label, merge_t=5)                 # 金标准逐分支标号图
        m['TD'] = tree_length_calculation(pred, skeleton)
        _, _, m['BD'] = branch_detected_calculation(pred, parsing, skeleton)
    except Exception as e:
        print('  [警告] 树级指标(TD/BD)计算失败:', e)
        m['TD'] = float('nan')
        m['BD'] = float('nan')
    m['DSC'] = dice_coefficient_score_calculation(pred, label)
    m['Pre'] = precision_calculation(pred, label)
    m['Sen'] = sensitivity_calculation(pred, label)
    m['Spe'] = specificity_calculation(pred, label)
    m['FPR'] = false_positive_rate_calculation(pred, label)
    m['FNR'] = false_negative_rate_calculation(pred, label)
    return m


# ---------------------------------------------------------------- 主流程 ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='SE-UNet 原始文件夹(ZYX nii.gz + mask)端到端测试评估')
    parser.add_argument('--input_dir', default=None,
                        help='原始数据文件夹(含 data/ 与 mask/ 子文件夹，或用 *_data/*_mask 后缀区分)')
    parser.add_argument('--data_dir', default=None, help='CT 文件夹(与 --mask_dir 搭配使用)')
    parser.add_argument('--mask_dir', default=None, help='mask 文件夹(与 --data_dir 搭配使用)')
    parser.add_argument('--weights', default=None,
                        help='权重 .pth；不给时须同时提供 --log 与 --model_dir，由 valid() 自动挑最佳 epoch')
    parser.add_argument('--log', default=None, help='训练日志，如 ./LOG/log_stage_three.txt')
    parser.add_argument('--model_dir', default=None, help='权重目录，如 ./saved_model/stage_three')
    parser.add_argument('--output_dir', default='./test_result/folder_test', help='结果输出目录')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES，如 "0"')
    parser.add_argument('--DTI', type=int, default=1, help='1=双阈值(0.5/0.35)受约束生长；0=固定阈值 0.5')
    parser.add_argument('--cube_size', type=int, default=128, help='滑窗块边长(须与训练一致)')
    parser.add_argument('--step', type=int, default=64, help='滑窗步长(须与训练一致)')
    parser.add_argument('--no_edge_crop', dest='edge_crop', action='store_false',
                        help='关闭边缘 15%% 裁剪')
    parser.add_argument('--save_gt', action='store_true', help='同时保存裁剪后的金标准 mask 便于目检')
    parser.set_defaults(edge_crop=True)
    args = parser.parse_args()

    if not args.input_dir and not (args.data_dir and args.mask_dir):
        parser.error('必须提供 --input_dir，或同时提供 --data_dir 与 --mask_dir')
    if args.weights is None:
        if args.log is None or args.model_dir is None:
            parser.error('未提供 --weights 时，必须同时提供 --log 与 --model_dir')
        best_epoch = valid(args.log)   # 与 test.py 相同：按 (TD+BD+DSC+Pre)/4 从日志挑 epoch
        args.weights = os.path.join(args.model_dir, 'SE_UNet_{}.pth'.format(best_epoch))
        print('自动选出最佳 epoch =', best_epoch, '-> 权重:', args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pred_dir = os.path.join(args.output_dir, 'pred')
    os.makedirs(pred_dir, exist_ok=True)
    if args.save_gt:
        os.makedirs(os.path.join(args.output_dir, 'gt_cut'), exist_ok=True)

    pairs = find_pairs(args)
    print('共 %d 个配对病例 | 权重: %s | DTI=%d | 边缘裁剪=%s | 树级指标=必算(TD/BD)'
          % (len(pairs), args.weights, args.DTI, args.edge_crop))

    # ---- 模型只加载一次，逐病例复用 ----
    model = SE_UNet(in_channel=2, n_classes=1)
    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)   # 容忍 DataParallel 的 module. 前缀差异
    model = torch.nn.DataParallel(model).cuda()
    model.eval()   # 必须 eval：否则 DropLayer 随机丢 30% 通道，滑窗平均不可复现

    # 落盘文件使用的指标集与顺序，严格对齐参考文件格式
    file_names = ['TD', 'BD', 'DSC', 'Pre', 'Sen', 'Spe', 'FPR', 'FNR']
    rows = []
    for i, (case, ct_path, mask_path) in enumerate(pairs):
        t0 = time.time()
        print('[%d/%d] %s' % (i + 1, len(pairs), case))
        try:
            # 步骤 1+2: ZYX->YXZ + 肺野裁剪(CT 与 mask 共用 box)
            data_cut, label_cut, crop_origin, spacing, direction = preprocess_case(ct_path, mask_path)
            # 步骤 3: 滑窗推理 + 后处理
            pred = infer_one_volume(model, data_cut, args.cube_size, args.step,
                                    args.DTI, 0.5, 0.35, args.edge_crop)
            # 步骤 4: 指标(预测与金标准都在同一裁剪 YXZ 空间，体素一一对应；TD/BD 每例必算)
            m = compute_metrics(pred, label_cut)
        except Exception as e:
            print('  [错误] 病例 %s 处理失败，跳过: %s' % (case, e))
            rows.append([case] + [''] * len(file_names))
            continue

        # 保存预测：沿用裁剪后 YXZ 几何，mha 格式(单文件自带元数据)，
        # 可与 YXZ 版 CT 在 ITK-SNAP 中直接叠加
        out_img = sitk.GetImageFromArray(pred.astype(np.byte))
        out_img.SetOrigin(crop_origin)
        out_img.SetSpacing(spacing)
        out_img.SetDirection(direction)
        sitk.WriteImage(out_img, os.path.join(pred_dir, case + '.mha'))
        if args.save_gt:
            gt_img = sitk.GetImageFromArray(label_cut.astype(np.byte))
            gt_img.SetOrigin(crop_origin)
            gt_img.SetSpacing(spacing)
            gt_img.SetDirection(direction)
            sitk.WriteImage(gt_img, os.path.join(args.output_dir, 'gt_cut', case + '.mha'))

        rows.append([case] + [m.get(k, '') for k in file_names])
        # 控制台逐例输出，分隔符与参考文件一致用逗号
        print('  ' + ', '.join('%s: %s' % (k, v) for k, v in m.items())
              + '  (%.1fs)' % (time.time() - t0))

    # ---- test_case_metrics.csv: 逐病例明细 ----
    # 格式对齐 LOG/test_log_stage_three_case_metrics.csv:
    #   epoch,case,TD,BD,DSC,Pre,Sen,Spe,FPR,FNR
    # epoch 从权重文件名 SE_UNet_{epoch}.pth 解析；解析不到则记 'NA'
    m_epoch = re.search(r'SE_UNet_(\d+)', os.path.basename(args.weights))
    epoch_tag = m_epoch.group(1) if m_epoch else 'NA'
    detail_path = os.path.join(args.output_dir, 'test_case_metrics.csv')
    with open(detail_path, 'w') as f:
        f.write('epoch,case,' + ','.join(file_names) + '\n')
        for r in rows:
            f.write(','.join([epoch_tag] + [str(v) for v in r]) + '\n')

    # ---- test_summary.csv: 汇总(均值/标准差各一列) ----
    # 表头与 train.py validation() 的 CSV 契约同构: epoch + 每指标 mean/std 两列，
    # 每次运行追加一行，方便多组权重/多批数据横向对比(与旧 test_summary.txt 行为一致)
    parts = []
    for k in file_names:
        vals = []
        for r in rows:
            try:
                v = float(r[1 + file_names.index(k)])
                if not np.isnan(v):        # 树指标失败例记 NaN，不参与均值
                    vals.append(v)
            except (ValueError, TypeError):
                pass                       # 整例失败的空串也跳过
        if vals:
            parts.append('%s: %0.4f (%0.4f)' % (k, np.mean(vals), np.std(vals)))
    line = ', '.join(parts)
    print('\n===== 汇总(均值(标准差)) =====\n' + line)
    summary_path = os.path.join(args.output_dir, 'test_summary.csv')
    _need_header = not os.path.exists(summary_path)
    with open(summary_path, 'a') as f:
        if _need_header:
            # 表头: epoch + 各指标 mean/std 成对列(列名与 file_names 顺序一致)
            f.write('epoch,' + ','.join(
                sum([['%s_mean' % k, '%s_std' % k] for k in file_names], [])) + '\n')
        # 数据行: 按指标顺序拼 mean/std; 某指标全例失败时补 NA, 保证列数与表头对齐
        cells = [epoch_tag]
        for k in file_names:
            vals = []
            for r in rows:
                try:
                    v = float(r[1 + file_names.index(k)])
                    if not np.isnan(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    pass
            cells += ['%0.4f' % np.mean(vals), '%0.4f' % np.std(vals)] if vals else ['NA', 'NA']
        f.write(','.join(cells) + '\n')
    print('完成。汇总行:', summary_path, ' 逐例明细:', detail_path, ' 预测掩码:', pred_dir)


if __name__ == '__main__':
    main()
