import numpy as np
import os
import nibabel
import copy
import cc3d
import pyvista as pv
from scipy import ndimage
from stl import mesh
import SimpleITK as sitk
import skimage.measure as measure
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize_3d,binary_closing,binary_dilation
from scipy.interpolate import interp1d
from data import load_json_file
from util import *
from ours_skel_parse import Topology_Tree, tree_parsing_func



def airway_parse(pred, merge_t):
    """
    气道树解析（airway tree parsing）主流程：把一份二值气道分割结果切成"逐分支标号"的结构化标注。

    参数:
        pred    : 3D 二值气道体数据 (H, W, D)，1 = 气道前景，0 = 背景
        merge_t : 分支合并阈值，单位是【骨架点数】：merging() 直接比较 len(start+member+end) <= merge_t，
                  全程不乘 spacing（本调用链也不做 resize），故它不是毫米，体素各向异性时
                  对应的物理长度随分支走向而异；作用是剔除过短的末梢碎枝以抑制噪声断枝

    返回:
        tree_parsing : 与 pred 同形状的 3D 标号图，每个气道体素的值 = 它所属分支的编号(0 = 非气道)，
                       即 ATM22  Challenge 所需的"气道树分层标号"标注。

    整体步骤:
        1) 自动判定体数据的 z 轴朝向(气管在上还是在下)，保证算法内部始终"从气管往肺外周"生长
        2) 构建拓扑树对象，依次做 骨架提取+分支切分(sub) → 末梢短枝合并(merge) → 层级编号(grade)
        3) 把每条分支的骨架点写成唯一编号，得到"标号化的骨架"
        4) 用最近邻(欧氏距离变换)把骨架上的分支编号传播到整个气道区域，得到最终解析结果
    """

    # ---------- 步骤 1：判断 z 轴方向 order ----------
    # 取气道在 z 轴上的起止范围(只统计前景所在的层面)，cha 为气道在 z 方向的总厚度
    minz = np.min(np.where(pred == 1)[2])
    maxz = np.max(np.where(pred == 1)[2])
    cha = maxz - minz

    # 在气道范围的 80% 高度处切一层的 2D 切片：此处已接近肺外周，气道已多次分叉，
    # 断成很多细小连通块，因此"最大连通块"的面积很小
    oneslice = pred[:, :, int(0.8 * cha + minz)]
    imLabel, _ = measure.label(oneslice, background=0, return_num=True)  # 2D 连通域标记
    noback = np.bincount(imLabel.reshape(-1))   # 统计每个标号(连通块)的体素个数
    noback[0] = 0                               # 把背景(标号0)剔除，不参与比较
    max_num8 = np.max(noback)                   # 80% 层处的最大连通块面积

    # 在气道范围的 20% 高度处切一层：此处靠近气管/主支气管，管腔粗，通常是一个大十字形断面
    oneslice = pred[:, :, int(0.2 * cha + minz)]
    imLabel, _ = measure.label(oneslice, background=0, return_num=True)
    noback = np.bincount(imLabel.reshape(-1))
    noback[0] = 0
    max_num2 = np.max(noback)                   # 20% 层处的最大连通块面积

    # 若低 z 端(20%)的断面明显更大，说明气管位于 z 轴小值端，符合算法默认约定 → order = 0；
    # 否则说明体数据上下颠倒(气管在 z 轴大值端) → order = 1，后续在树内部对 z 做一次翻转再翻回来
    if max_num2 > max_num8:
        order = 0
    else:
        order = 1

    # ---------- 步骤 2：构建拓扑树并完成 sub / merge / grade ----------
    # remerge_l：需要强制二次合并的分支编号列表；'000' 对应左主支气管一侧，
    # 该处易被误切成多段，故显式要求重新合并以保证层级正确
    remerge_l = ['000']
    airway_topo = Topology_Tree(pred, order, merge_t, remerge_l)

    # sub：先做 孔洞填充 + 膨胀 + 二值闭运算 + 3D 极大值滤波 来修补分割空洞与毛刺，
    #      再 skeletonize_3d 提取中心线骨架；按 z 排序骨架点、计算树重心原点，
    #      必要时对气管主干做中心线平滑，最后在每个分叉点/端点处把骨架切成若干分支段 → 结果存入 airway_topo.Bi
    airway_topo.sub()  #Skeleton extraction and segmentation operations yield airway_topo.Bi

    # merge：把长度小于 merge_t 的末梢叶枝/毛刺分支合并回父分支，抑制过分割；
    #        若 order == 1，合并后会把 z 坐标再翻转回原始体数据坐标系 → 更新 airway_topo.Bi
    airway_topo.merge()  #Remove small segments in leaves and branches to update airway_topo.Bi

    # grade：按拓扑父子关系给每条分支赋予层级编码字符串(如 '0' 气管 → '00'/'01' 左右主支气管 → '000'…)，
    #        编码本身即表示气道树的层级(代)结构，供后续 remerge 与标号使用
    airway_topo.grade()

    # ---------- 步骤 3：把每条分支的骨架点写成唯一编号，得到"标号化骨架" cd ----------
    # 注意这里用了链式赋值：skeleton_parse 与 cd 此刻指向同一个数组对象，
    # 后面再对 skeleton_parse 重新 copy() 才会断开这种共享关系
    skeleton_parse = cd = np.zeros(airway_topo.label.shape, dtype=np.int32)
    iii = 1                                  # 分支编号从 1 开始，0 保留给"非骨架/背景"
    for i in airway_topo.Bi:                 # 遍历合并后的每一条分支
        bi = i['member'].copy()              # 分支中间的骨架点列表
        bi.insert(0, i['start'])             # 在头部补上分支起点(通常是分叉节点)
        if 'end' in i:                       # 叶枝才有 end(末端点)，主干分支可能没有
            bi.append(i['end'])
        bi = np.array(bi)                    # 组织成 (N, 3) 的坐标数组
        for u in range(bi.shape[0]):
            # 只在"尚未被占用"的体素上写编号：分叉节点被多条分支共享时，
            # 采用先到先得策略，保证一个体素只属于一个分支，避免标号冲突
            if cd[bi[u, 0], bi[u, 1], bi[u, 2]] == 0:
                cd[bi[u, 0], bi[u, 1], bi[u, 2]] = iii
        iii += 1                             # 下一条分支使用下一个编号

    # ---------- 步骤 4：把分支编号从骨架传播到整个气道区域 ----------
    skeleton_parse = cd.copy()                       # 独立副本，断开与 cd 的共享
    skeleton_parse[np.where(skeleton_parse != 0)] = 1  # 二值化：只保留"是否属于骨架"，供距离变换使用

    # tree_parsing_func 内部：对 (1 - 骨架) 做欧氏距离变换，得到每个体素最近的骨架点坐标 inds，
    # 再用 cd[inds] 把该骨架点的分支编号"回填"到所有气道体素上，并乘以 label 限制在气道前景内
    tree_parsing = tree_parsing_func(skeleton_parse, airway_topo.label, cd)

    return tree_parsing

def tree_parse(label_root, file_path, save_root, save_root2):
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_root2, exist_ok=True)
    file_list = load_json_file(file_path, folder='0', mode=['train'])
    file_list.sort()

    for ids in range(len(file_list)):
        f = file_list[ids] + 'mask_cut.mha'
        # if os.path.exists(os.path.join(save_root, f)):
        #     continue
        label, origin, spacing, direction = load_itk_image(os.path.join(label_root, f))
        # label = label.transpose(2, 1, 0) 
        label = (label > 0).astype(np.uint8)
        label = large_connected_domain26(label)
        # LABEL_TRANS = binary_closing(binary_fill_holes(binary_dilation(label)))
        
        # skeleton = skeletonize_3d(LABEL_TRANS)
        # 【注意事项：两条骨架管线错配，直接影响 BD 指标】
        # 本行存盘(skeleton_*)的骨架提取自"原始 label"(仅二值化+最大连通域+fill_holes)；
        # 而下方存盘(tree_parse_*)的 tree_parsing 的分支编号，来自 airway_parse -> Topology_Tree.sub()
        # 内部对 fill_holes+dilation+closing+maximum_3d 之后的【加工骨架】(见 ours_skel_parse.py L639-642)。
        # 两者几何不一致：metrics.branch_detected_calculation 做 label_skeleton * label_parsing 时，
        # 某分支可能与原始骨架零交集 -> 分母 bincount 对应位为 0 -> 0/0 = NaN
        # (RuntimeWarning: invalid value encountered in divide)，且 NaN >= 0.8 为 False 被静默判未检出，
        # 导致 BD 被系统性低估。若要根治：本行改用加工骨架，或让 sub() 直接对原始 label 提骨架，二者取一并重跑预处理。
        skeleton = skeletonize_3d(label)
        save_itk(skeleton, origin, spacing, direction, os.path.join(save_root2, f))

        tree_parsing = airway_parse(label,merge_t=5)

        print(ids+1, '/', len(file_list), f, "finished!")
        save_itk(tree_parsing, origin, spacing, direction, os.path.join(save_root, f))

def tree_parse_val(label_root,file_path,save_root,save_root2):
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_root2, exist_ok=True)
    file_list = load_json_file(file_path, folder='0', mode=['val'])
    file_list.sort()

    for ids in range(len(file_list)):
        f = file_list[ids] + 'mask_cut.mha'

        label, origin, spacing, direction = load_itk_image(os.path.join(label_root, f))
        # label = label.transpose(2, 1, 0)
        label = (label > 0).astype(np.uint8)
        label = large_connected_domain26(label)
        # LABEL_TRANS = binary_closing(binary_fill_holes(binary_dilation(label)))
        
        # skeleton = skeletonize_3d(LABEL_TRANS)
        # 【注意事项：两条骨架管线错配，直接影响 BD 指标(验证集)】
        # 本行存盘(skeleton_val)的骨架提取自"原始 label"(仅二值化+最大连通域+fill_holes)；
        # 而下方存盘(tree_parse_val)的分支编号来自 airway_parse -> Topology_Tree.sub() 内部
        # 对 fill_holes+dilation+closing+maximum_3d 之后的【加工骨架】(见 ours_skel_parse.py L639-642)。
        # 两者几何不一致：metrics.branch_detected_calculation 做 label_skeleton * label_parsing 时，
        # 某分支可能与原始骨架零交集 -> 分母 bincount 对应位为 0 -> 0/0 = NaN
        # (RuntimeWarning: invalid value encountered in divide)，且 NaN >= 0.8 为 False 被静默判未检出，
        # 导致 BD 被系统性低估。train.py 的 evaluation_case 读的就是这两个目录，错配在此处已注定。
        skeleton = skeletonize_3d(label)
        save_itk(skeleton, origin, spacing, direction, os.path.join(save_root2, f))
        


        tree_parsing = airway_parse(label, merge_t=5)

        print(ids+1, '/', len(file_list), f, "finished!")
        save_itk(tree_parsing, origin, spacing, direction, os.path.join(save_root, f))



def tree_parse_test(label_root,file_path,save_root,save_root2):
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_root2, exist_ok=True)
    file_list = load_json_file(file_path, folder='-1', mode=['test'])
    file_list.sort()

    for ids in range(len(file_list)):
        f = file_list[ids] + 'mask_cut.mha'
        # if os.path.exists(os.path.join(save_root, f)):
        #     continue
        label, origin, spacing, direction = load_itk_image(os.path.join(label_root, f))
        # label = label.transpose(2, 1, 0) 
        label = (label > 0).astype(np.uint8)
        label = large_connected_domain26(label)
        
        # LABEL_TRANS = binary_closing(binary_fill_holes(binary_dilation(label)))
        
        # skeleton = skeletonize_3d(LABEL_TRANS)
        # 【注意事项：两条骨架管线错配，直接影响 BD 指标(测试集)】
        # 本行存盘(skeleton_test)的骨架提取自"原始 label"(仅二值化+最大连通域+fill_holes)；
        # 而下方存盘(tree_parse_test)的分支编号来自 airway_parse -> Topology_Tree.sub() 内部
        # 对 fill_holes+dilation+closing+maximum_3d 之后的【加工骨架】(见 ours_skel_parse.py L639-642)。
        # 两者几何不一致：metrics.branch_detected_calculation 做 label_skeleton * label_parsing 时，
        # 某分支可能与原始骨架零交集 -> 分母 bincount 对应位为 0 -> 0/0 = NaN
        # (RuntimeWarning: invalid value encountered in divide)，且 NaN >= 0.8 为 False 被静默判未检出，
        # 导致 BD 被系统性低估。test.py 的 evaluation_case_test 读的就是这两个目录。
        # 注：上方被注释掉的 LABEL_TRANS 写法正是"加工骨架"版本，若启用可与编号侧管线对齐。
        skeleton = skeletonize_3d(label)
        # skeleton = skeleton.transpose(2, 1, 0)
        save_itk(skeleton, origin, spacing, direction, os.path.join(save_root2, f))

        tree_parsing = airway_parse(label, merge_t=5)

        print(ids+1, '/', len(file_list), f, "finished!")
        save_itk(tree_parsing, origin, spacing, direction, os.path.join(save_root, f))




if __name__ == '__main__':

    # label_root = r'F:\datasets\ATM22\TrainBatch1\after_yxz\mask'
    # file_path = './data/base_dict.json'
    # save_root = './data/tree_parse'
    # save_root2 = './data/skeleton'

    # tree_parse(label_root, file_path, save_root, save_root2)

    # label_root = r'F:\datasets\ATM22\TrainBatch1\after_yxz\mask'
    # file_path = './data/base_dict.json'
    # save_root = './data/tree_parse_val'
    # save_root2 = './data/skeleton_val'

    # tree_parse_val(label_root, file_path, save_root, save_root2)

    label_root = r'F:\datasets\ATM22\TrainBatch1\after\mask'
    file_path = './data/test.json'
    save_root = './data/tree_parse_test'
    save_root2 = './data/skeleton_test'

    tree_parse_test(label_root, file_path, save_root, save_root2)


