import numpy as np
import copy
import pyvista as pv
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize_3d,binary_closing,binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
from scipy.interpolate import interp1d
from scipy import ndimage
import os
import time

from util import *


# 轻量稀疏矩阵：dict{坐标元组 -> 标量}，只为 subsection 里"该骨架点是否存在/是否已访问"的 O(1) 查询服务。
class NDSparseMatrix:
    # 初始化空 elements 字典。
    def __init__(self):
        self.elements = {}

    # 写入一个非零元素：elements[坐标元组] = value，重复键直接覆盖。
    def addValue(self, tuple, value) -> object:
        self.elements[tuple] = value

    # 读取坐标元组处的值；键不存在(稀疏零元)时返回 0，故可直接用于 if 真值判断。
    def readValue(self, tuple):
        try:
            value = self.elements[tuple]
        except KeyError:
            # could also be 0.0 if using floats...
            value = 0
        return value

# 把骨架点集切分成"分支"列表：从 z 最小的点作根，沿 26 邻域 BFS 游走，遇邻居骨架点数>=3 的分支点即截断成枝。
# skeil: (N,3) 骨架点坐标数组(列为 numpy 轴索引)，先装入 NDSparseMatrix 做存在性查询。
# 返回 branch_dict: list[dict]，每支含 'index'(1-based 生成序号)、'start'、'member'(中间点)、
#   'end'(仅终止于分支点时才有)、'fatherindex'(来源分支序号)；列表顺序 = BFS 生成顺序。
# 坑：同一分支点裂出的多支里，l 循环内 member 不逐支清空，第 2 支起的 member 累计包含前支的点；
#     debug!=0(本文件调用传 1)只在 l 循环结束后把 member 重置一次给第 0 支用。下游长度统计受此影响。
def subsection(skeil, debug=0):
    """
    Segments a skeletonized 3D structure into branches and stores information about each branch.

    Args:
        skeil (ndarray): 3D array representing the skeleton.
        debug (int, optional): If non-zero, enables debug mode for additional processing steps. Defaults to 0.

    Returns:
        list: A list of dictionaries, each containing information about a branch (index, start, end, member points, father index).
    
    Updates:
        - branch_dict: List of segmented branches with detailed information.
    """
    skeilvol = NDSparseMatrix()
    for i in range(0, len(skeil)):
        skeilvol.addValue((skeil[i, 0], skeil[i, 1], skeil[i, 2]), 1)
    neigb = [[-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0], [0, 1, 0],
             [1, -1, 0], [1, 0, 0], [1, 1, 0], [-1, -1, -1], [-1, 0, -1],
             [-1, 1, -1], [0, -1, -1], [0, 0, -1], [0, 1, -1], [1, -1, -1],
             [1, 0, -1], [1, 1, -1], [-1, -1, 1], [-1, 0, 1], [-1, 1, 1],
             [0, -1, 1], [0, 0, 1], [0, 1, 1], [1, -1, 1], [1, 0, 1],
             [1, 1, 1]]
    index = np.where(skeil[:, 2] == np.min(skeil[:, 2]))
    s1 = skeil[index[0][0]]
    branchn = 0
    branch_dict = [] # List to store branch information
    visited = NDSparseMatrix() # To track visited points
    visited.addValue((s1[0], s1[1], s1[2]), 1)
    startnode = [[s1[0], s1[1], s1[2], 0]]

    while startnode:
        branch_dictn = {}
        branchn = branchn + 1
        branch_dictn['index'] = branchn
        branch_dictn['start'] = [
            startnode[0][0], startnode[0][1], startnode[0][2]
        ]
        linkstack = []
        member = []
        for i in range(0, 26):
            xi = startnode[0][0] + neigb[i][0]
            yi = startnode[0][1] + neigb[i][1]
            zi = startnode[0][2] + neigb[i][2]
            if skeilvol.readValue((xi, yi, zi)) and not (visited.readValue(
                (xi, yi, zi))):
                linkstack.append([xi, yi, zi])
        if linkstack.__len__() > 1:
            flag = 0
            for l in range(1, linkstack.__len__()):
                branch_dictn = {}
                branchn = branchn + l - 1
                branch_dictn['index'] = branchn
                branch_dictn['start'] = [
                    startnode[0][0], startnode[0][1], startnode[0][2]
                ]
                linkstackk = [linkstack[l - flag]]
                while linkstackk:
                    count = 0
                    neigb_skeil = []
                    for j in range(0, 26):
                        xj = linkstackk[0][0] + neigb[j][0]
                        yj = linkstackk[0][1] + neigb[j][1]
                        zj = linkstackk[0][2] + neigb[j][2]
                        if skeilvol.readValue((xj, yj, zj)):
                            count = count + 1
                            if not visited.readValue((xj, yj, zj)):
                                linkstackk.append([xj, yj, zj])
                                neigb_skeil.append([xj, yj, zj, branchn])

                    visited.addValue(
                        (linkstackk[0][0], linkstackk[0][1], linkstackk[0][2]),
                        1)

                    if count < 3: # Point belongs to current branch
                        member.append((linkstackk[0]))
                    else:
                        branch_dictn['end'] = (linkstackk[0])
                        startnode.extend(
                            neigb_skeil)
                        for k in range(0, len(neigb_skeil)):
                            visited.addValue(
                                (neigb_skeil[k][0], neigb_skeil[k][1],
                                 neigb_skeil[k][2]), 1)
                        break

                    del linkstackk[0]
                branch_dictn['member'] = copy.deepcopy(member)
                branch_dictn['fatherindex'] = startnode[0][3]
                branch_dict.append(branch_dictn)

                del linkstack[l - flag]
                flag = flag + 1
            branch_dictn = {}
            branchn = branchn + 1
            branch_dictn['index'] = branchn
            branch_dictn['start'] = [
                startnode[0][0], startnode[0][1], startnode[0][2]
            ]
            if debug != 0:
                member = []
        while linkstack:
            count = 0
            neigb_skeil = []
            for j in range(0, 26):
                xj = linkstack[0][0] + neigb[j][0]
                yj = linkstack[0][1] + neigb[j][1]
                zj = linkstack[0][2] + neigb[j][2]
                if skeilvol.readValue((xj, yj, zj)):
                    count = count + 1
                    if not visited.readValue((xj, yj, zj)):
                        linkstack.append([xj, yj, zj])
                        neigb_skeil.append([xj, yj, zj, branchn])

            visited.addValue(
                (linkstack[0][0], linkstack[0][1], linkstack[0][2]), 1)

            if count < 3:
                member.append((linkstack[0]))
            else:
                branch_dictn['end'] = (linkstack[0])
                startnode.extend(neigb_skeil)
                for k in range(0, len(neigb_skeil)):
                    visited.addValue((neigb_skeil[k][0], neigb_skeil[k][1],
                                      neigb_skeil[k][2]), 1)
                break

            del linkstack[0]

        branch_dictn['member'] = copy.deepcopy(member)
        branch_dictn['fatherindex'] = startnode[0][3]
        branch_dict.append(branch_dictn)
        del startnode[0]

    return branch_dict

# 由二值气道体积算一条"从根端层指向肺内层"的基准向量 basev，供 find_mainpart_index 判主干。
# 取两层切片 LABEL_TRANS[:, :, z]：order=1 用 maxz-0.1*cha 与 0.6*cha+minz；order=0 用 minz+0.1*cha 与 0.4*cha+minz。
# 每层取前景像素质心；返回 [Δ轴0, Δ轴1, Δ轴2]，z 分量按 order 定号，等价于在 sub() 的工作坐标系里由根指向外。
# 坑：变量名 center*_x/_y 实际是 numpy 轴 0/1 的质心(非物理 x/y)。
def compute_base_vector(LABEL_TRANS, order):
    minz = np.min(np.where(LABEL_TRANS == 1)[2])
    maxz = np.max(np.where(LABEL_TRANS == 1)[2])
    cha = maxz - minz
    if order == 1:
        center1_z = int(maxz - 0.1 * cha)
        zzz2 = 0.6
        center2_z = int(zzz2 * cha + minz)
    else:
        center1_z = int(minz + 0.1 * cha)
        zzz2 = 0.4
        center2_z = int(zzz2 * cha + minz)
    image1 = LABEL_TRANS[:, :, center1_z]
    image2 = LABEL_TRANS[:, :, center2_z]
    center1_x = np.mean(np.argwhere(image1 > 0)[:, 0])
    center1_y = np.mean(np.argwhere(image1 > 0)[:, 1])
    center2_x = np.mean(np.argwhere(image2 > 0)[:, 0])
    center2_y = np.mean(np.argwhere(image2 > 0)[:, 1])

    if order == 1:
        basev = np.array([
            center2_x - center1_x, center2_y - center1_y,
            center1_z - center2_z
        ])
    else:
        basev = np.array([
            center2_x - center1_x, center2_y - center1_y,
            center2_z - center1_z
        ])

    return basev

# 余弦相似度 dot(a,b)/(|a||b|)；任一向量为零范数时结果为 nan(不报错)。用于分支走向与解剖基准向量的比对。
def cosine(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    cosine_sim = dot_product / (norm_a * norm_b)
    return cosine_sim

# 判定气管主干由 Bi 的前 mmm 条分支组成（返回该下标/条数，找不到则 0）。
# maxzzz: 调用处传 B[0,2]，即工作坐标系骨架 z 最大值(≈气道纵向跨度)；分支长 > maxzzz/3.6 视为已出主干直接 break，且只扫前 21 条。
# 对 member>12 的分支收集 (下标, 与basev的cos, 长度)：先跳过 cos<0.928 的(尚未进入主干)，
#   出现 cos>0.928 置 flag 后，再回落到 cos<0.93 的那条即认定主干结束，返回其下标 mmm。
# 坑：0.928~0.93 的空隙使 cos 恰落其间时既不置 flag 也不结束；mmm<=1 时 sub() 不做主干平滑。
def find_mainpart_index(maxzzz,Bi, basev):
    """
    This function processes the 'Bi' structure and finds the index of the main part
    based on cosine similarity with the provided base vector.
    
    Args:
    - Bi: List of airway branch information, where each element contains a 'member' and 'start'.
    - basev: A numpy array representing the base vector for cosine similarity calculation.
    
    Returns:
    - mmm: The index of the main part in 'Bi', or None if no main part is found.
    """
    mainpart = []

    # Collect cosine similarities for branches with more than 20 members
    for i, bi in enumerate(Bi):
        if i > 20:
            break
        if len(bi['member']) == 0:
            continue
        if len(bi['member']) >maxzzz/3.6: 
            break
        v = np.array(bi['member'][-1]) - np.array(bi['start'])

        if len(bi['member']) > 12:
            mainpart.append([i, cosine(basev, v),len(bi['member'])])

    flag = 0
    mmm = 0

    # Find the index of the main part based on cosine similarity
    for index,(i, cos,lll) in enumerate(mainpart):
        if cos < 0.928 and flag == 0:
            continue
        if cos > 0.928:
            flag = 1
        if cos < 0.93 and flag == 1:
            mmm = i
            break

    return mmm

# 把一条 (N,3) 点列(假定按 z 升序)重采样平滑成连续中心线，返回 int 数组(可能短于 N，按 z 降序)。
# 步骤：等距抽 ~3 个关键点 -> interp1d 线性插值回 N 点 -> 逐点钳制与上一点各轴差<=1(保 26 邻域连续)
#   -> 按 z 排序、每个 z 层只留一点去重 -> 反转后再修一次连续性 -> flip 回降序输出。
# 坑：N<3 时 len//inter=0 使 np.arange 步长为 0 报错；去重后长度可能远小于 N，
#     调用方 process_mainairway_points 依赖 len(newmain) 反推旧主干被截掉的点数。
def smooth_points(ori_ps):
    inter = 3
    selected_indices = np.arange(0, len(ori_ps), len(ori_ps) // inter)
    selected_indices = np.append(selected_indices, [len(ori_ps) - 1], axis=0)
    if abs(selected_indices[-2]-selected_indices[-1])<5:
        selected_indices = np.delete(selected_indices, -2)

    selected_points = ori_ps[selected_indices, :] 

    x_interp = interp1d(selected_indices,
                        selected_points[:, 0],
                        kind='linear',
                        fill_value="extrapolate")
    y_interp = interp1d(selected_indices,
                        selected_points[:, 1],
                        kind='linear',
                        fill_value="extrapolate")
    z_interp = interp1d(selected_indices,
                        selected_points[:, 2],
                        kind='linear',
                        fill_value="extrapolate")

    interpolated_points = np.array([
        x_interp(np.linspace(0,
                             len(ori_ps) - 1, len(ori_ps))),
        y_interp(np.linspace(0,
                             len(ori_ps) - 1, len(ori_ps))),
        z_interp(np.linspace(0,
                             len(ori_ps) - 1, len(ori_ps)))
    ]).T

    new_a = np.zeros_like(interpolated_points, dtype=int)

    new_a[0] = np.round(interpolated_points[0]).astype(int)

    for i in range(1, len(interpolated_points)):
        x, y, z = np.round(interpolated_points[i]).astype(int)

        if abs(x - new_a[i - 1][0]) > 1:
            x = new_a[i - 1][0] + np.sign(x - new_a[i - 1][0])
        if abs(y - new_a[i - 1][1]) > 1:
            y = new_a[i - 1][1] + np.sign(y - new_a[i - 1][1])
        if abs(z - new_a[i - 1][2]) > 1:
            z = new_a[i - 1][2] + np.sign(z - new_a[i - 1][2])

        new_a[i] = [x, y, z]

    indices = np.argsort(new_a[:, 2])
    new_a = new_a[indices, :]


    # Step 1: Remove duplicates based on the third column (z)
    unique_z_points = []
    last_z = None
    for point in new_a:
        if point[2] != last_z:
            unique_z_points.append(point)
            last_z = point[2]

    unique_z_points = list(reversed(unique_z_points))

    # Step 2: Ensure continuity between adjacent points
    final_points = [unique_z_points[0]]  # Start with the first point

    for i in range(1, len(unique_z_points)):
        x, y, z = unique_z_points[i]

        # Ensure continuity in x, y, and z within ±1 range from the last point
        prev_x, prev_y, prev_z = final_points[-1]

        # Adjust to ensure the difference is within ±1
        if abs(x - prev_x) > 1:
            x = prev_x + np.sign(x - prev_x)
        if abs(y - prev_y) > 1:
            y = prev_y + np.sign(y - prev_y)
        if abs(z - prev_z) > 1:
            z = prev_z + np.sign(z - prev_z)

        final_points.append([x, y, z])

    final_points = np.array(final_points)
    flipped_points = np.flip(final_points, axis=0)

    return flipped_points


# 用平滑后的中心线替换原始气管主干骨架点，消除骨架锯齿。
# B: (N, 3) 全部骨架点（已按 z 升序）；Bi: 分支列表；mmm: 主干分支条数（find_mainpart_index 的结果）。
# 流程：把前 mmm 条分支的点拼成主干点列并去重 → 按"z 降序位置"排序 → smooth_points 平滑得到 newmain
#      → 从主干尾部截掉 len(newmain) 个点，剩下的旧主干点从 B 中删除
#      → 再把 newmain 逆序回写（rows_to_replace 也是逆序生成）到 B 中被腾出的行上，保证 z 小的在前。
# 维度：返回的 B 仍是 (N', 3)，N' <= N；newmain 与 rows_to_replace 长度相等，能一一对应。
# 风险：np.where 找不到时返回空数组，展开后长度不等，则后半段 newmain 点会静默丢弃；
#      且 B 行位置依赖"旧主干点恰好连续排在末尾"这一隐含假设。
def process_mainairway_points(B, Bi, mmm):
    """
    This function processes the main airway based on input data, modifies airway structure, 
    and returns the updated airway points (B) after smoothing and filtering.

    Args:
    - B: The current airway structure represented as a 2D numpy array.
    - Bi: The airway branch information.
    - mmm: The index of the main airway in Bi (used for iteration limit).

    Returns:
    - B: The updated airway structure after processing.
    """
    mainairway = []
    for i, bi in enumerate(Bi):
        if i >= mmm:
            break
        mainairway.append(bi['start'])
        mainairway += bi['member']
        if 'end' in bi:
            mainairway.append(bi['end'])
    mainairway = np.array(mainairway)
    mainairway = np.unique(mainairway, axis=0)


    fullyB_reversed = np.array([row for row in B[::-1]])
    index_map = {tuple(row): i for i, row in enumerate(fullyB_reversed)}

    # 排序键：按该点在"倒序 B"中的行号给主干点列排序，使其顺序与骨架原始排列一致。
    def sort_key(row):
        """Sort key based on the reversed index map."""
        return index_map[tuple(row)]

    mainairway = np.array(sorted(mainairway, key=sort_key))


    newmain = smooth_points(mainairway)

    cut_main = mainairway[:len(mainairway) - len(newmain)]
    set_b = set(map(tuple, cut_main))
    result = [row.tolist() for row in B if tuple(row) not in set_b]
    B = np.array(result)
    mainairway = mainairway[len(mainairway) - len(newmain):] 

    rows_to_replace = [
        np.where(np.all(B == m, axis=1))[0] for m in reversed(mainairway)
    ]
    rows_to_replace = [item for sublist in rows_to_replace for item in sublist]

    i = len(newmain) - 1
    for index in rows_to_replace:
        B[index, :] = newmain[i]
        i -= 1

    return B

# 分支合并主函数（两阶段）：
# 阶段一：长度(start+member+end 点数) <= len_thre 的短枝，合并进它的第一个子分支
#         （短枝的点列拼到子分支头部，子分支的 fatherindex 改指短枝的父亲），短枝从列表删除；
#         若短枝没有子分支（孤立叶枝）也直接删除其记录 —— 注意这会丢失该段几何。
# 阶段二：找出"只有一个孩子"的父分支，把孩子整个吸收上来（孩子点列接到父 member 尾部），
#         避免树中出现无分叉的冗余节点；剩余孩子的 fatherindex 改指新祖父。
# 维度：输入输出都是 list[dict]，长度只减不增；'index' 字段保持不变（不重新连续编号），
#       所以下游 grade 里 fatherindex(旧 1-based 序号) 与列表下标的对应关系依赖 subsection 的生成顺序。
def merging(branch_dict, len_thre):
    """
    Merges branches from a branch dictionary based on the threshold length and the relationship 
    between parent and child branches. The function operates in two main phases: 
    - It first identifies and merges short branches with their parent branches.
    - Then it handles branches that are single descendants of a parent, merging them accordingly.

    Args:
        branch_dict (list): A list of dictionaries representing branches. Each dictionary contains 
                             information about a branch, such as 'start', 'end', 'member', and 'fatherindex'.
        len_thre (int): A threshold length used to determine whether a branch should be merged. 
                         Branches shorter than or equal to this length may be merged with their parent.

    Returns:
        list: The updated branch dictionary after merging operations.
    """
    cut_list = []
    for i in range(len(branch_dict)):
        bi = branch_dict[i]['member'].copy()
        bi.insert(0, branch_dict[i]['start'])
        if 'end' in branch_dict[i]:
            bi.append(branch_dict[i]['end'])
        if len(bi) <= len_thre:
            sons = 0
            for j in range(i + 1, len(branch_dict)):
                fj = branch_dict[j]['fatherindex']
                if fj == i + 1:
                    sons += 1
                    if sons == 1:
                        cut_list.append(i)
                        branch_dict[j]['fatherindex'] = branch_dict[i][
                            'fatherindex']
                        cuti = branch_dict[i]['member'].copy()
                        if 'end' in branch_dict[i]:
                            cuti.append(branch_dict[i]['end'])
                        cuti.append(branch_dict[j]['start'])
                        branch_dict[j]['start'] = branch_dict[i]['start'].copy(
                        )
                        branch_dict[j][
                            'member'] = cuti + branch_dict[j]['member']
                    if sons > 1:
                        branch_dict[j]['fatherindex'] = branch_dict[i][
                            'fatherindex']
                        cuti = branch_dict[i]['member'].copy()
                        if 'end' in branch_dict[i]:
                            cuti.append(branch_dict[i]['end'])
                        cuti.append(branch_dict[j]['start'])
                        branch_dict[j]['start'] = branch_dict[i]['start'].copy(
                        )
                        branch_dict[j][
                            'member'] = cuti + branch_dict[j]['member']
            if sons == 0:
                cut_list.append(i)
    new_list = [x for i, x in enumerate(branch_dict) if i not in cut_list]
    branch_dict = new_list

    cut_s = []
    child_num = np.zeros(branch_dict[-1]['index'], dtype=int)
    for i in range(len(branch_dict)):
        f = branch_dict[i]['fatherindex']
        child_num[f] += 1
    single = np.where(child_num == 1)
    single = list(single[0])[1:]
    single_index = []
    for s in single:
        for i in range(len(branch_dict)):
            if branch_dict[i]['index'] == s:
                single_index.append(i)
    fs_index = np.zeros((len(single_index), 2), dtype=int)
    for s in range(len(single_index) - 1, -1, -1):
        for i in range(len(branch_dict) - 1, -1, -1):
            if branch_dict[i]['fatherindex'] == branch_dict[
                    single_index[s]]['index']:
                fs_index[s, 0] = branch_dict[i]['fatherindex']
                fs_index[s, 1] = branch_dict[i]['index']
                cut_s.append(i)
                bi = branch_dict[i]['member'].copy()
                bi.insert(0, branch_dict[i]['start'])
                bi.insert(0, branch_dict[single_index[s]]['end'])
                if 'end' in branch_dict[i]:
                    branch_dict[single_index[s]]['end'] = branch_dict[i]['end']
                else:
                    branch_dict[single_index[s]]['end'] = bi[-1]
                    bi = bi[:-1]
                branch_dict[single_index[s]]['member'] = branch_dict[
                    single_index[s]]['member'] + bi
    for s in range(len(fs_index) - 1, -1, -1):
        for i in range(len(branch_dict) - 1, -1, -1):
            if branch_dict[i]['fatherindex'] == fs_index[s, 1]:
                branch_dict[i]['fatherindex'] = fs_index[s, 0]
    new_list = [x for i, x in enumerate(branch_dict) if i not in cut_s]
    branch_dict = new_list

    return branch_dict

# 二次合并：针对指定父层级 remerge_l（如 ['000'] 右肺上叶），只合并该父节点下**最短的那一条**子分支，
# 用于修正 merging 逐层阈值合并后仍残留的异常短枝（典型是上叶尖支被误切）。
# 注意：branch_dict 与 branch_dict_g 必须**同序同长度**（本文件仅在 grade() 之后调用，此时成立），
#       因为这里用同一个下标 i 分别索引两者来取 'fatherindex' / 'index'；若两者长度不一致会静默错位。
# flag>2 的分组（子分支数 >= 3）不参与，避免在真正多分支处错删合法分支。
def remerging(branch_dict,branch_dict_g,remerge_l): 
    
    cut_l=np.zeros(len(remerge_l),dtype=int)+1000
    t=np.zeros(len(remerge_l),dtype=int)+1000
    flag=np.zeros(len(remerge_l),dtype=int)
    for i in range(len(branch_dict)):
        if branch_dict_g[i]['fatherindex'] in remerge_l:
            flag[remerge_l.index(branch_dict_g[i]['fatherindex'])]+=1
            bi = branch_dict[i]['member'].copy()
            bi.insert(0, branch_dict[i]['start'])
            if 'end' in branch_dict[i]:
                bi.append(branch_dict[i]['end'])
            if len(bi)<=t[remerge_l.index(branch_dict_g[i]['fatherindex'])]:
                t[remerge_l.index(branch_dict_g[i]['fatherindex'])]=len(bi)
                cut_l[remerge_l.index(branch_dict_g[i]['fatherindex'])]=i
    cut_l=list(cut_l)
    br3=list(np.where(flag>2)[0])
    cut_l=[n for i, n in enumerate(cut_l) if i not in br3]
    for i in cut_l:
        for j in range(i+1,len(branch_dict)):
            if branch_dict[j]['fatherindex']==branch_dict[i]['index']:
                branch_dict[j]['fatherindex']=branch_dict[i]['fatherindex']
                cuti = branch_dict[i]['member'].copy()
                if 'end' in branch_dict[i]:
                    cuti.append(branch_dict[i]['end'])
                cuti.append(branch_dict[j]['start'])
                branch_dict[j]['start']=branch_dict[i]['start'].copy()
                branch_dict[j]['member']= cuti+branch_dict[j]['member']
    new_list = [x for i, x in enumerate(branch_dict) if i not in cut_l]
    branch_dict=new_list
    return branch_dict

# 把"分支编号图"从细骨架传播到整个气道区域（最近邻/Voronoi 分配）。
# skeleton_parse: (H,W,D) 二值骨架（1=骨架）；label: (H,W,D) 二值气道前景；cd: (H,W,D) 骨架上的分支编号。
# edt(1-骨架) 返回每个体素到最近骨架点的距离与坐标 inds，用 cd[inds] 回填编号，再乘 label 限制在前景内。
# 输出 tree_parsing: (H,W,D) uint16，体素值 = 所属分支编号（0=非气道）。
def tree_parsing_func(skeleton_parse, label, cd):
    #parse the airway tree
    edt, inds = ndimage.distance_transform_edt(1-skeleton_parse, return_indices=True)
    tree_parsing = np.zeros(label.shape, dtype = np.uint16)
    tree_parsing = cd[inds[0,...], inds[1,...], inds[2,...]] * label
    return tree_parsing

class Topology_Tree:
    """
    Class for processing a 3D skeleton representation of a tree-like structure.
    It handles tasks like branch extraction, tree topology analysis, and calculating properties such as branch size and position.

    Attributes:
        label: A 3D binary array representing the segmented tree structure.
        order: Defines the order of the z-axis in the tree's skeletal structure.
               If order == 1, the z-axis is inverted in the tree structure.
               If order == 0 (default), no inversion is applied.
        B: List of coordinates representing the 3D skeleton.
        Bi: List of branches derived from the skeleton.
        o: Coordinates of the tree's origin (mean position of the skeleton points).
        psize: Scaling factors for resizing the skeleton's dimensions.
        merge_t: Threshold used for merging branches in the tree.
        colors: List of colors for visualization.
    """

    # 构造解析对象。LA: (Z,Y,X) 二值气道掩码(0/1)；order: 根端朝向(0=轴2小值端为根, 1=大值端为根, 由调用方按切片最大连通块判定)；
    # merge_t: merging 的短枝阈值(单位=骨架点数, 非毫米)；remerge_l: 二次合并的目标父编码列表(如 ['000'])。
    # 产物字段：B 骨架点(N,3)、Bi 分支表、Bi_g 带 ATM22 编码的分支表、o 坐标原点、psize 轴间距、B_resize/Bi_resize 毫米坐标。
    # 坑：rb23/rb12/l010/lb123/rb45/rb6/rb123/numofzs 是异常分支标志位，本文件只写 rb123 等、从不写 rb23/rb12，
    #     而 tree_parsing.py 只读 rb23/rb12 —— 故那条 remerge 分支实际永不触发(见 regrade 注释)。
    def __init__(self, LA, order, merge_t,remerge_l=[]):
        """
        Initializes the Topology_Tree instance with the given parameters.
        
        Parameters:
            LA: 3D binary array representing the segmented tree structure.
            order: Defines the order of the z-axis in the tree's skeletal structure (0 or 1).
            merge_t: Threshold for merging branches in the tree.
        """
        self.label = LA
        self.order = order
        self.B = []
        self.Bi = []
        self.o = []
        self.psize = []
        self.B_resize = []
        self.Bi_resize = []
        self.merge_t = merge_t
        self.remerge_l=remerge_l
        self.colors = []
        self.rb23=0
        self.rb12=0
        self.rb45=0
        self.rb6=0
        self.lb123=0
        self.l010=0
        self.numofzs=0
        self.rb123=0

    # 骨架提取 + 分支切分 + 主干平滑，结果写入 self.B（(N,3) 骨架点，z 升序）与 self.Bi（分支列表）。
    # 注意：这里传入 skeletonize_3d 的是 LABEL_TRANS（修补后的 label）而不是原始 self.label，
    #       但后续步骤 3 给体素编号时用的坐标索引仍是原始 label 坐标系，两者形状一致故可对齐。
    # order==1 时先把 B 的 z 翻转为"从气管向外"，使 subsection 的根点假设成立；
    #       翻转后的坐标不再对应原始体素，因此必须靠 merge() 里的反向翻转才能回到原坐标系。
    def sub(self):
        """
        Processes the skeletal structure of the tree, including:

        Refines the skeleton with hole-filling, dilation, closing, and skeletonization.
        Analyzes and smooths the centerline of the trachea if necessary, updating the skeleton.
        Extracts branches and updates them after centerline smoothing.
        """
        #中文
        LABEL_TRANS = binary_fill_holes(binary_dilation(self.label))
        LABEL_TRANS = binary_closing(LABEL_TRANS)
        LABEL_TRANS = maximum_3d(LABEL_TRANS)
        skel = skeletonize_3d(LABEL_TRANS)
        B = np.array(np.where(skel != 0))
        B = B[:, B[2].argsort()]
        B = B.T
        ox = np.mean(B[:, 0])
        oy = np.mean(B[:, 1])
        oz = np.mean(B[:, 2])
        self.o = [ox, oy, oz]
        if self.order == 1:
            B[:, 2] = self.label.shape[2] - B[:, 2]
        Bi = subsection(B, debug=1)
        basev = compute_base_vector(LABEL_TRANS, self.order)
        mmm = find_mainpart_index(B[0,2],Bi, basev)
        if mmm > 1:
            B = process_mainairway_points(B, Bi, mmm)
            newBi = subsection(B, debug=1)
            self.B = B
            self.Bi = newBi
        else:
            self.B = B
            self.Bi = Bi

    # 合并短枝(merging)后，把分支坐标从"工作坐标系"翻回原始体素坐标系。
    # order==1 时 sub() 里对 z 做过 shape[2]-z 翻转，这里对每条分支的 start/end/member 逐点做同样翻转(shape2 - z)。
    # 坑：翻转是**原地修改** self.Bi 里的 dict(list 元素共享引用)，所以若先 grade 再 merge 会看到编码与坐标不一致；
    #     调用顺序必须是 sub -> merge -> grade。member 为空时跳过(只判 != [])。
    def merge(self):
        """
        Merges branches in the tree based on the given threshold.
        
        The method processes the branches stored in `Bi` and merges them if their 
        distance is smaller than the threshold defined in `merge_t`. 
        """
        Bi = merging(self.Bi, self.merge_t)
        if self.order == 1:
            for i in range(Bi.__len__()):
                Bi[i]['start'][2] = self.label.shape[2] - Bi[i]['start'][2]
                if 'end' in Bi[i].keys():
                    Bi[i]['end'][2] = self.label.shape[2] - Bi[i]['end'][2]
                if Bi[i]['member'] != []:
                    member = np.array(Bi[i]['member'])
                    member[:, 2] = self.label.shape[2] - member[:, 2]
                    Bi[i]['member'] = member.tolist()
        self.Bi = Bi

    # 给每条分支赋予层级编码字符串（存于 Bi_g，不动 Bi）：
    #   '0' 气管 → '00'/'01' 左/右主支气管 → '000'… 逐层追加 0/1，编码长度即气道树"代"。
    # 依赖隐含约定：Bi[0] 是主干，Bi[1]/Bi[2] 是左右主支气管（用 start 的 y 大小判左/右），
    #   Bi[i]['fatherindex'] 必须在 Bi 中已出现过（否则 string 未赋值会报 NameError）。
    # Bi_g 与 Bi 同序同长度，remerging / regrade 都依赖这一点。
    def grade(self):
        Bi_g = copy.deepcopy(self.Bi)
        flag = np.zeros(self.Bi.__len__(), dtype=np.int8)
        grade = '0'
        Bi_g[0]['index'] = grade
        Bi_g[0]['fatherindex'] = '-1'
        if self.Bi[1]['start'][1] > self.Bi[2]['start'][1]:
            Bi_g[1]['index'] = '01'
            Bi_g[1]['fatherindex'] = '0'
            Bi_g[2]['index'] = '00'
            Bi_g[2]['fatherindex'] = '0'
        else:
            Bi_g[1]['index'] = '00'
            Bi_g[1]['fatherindex'] = '0'
            Bi_g[2]['index'] = '01'
            Bi_g[2]['fatherindex'] = '0'
        for i in range(3, self.Bi.__len__()):
            for g in range(len(self.Bi)):
                if self.Bi[g]['index'] == self.Bi[i]['fatherindex']:
                    string = Bi_g[g]['index'] + str(flag[g])
                    break
            flag[g] += 1
            Bi_g[i]['index'] = string
            Bi_g[i]['fatherindex'] = Bi_g[g]['index']

        self.Bi_g = Bi_g

    # 对 remerge_l 指定父层级执行 remerging，然后重新 grade（编号体系整体刷新）。
    # 前提：self.Bi 与 self.Bi_g 同序同长度（grade 刚执行完即成立）。
    def remerge(self):
        Bi=remerging(self.Bi,self.Bi_g,self.remerge_l)
        self.Bi=Bi
        Topology_Tree.grade(self)

    # 按 ATM22 命名体系逐代细分编码：对每个已定编码层级调用 _process_grade，用解剖基准向量夹角把兄弟分支映射到子编码。
    # 覆盖的层级(编码 -> 解剖含义)：'0'左右主支气管(00/01)、'00'右肺上叶/中间支气管(000/001)、
    #   '000'右上叶三支(0000/0001/0002)、'001'中叶/右下叶(0010/0011)、'0010'中叶段、'00111'/'001111'/'0011111'右下叶底段系、
    #   '01'左上叶/左下叶(010/011)、'010'左上叶分界(0100/0101)、'0101'舌段、'011'左下叶(0110/0111)、'0111'/'01111'左底段系。
    # vectors 的顺序即 haoma 的候选顺序；order==1 时 z 分量整体取反(因 sub() 翻转过 z，方向基准要跟着镜像)。
    # 坑：只改 Bi_g 的 'index'/'fatherindex' 字符串，不动 Bi；异常兜底(_handle_missing_branch)会整体插 '1'，
    #     所以外部按编码取段时必须用**前缀匹配**而非固定长度。本方法不置 rb23/rb12，rb123/l010/lb123/rb45/rb6 置位后无人读。
    def regrade(self):
        
        vectors = [np.array([0, -1, 0]),np.array([0, 1, 0])]
        self._process_grade('0',vectors, self._main_left_right)
        
        if self.order==1:
            vectors = [np.array([0, -1, 0.1]),np.array([0, -1, -1])]
        else:
            vectors = [np.array([0, -1, 0.1]),np.array([0, -1, 1])]
        self._process_grade('00', vectors,self._right)
        
        if self.order==1:
            vectors = [np.array([0, 0, 1]), np.array([-1, -1, 0]),np.array([1, 0, 0])] 
        else:
            vectors = [np.array([0, 0, -1]), np.array([-1, -1, 0]),np.array([1, 0, 0])]
        self._process_grade('000', vectors,self._right_upper)
        
        if self.order==1:
            vectors = [np.array([1, -1, -0.25]), np.array([0, 0, -1])] 
        else:
            vectors = [np.array([1, -1, 0.25]),  np.array([0, 0, 1])]
        self._process_grade('001', vectors,self._right_middle)
        
        vectors = [np.array([0, -1, 0]),np.array([1, 0, 0])]
        self._process_grade('0010', vectors,self._seg0010)
        
        vectors = [np.array([0, 1, 0]),np.array([0, -1, 0])]
        self._process_grade('00111', vectors,self._seg00111)
        
        vectors = [np.array([0, -1,0]),np.array([0, 1, 0])]
        self._process_grade('001111', vectors,self._seg001111)
        
        vectors = [np.array([0, -1,0]),np.array([0, 1, 0])]
        self._process_grade('0011111', vectors,self._seg0011111)

        
        if self.order==1:
            vectors = [np.array([0, 1, 0]),np.array([0, 0.18, -1])]
        else:
            vectors = [np.array([0, 1, 0]),np.array([0, 0.18, 1])]
        self._process_grade('01', vectors,self._left)
        
        if self.order==1:
            vectors = [np.array([0, 0, 1]), np.array([0, 0, -1])]
        else:
            vectors = [np.array([0, 0, -1]),  np.array([0, 0, 1])]
        self._process_grade('010', vectors,self._left_upper)
        
        if self.order==1:
            vectors = [np.array([0, 1, 0]),np.array([1, 0, -1])]
        else:
            vectors = [np.array([0, 1, 0]),np.array([1, 0, 1])]
        self._process_grade('0101', vectors,self._seg0101)
        
        if self.order==1:
            vectors = [np.array([-1, 0, 0]), np.array([0, 0, -1])]
        else:
            vectors = [np.array([-1, 0, 0]),  np.array([0, 0, 1])]
        self._process_grade('011', vectors,self._seg011)
        
        if self.order==1:
            vectors= [np.array([1, 1, 0]),np.array([0,0 , -1])]
        else:
            vectors = [np.array([1, 1, 0]),np.array([0,0 , 1])]
        self._process_grade('0111', vectors,self._seg0111)
        
        vectors= [np.array([0, 1, 0]),np.array([0, -1, 0])]
        self._process_grade('01111', vectors,self._seg01111)

    # 取出 startgrade 下的所有直接子分支（按 index 排序），交给 grade_func 做"按方向区分左右/上下叶"的编号。
    # len(segments) > 1 才处理：独子无需区分方向。vectors 是 2~3 个解剖参考方向，
    # 子分支数必须与 haoma（在 grade_func 里定义）匹配，否则 _update_segment_codes 越界。
    def _process_grade(self, startgrade, vectors,grade_func):
        segments = [seg.copy() for seg in self.Bi_g if seg['fatherindex'] == startgrade]
        segments = sorted(segments, key=lambda x: x['index'])
        if len(segments) > 1:
            grade_func(startgrade, vectors,segments)

    # 气管 '0' 的两个子分支分派左右主支气管编码 '00'/'01'：分支方向(end-start)分别与 ±axis1 向量比余弦。
    # 仅在恰好 2 个子分支(shape[1]==2)时执行；数量不符则整层不重分派，保留 grade() 的数字编码。
    def _main_left_right(self, startgrade,vectors, Blr):
        Blr_values = np.array([self._calculate_similarity(Blr, vector) for vector in vectors])
        haoma = ['00', '01'] 
        if Blr_values.shape[1] == 2 :
            self._update_segment_codes( Blr,Blr_values,haoma)

    # 左主支气管 '01' 的子分支分派：'010' 左上叶支气管 / '011' 左下叶支气管。
    # 参考向量 [0,-1,~±axis2] 指向上叶方向、[0,-1,大幅∓axis2] 指向下叶方向(order=1 时 z 分量镜像)。
    # 若上叶方向最大相似度 <=0.7 判"左上叶缺失/共干"：置 l010=1 并 _handle_missing_branch 给 '01' 全部子孙插 '1'。
    # 坑：l010 置位后全仓库无人读取；_handle_missing_branch 先于 _update_segment_codes 改写 Bi_g，
    #     而 bro 里的 seg['index'] 仍是插号前的旧码，后续前缀匹配只对"恰好仍含旧前缀"的分支生效，二者交互微妙。
    def _left(self, startgrade,vectors, Budr):
        Bud_values = np.array([self._calculate_similarity(Budr, vector) for vector in vectors])
        haoma = ['010', '011'] 
        if max(Bud_values[0,:])<=0.7 or max(Bud_values[:,0])<=0.7:
            self.l010 = 1 
            self._handle_missing_branch(startgrade)
        self._update_segment_codes( Budr,Bud_values,haoma)

    # 左上叶支气管 '010' 的子分支分派：2 支 -> '0100' 上部分(尖后支干) / '0101' 舌段；
    # 3 支 -> 直接分 '01000','01001','0101'（上部分已裂成两支 + 舌段一支，不再递归）。
    # 2 支时若上部分方向相似度 <=0.4 判缺失：置 lb123=1 并 _handle_missing_branch；随后递归 _process_grade('0100') 细分尖支。
    # 参考向量 [-axis0,0,±axis2] 与 [axis0,0,0]（order=1 时首向量 axis2 取 +1）。
    def _left_upper(self, startgrade,vectors, B12345r):
        B12345_values = np.array([self._calculate_similarity(B12345r, vector) for vector in vectors])
        haoma = ['0100', '0101'] 
        if B12345_values.shape[1] ==2:
            if max(B12345_values[0,:])<=0.4:
                self.lb123 = 1 
                self._handle_missing_branch(startgrade)
            self._update_segment_codes( B12345r,B12345_values,haoma)

            if self.order==1:
                vectors = [np.array([-1, 0, 1]), np.array([1, 0, 0])]
            else:
                vectors = [np.array([-1, 0, -1]),  np.array([1, 0, 0])]
            self._process_grade('0100', vectors,self._seg0100)
        elif B12345_values.shape[1] ==3:
            if self.order==1:
                vectors = [np.array([-1, 0, 1]), np.array([1, 0, 0]), np.array([0, 0, -1])]
            else:
                vectors = [np.array([-1, 0, -1]),np.array([1, 0, 0]),  np.array([0, 0, 1])]
            B12345_values = np.array([self._calculate_similarity(B12345r, vector) for vector in vectors])
            haoma=['01000','01001','0101']
            self._update_segment_codes( B12345r,B12345_values,haoma)

    # '0100'(左上叶上部分/尖前干)的子分支分派：2 支 -> '01000' 尖支 / '01001' 前支(外翻支)。
    # 3 支 -> 换用 [-axis0,0,±axis2]、[0,+axis1,∓0.1axis2]、[axis0,0,0] 三个基准重算相似度，分 '01000','01001','01002'（多出的尖上亚段）。
    # 仅处理 2/3 支两种情况，其余数量保持原编码不动。
    def _seg0100(self, startgrade,vectors, B123r):
        B123_values = np.array([self._calculate_similarity(B123r, vector) for vector in vectors])
        haoma = ['01000', '01001'] 
        if B123_values.shape[1] == 2 :
            self._update_segment_codes( B123r,B123_values,haoma)
        elif B123_values.shape[1] ==3:
            if self.order==1:
                vectors = [np.array([-1, 0, 1]), np.array([0, 1, -0.1]),np.array([1, 0, 0])]
            else:
                vectors = [np.array([-1, 0, -1]), np.array([0, 1, 0.1]),np.array([1, 0, 0])]
            B123_values = np.array([self._calculate_similarity(B123r, vector) for vector in vectors])
            haoma=['01000','01001','01002']
            self._update_segment_codes( B123r,B123_values,haoma)

    # '0101'(舌段支气管)的子分支分派：2 支 -> '01010' 舌上支 / '01011' 舌下支。
    # 参考向量 [0,+1,0] 与 [axis0,0,±axis2]；仅恰好 2 支时处理，3 支无对应逻辑。
    def _seg0101(self, startgrade,vectors, B45r):
        B45_values = np.array([self._calculate_similarity(B45r, vector) for vector in vectors])
        haoma = ['01010', '01011'] 
        if B45_values.shape[1] ==2:
            self._update_segment_codes( B45r,B45_values,haoma)

    # '011'(左下叶支气管)的子分支分派：2 支 -> '0110' 背侧段支(上段) / '0111' 基底干。
    # 参考向量 [-axis0,0,0] 与 [0,0,±axis2]（order=1 取 -axis2）；仅恰好 2 支时处理。
    def _seg011(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['0110', '0111'] 
        if B610_values.shape[1] ==2:
            self._update_segment_codes( B610r,B610_values,haoma)

    # '0111'(左下叶基底干)的子分支分派：2 支 -> '01110' 前内基底支干 / '01111' 后外基底支干。
    # 3 支 -> 换用 [axis0,axis1,0]、[0,+0.3axis1,∓axis2]、[0,-0.3axis1,∓axis2] 重算，分 '01110','011110','011111'
    #   （即基底干直接裂成三支时，'01111' 下再分两支）。仅处理 2/3 支。
    def _seg0111(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['01110', '01111'] 
        if B610_values.shape[1] == 2 :
            self._update_segment_codes( B610r,B610_values,haoma)
        elif B610_values.shape[1] ==3:
            if self.order==1:
                vectors= [np.array([1, 1, 0]),np.array([0,0.3 , -1]),np.array([0,-0.3 , -1])]
            else:
                vectors = [np.array([1, 1, 0]),np.array([0,0.3 , 1]),np.array([0,-0.3 , 1])]
            B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
            haoma=['01110','011110','011111']
            self._update_segment_codes( B610r,B610_values,haoma)

    # '01111'(左下叶后外基底干)的子分支分派：2 支 -> '011110' / '011111'（外侧底与后底两支，靠 ±axis1 方向区分）。
    # 仅恰好 2 支时处理。
    def _seg01111(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['011110', '011111'] 
        if B610_values.shape[1] ==2:
            self._update_segment_codes( B610r,B610_values,haoma)

    # 右主支气管 '00' 的子分支分派：'000' 右上叶支气管(支气管上支) / '001' 中间支气管(肺下方向)。
    # 参考向量 [0,-1,±0.1](近水平偏上叶) 与 [0,-1,大幅∓axis2](向下)；order=1 时 axis2 分量镜像。
    # 三分支且恰有一列相似度全 <=0.85 -> 走 _handle_multiple_branches（多出的异常短枝重排编码）；
    # 首行最大值 <=0.85 -> 判右上叶缺失：置 rb123=1 并 _handle_missing_branch。
    # 坑：rb123 置位后无人读取（tree_parsing 只查 rb23/rb12），该异常不会触发 remerge。
    def _right(self, startgrade,vectors, Budr):
        Bud_values = np.array([self._calculate_similarity(Budr, vector) for vector in vectors])
        haoma = ['000', '001'] 
        if Bud_values.shape[1] > 2 and np.where(np.max(Bud_values, axis=0) <= 0.85)[0].size == 1:##
            self._handle_multiple_branches(haoma, Budr, Bud_values, startgrade,vectors)
        elif max(Bud_values[0, :]) <= 0.85:
            self.rb123 = 1 
            self._handle_missing_branch(startgrade)
        elif Bud_values.shape[1] ==2:
            self._update_segment_codes( Budr,Bud_values,haoma)

    # '000'(右上叶支气管)的子分支分派：恰好 3 支时 -> '0000' 尖支 / '0001' 后支 / '0002' 前支。
    # 参考向量 [0,0,∓axis2]、[-axis0,-axis1,0]、[axis0,0,0]（order=1 时首向量 axis2 取反）。
    # 坑：2 支或 4 支均不处理，保留 grade() 的数字编码，下游按 '000x' 前缀取段会落空。
    def _right_upper(self, startgrade,vectors, B123r):
        B123_values = np.array([self._calculate_similarity(B123r, vector) for vector in vectors])
        haoma=['0000','0001','0002']
        if B123_values.shape[1] ==3 :
            self._update_segment_codes(B123r,B123_values,haoma)

    # '001'(中间支气管)的子分支分派：2 支 -> '0010' 中叶支气管 / '0011' 右下叶支气管。
    # 2 支时若中叶方向相似度 <=0.5 判中叶缺失：置 rb45=1 并 _handle_missing_branch；随后递归 _process_grade('0011') 细分右下叶。
    # 3 支 -> 用 [axis0-0.7axis1, -axis0, 0,-0.4axis1∓axis2] 分 '0010','00110','00111'
    #   （即中叶、右下叶上段支、基底干在同一分叉处直接三分，跳过 _seg0011 层级）。
    def _right_middle(self, startgrade,vectors, B45r):
        B45_values = np.array([self._calculate_similarity(B45r, vector) for vector in vectors])
        haoma = ['0010', '0011'] 
        if B45_values.shape[1] ==2:
            if max(B45_values[0, :]) <= 0.5: 
                self.rb45 = 1 
                self._handle_missing_branch(startgrade)
            self._update_segment_codes( B45r,B45_values,haoma)

            if self.order==1:
                vectors = [np.array([-1, -0.1, 0]), np.array([0, 0, -1])]
            else:
                vectors = [np.array([-1, -0.1, 0]),  np.array([0, 0, 1])]
            self._process_grade('0011', vectors,self._seg0011)
        elif B45_values.shape[1] ==3:
            if self.order==1:
                vectors = [np.array([1, -0.7, 0]), np.array([-1, 0, 0]), np.array([0, -0.4, -1])]
            else:
                vectors = [np.array([1, -0.7, 0]), np.array([-1, 0, 0]),  np.array([0, -0.4, 1])]
            B45_values = np.array([self._calculate_similarity(B45r, vector) for vector in vectors])
            haoma=['0010','00110','00111']
            self._update_segment_codes( B45r,B45_values,haoma)

    # '0011'(右下叶支气管)的子分支分派：2 支 -> '00110' 上段支(背侧段) / '00111' 基底干。
    # 参考向量 [-axis0,-0.1axis1,0] 与 [0,0,±axis2]（order=1 取 -axis2）；若上段方向相似度 <=0.5 判缺失：
    # 置 rb6=1 并 _handle_missing_branch（rb6 同样无人读取）。仅恰好 2 支时处理，3 支不处理。
    def _seg0011(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['00110', '00111'] 
        if B610_values.shape[1] ==2:
            if max(B610_values[0, :]) <= 0.5:
                self.rb6 = 1
                self._handle_missing_branch(startgrade)
            self._update_segment_codes( B610r,B610_values,haoma)

    # '0010'(中叶支气管)的子分支分派：2 支 -> '00100' 外侧段 / '00101' 内侧段。
    # 参考向量 [0,-axis1,0] 与 [axis0,0,0]；仅恰好 2 支时处理。
    def _seg0010(self, startgrade,vectors, B45r):
        B45_values = np.array([self._calculate_similarity(B45r, vector) for vector in vectors])
        haoma = ['00100', '00101'] 
        if B45_values.shape[1] == 2 :
            self._update_segment_codes( B45r,B45_values,haoma)

    # '00111'(右下叶基底干)的子分支分派：2 支 -> '001110' 前内基底支干 / '001111' 后外基底支干。
    # 3 支 -> 换用 [0,-axis1,0]、[0,-0.1axis1,∓axis2]、[0,+0.3axis1,∓axis2] 重算，
    #   注意 haoma 顺序是 ['0011110','0011111','001110']（第 1 向量对应前内基底 '001110'，排列刻意错位）。仅处理 2/3 支。
    def _seg00111(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['001110', '001111'] 
        if B610_values.shape[1] == 2 :
            self._update_segment_codes( B610r,B610_values,haoma)
        elif B610_values.shape[1] ==3:
            if self.order==1:
                vectors= [np.array([0, -1, 0]),np.array([0,-0.1 , -1]),np.array([0,0.3 , -1])]
            else:
                vectors = [np.array([0, -1, 0]),np.array([0,-0.1, 1]),np.array([0,0.3 , 1])]
            B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
            haoma=['0011110','0011111','001110']
            self._update_segment_codes( B610r,B610_values,haoma)

    # '001111'(右下叶后外基底支干)的子分支分派：2 支 -> '0011110' 后基底 / '0011111' 外基底。
    # 3 支 -> 换用 [0,-axis1,0]、[0,-0.4axis1,∓axis2]、[0,+0.2axis1,∓axis2] 重算，分 '0011110','00111110','00111111'。仅处理 2/3 支。
    def _seg001111(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['0011110', '0011111'] 
        if B610_values.shape[1] == 2 :
            self._update_segment_codes( B610r,B610_values,haoma)
        elif B610_values.shape[1] ==3:
            if self.order==1:
                vectors= [np.array([0, -1, 0]),np.array([0,-0.4 , -1]),np.array([0,0.2 , -1])]
            else:
                vectors = [np.array([0, -1, 0]),np.array([0,-0.4, 1]),np.array([0,0.2 , 1])]
            B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
            haoma=['0011110','00111110','00111111']
            self._update_segment_codes( B610r,B610_values,haoma)

    # '0011111'(右下叶外基底支)的子分支分派：2 支 -> '00111110' / '00111111'（靠 ±axis1 区分）。仅恰好 2 支时处理。
    def _seg0011111(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['00111110', '00111111'] 
        if B610_values.shape[1] == 2 :
            self._update_segment_codes( B610r,B610_values,haoma)

    # 算一组兄弟分支与参考向量的余弦相似度。分支方向取 end - start（叶枝无 end 则用最后一个 member）。
    # 返回 list[float]，长度 = len(segments)，即后续 similarity_values 的**列数 = 分支个数**，
    # 而 vectors 个数决定**行数** —— 这就是各 _xxx 方法里用 shape[1] == 2/3 判分支数的依据。
    def _calculate_similarity(self,segments,vector):
        similarities = []
        for seg in segments:
            start_point = np.array(seg['start'])
            end_point = np.array(seg['end'] if 'end' in seg else seg['member'][-1])
            similarities.append(cosine(end_point-start_point, vector))
        return similarities
    
    # 处理某父节点下"兄弟分支多于参考向量数"的异常(如 '00' 下多出假短枝)：
    # 1) wrongb = 与所有向量相似度都 <=0.75 的分支列(判为异常枝)，跳过不赋新码；
    # 2) 其余分支按 haoma 弹出顺序依次重命名，子孙 index/fatherindex 用前缀替换跟着改(viewed 防重复)；
    # 3) 重新收集该层 segments 算相似度、删掉 wrongb 列，若两列的 argmax 位置与期望(0/1)不符则调 _exchange_grade 对调。
    # 坑：_exchange_grade 在本文件中**未定义**，走到最后一步会 AttributeError；haoma 是传入的共享 list 会被 pop 破坏。
    def _handle_multiple_branches(self, haoma, segments, segment_values, startgrade,vectors):
        viewed = []
        wrongb = list(np.where(np.max(segment_values, axis=0) <= 0.75)[0])##33
        for i, seg in enumerate(segments):
            if i in wrongb:
                continue
            newgrade = haoma[0]
            haoma.pop(0)
            if newgrade == seg['index']:
                continue
            for j in range(len(self.Bi_g)):
                if self.Bi_g[j]['index'][:len(seg['index'])] == seg['index'] and self.Bi_g[j]['index'] != seg['index'] and j not in viewed:
                    viewed.append(j)
                    numj = self.Bi_g[j]['index'][len(seg['index']):]
                    self.Bi_g[j]['index'] = newgrade + numj
                    numfj = self.Bi_g[j]['fatherindex'][len(seg['index']):]
                    self.Bi_g[j]['fatherindex'] = newgrade + numfj
            seg['index'] = newgrade
        segments = [seg.copy() for seg in self.Bi_g if seg['fatherindex'] == startgrade]
        segments = sorted(segments, key=lambda x: x['index'])
        segment_values = np.array([self._calculate_similarity(segments, vector) for vector in vectors])
        segment_values = np.delete(segment_values, wrongb, axis=1)
        if np.argmax(segment_values[:, 0]) != 0 and np.argmax(segment_values[:, 1]) != 1:
            self._exchange_grade(startgrade, segments)

    # 缺失分支兜底：当某层应有的解剖分支没检到（相似度太低），把该层所有子孙编号统一插入 '1'
    #   （如 '0010' -> '00110'），保持编码体系与 ATM22 标注约定对齐。
    def _handle_missing_branch(self, startgrade):
        for j in range(len(self.Bi_g)):
            if self.Bi_g[j]['index'][:len(startgrade)] == startgrade and self.Bi_g[j]['index'] != startgrade:
                self.Bi_g[j]['index'] = startgrade + str(1) + self.Bi_g[j]['index'][len(startgrade):]
                self.Bi_g[j]['fatherindex'] = startgrade + str(1) + self.Bi_g[j]['fatherindex'][len(startgrade):]
    
    # 按相似度给兄弟分支分配解剖编号 haoma（如 ['00','01']），并同步改写它们及其所有子孙的 index/fatherindex。
    # 约束：len(haoma) 必须等于 similarity_values.shape[0]（参考向量个数）也等于分支个数，
    #       否则下面 haoma[idx] 会 IndexError —— 这就是本文件反复用 shape[1] 做分支数判别的根本原因。
    # 子孙递归：用前缀匹配 seg['index'] 改写为 newgrade + 后缀，保证整棵子树跟着换编号；viewed 防重复改写。
    def _update_segment_codes(self,bro, similarity_values, haoma):
        viewed = []
        # Initialize grades and assignment states
        new_grades = [None] * len(bro)
        assigned = [False] * len(haoma)
        used_codes = set()
        remaining_grades = list(range(len(bro)))

        # Assign codes dynamically
        while remaining_grades:
            # Sort remaining segments by max similarity
            remaining_grades.sort(key=lambda i: -max(similarity_values[:, i]))
            current = remaining_grades.pop(0)

            # Assign the most suitable code
            sorted_indices = np.argsort(-similarity_values[:, current])
            for idx in sorted_indices:
                candidate_code = haoma[idx]
                if not assigned[idx] or len(remaining_grades) == len(haoma) - len(used_codes):
                    new_grades[current] = candidate_code
                    assigned[idx] = True
                    used_codes.add(candidate_code)
                    break

        # Update segment codes in Bi_g
        for i, seg in enumerate(bro):
            newgrade = new_grades[i]
            if newgrade == seg['index']:
                continue

            for j in range(len(self.Bi_g)):
                if self.Bi_g[j]['index'][:len(seg['index'])]==seg['index'] and self.Bi_g[j]['index']!=seg['index'] and j not in viewed: ###### B8 9 10    and j not in viewed
                    viewed.append(j)                        
                    numj=self.Bi_g[j]['index'][len(seg['index']):]
                    self.Bi_g[j]['index']=newgrade+numj
                    numfj=self.Bi_g[j]['fatherindex'][len(seg['index']):]
                    self.Bi_g[j]['fatherindex']=newgrade+numfj
                if self.Bi_g[j]['index']==seg['index'] and j not in viewed:
                    self.Bi_g[j]['index']=newgrade  
                    viewed.append(j)

    # 把骨架/分支坐标从"体素索引空间"转到"以树重心 o 为原点、按物理间距放缩的空间"：(coord - o) * psize。
    # 【轴序陷阱】B 的列是 numpy 轴索引；SimpleITK 恒有 numpy axis_i ↔ spacing[2-i]
    #   （GetArrayFromImage 把数组反序，GetSpacing 不反序；permuteAxes 同步重排两者，该关系不变）。
    #   因此只有调用方传入"按 numpy 轴排好的间距"（即 spacing 反序后的 px,py,pz，
    #   见 tree_parsing.py 的 px,py,pz = spacing[2],spacing[1],spacing[0]）时，逐轴配对才正确；
    #   若直接传 GetSpacing() 原序，各向异性数据会把 axis0/axis2 的间距互换（axis1 恒对）。
    #   佐证：preprocessing.crop_origin_yxz 中 ZXY 序的 spacing/direction 用 [1,2,0] 还原物理 XYZ，
    #   而 YXZ 数组序的 start 索引须用 [1,0,2]（两者内部顺序不同、置换不同；旧版曾误对 start 也
    #   用 [1,2,0] 致裁剪原点 Y/Z 互换，已修复，对照验证见 verify_crop_origin_yxz.py）。
    # 维度：B (N,3) int -> B_resize (N,3) float32；Bi_resize 中 start/end 是 [x,y,z] 列表，
    #       member 是 list[list]，转成 (M,3) 数组放缩后再 tolist() 回写。
    # 注意：只改坐标不改拓扑，Bi_resize 与 Bi **同序同长度**，所以下游能用分支号 k 直接索引 Bi_resize[k-1]。
    # 小瑕：前两行先 astype 了 B_resize，但计算时读的是 self.B（原值），结果等价；dir 参数名遮蔽了内置 dir。
    # 【体素坐标 → 相对毫米坐标 的换算步骤】
    # 把 sub/merge 阶段产出的骨架点(B)和分支表(Bi)整体平移+缩放：
    #   new = (voxel_index - o) * psize
    # o 是体素包围盒最小角(不是 ITK 物理 origin)，所以结果是"以包围盒角为原点、
    # 单位毫米"的相对坐标，只用于几何可视化/长度计算，不能直接叠回 ITK 世界坐标
    # (缺 origin 平移和 direction 旋转)。
    # 注意命名陷阱：px/py/pz 不是"对应 x/y/z 轴"，而是"对应 numpy 轴 0/1/2"的间距。
    # 调用方 tree_parsing.ours_skel_parse 传入前已反序: px,py,pz = spacing[2],spacing[1],spacing[0]，
    # 因此这里 轴0(Z)*px(=z向spacing)、轴1(Y)*py、轴2(X)*pz(=x向spacing) 的配对是正确的。
    # 本方法是可选步骤：ske_and_parse.airway_parse 那条链从不调用它，全程留在体素空间。
    def resize(self, px, py, pz,dir=None):
        """
        Resizes the branches and skeleton coordinates based on the given scaling factors.

        Parameters:
            px: 轴 0(numpy 的 Z 轴, 切片方向)的体素间距, 毫米
            py: 轴 1(numpy 的 Y 轴)的体素间距, 毫米
            pz: 轴 2(numpy 的 X 轴, 列方向)的体素间距, 毫米
            dir: 可选输出路径; 非空时把换算后的分支表 Bi_resize 存成 .npy
                 (即 tree_parsing 流程里的 *_parse.npy 中心线文件)

        产出: self.B_resize (N,3 float) 与 self.Bi_resize (逐分支 start/end/member 毫米坐标),
        原 B/Bi 保持不变; self.psize 同时被记录, 供 recons/sub_model 里 marching cubes
        顶点的缩放复用。
        """
        self.Bi_resize = copy.deepcopy(self.Bi)   # 深拷贝：毫米版不能污染体素版原始数据
        self.B_resize = copy.deepcopy(self.B)
        self.psize = [px, py, pz]                 # 存下间距, recons 等后续方法还要用
        self.B_resize = self.B_resize.astype(np.float32)  # 毫米是小数, 必须先转 float 否则截断
        # 骨架点整体: 先平移到包围盒原点, 再逐轴乘各自间距 (列号=轴号, 与 psize 下标配对)
        self.B_resize[:, 0] = (self.B[:, 0] - self.o[0]) * self.psize[0]
        self.B_resize[:, 1] = (self.B[:, 1] - self.o[1]) * self.psize[1]
        self.B_resize[:, 2] = (self.B[:, 2] - self.o[2]) * self.psize[2]
        for i in range(self.Bi_resize.__len__()):
            self.Bi_resize[i]['start'][0] = (self.Bi_resize[i]['start'][0] -
                                             self.o[0]) * self.psize[0]
            self.Bi_resize[i]['start'][1] = (self.Bi_resize[i]['start'][1] -
                                             self.o[1]) * self.psize[1]
            self.Bi_resize[i]['start'][2] = (self.Bi_resize[i]['start'][2] -
                                             self.o[2]) * self.psize[2]
            if 'end' in self.Bi_resize[i].keys():
                self.Bi_resize[i]['end'][0] = (self.Bi_resize[i]['end'][0] -
                                               self.o[0]) * self.psize[0]
                self.Bi_resize[i]['end'][1] = (self.Bi_resize[i]['end'][1] -
                                               self.o[1]) * self.psize[1]
                self.Bi_resize[i]['end'][2] = (self.Bi_resize[i]['end'][2] -
                                               self.o[2]) * self.psize[2]
            if self.Bi_resize[i]['member'] != []:
                member = np.array(self.Bi_resize[i]['member'])
                member = member.astype(np.float32)
                member[:, 0] = (member[:, 0] - self.o[0]) * self.psize[0]
                member[:, 1] = (member[:, 1] - self.o[1]) * self.psize[1]
                member[:, 2] = (member[:, 2] - self.o[2]) * self.psize[2]
                self.Bi_resize[i]['member'] = member.tolist()
        if dir:
            np.save(dir, np.array(self.Bi_resize))
    
    # 用 marching cubes 从二值 label 提取气道表面网格并平滑、保存。
    # 维度：verts (V,3) 为体素轴索引坐标（axis0/1/2 顺序与 numpy 一致，故可与 self.o/psize 直接相减放缩）；
    #       faces (F,3) 顶点索引，前面加一列常数 3 变成 pyvista 需要的 (F,4) 格式（[3,i,j,k]）。
    # iso=0.95 对 0/1 体数据等价于取 0.5 以上的等值面，即前景外表面。
    def recons(self, dir):
        """
        Reconstructs the airway tree surface using marching cubes algorithm.

        Parameters:
            dir: The directory where the reconstructed mesh will be saved.
        
        This method uses the marching cubes algorithm to generate a 3D mesh 
        representing the airway tree's surface and saves it in the specified directory.
        """
        iso = 0.95
        verts, faces, _, _ = marching_cubes(self.label, iso)
        verts[:, 0] = verts[:, 0] - self.o[0]
        verts[:, 1] = verts[:, 1] - self.o[1]
        verts[:, 2] = verts[:, 2] - self.o[2]
        verts[:, 0] = verts[:, 0] * self.psize[0]
        verts[:, 1] = verts[:, 1] * self.psize[1]
        verts[:, 2] = verts[:, 2] * self.psize[2]
        faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
        mesh_airway = pv.PolyData(verts, faces)
        mesh_airway_smooth = mesh_airway.smooth(relaxation_factor=0.2)
        pl = pv.Plotter()
        pl.add_mesh(mesh_airway_smooth, color='#E96C6F', style='surface')
        mesh_airway_smooth.save(dir)
    # 逐分支重建 3D 模型并输出旋转 GIF 动画 + 静态截图，用于直观检查树解析结果。
    # 内部重复了 airway_parse 的步骤 3/4（骨架编号 cd -> tree_parsing），所以必须与 Bi 同序同长度：
    #   分支号 k 从 1 开始递增，对应 Bi[k-1] 与 Bi_resize[k-1]、colors[k-1]，三者靠下标对齐。
    # 每条分支单独取 (tree_parsing == k) 做 marching cubes，再平移到 resize 后的物理坐标系。
    # 返回 end_time（解析结束时间戳），st 为开始时间戳，调用处用差值统计耗时。
    def sub_model(self, st, save_dir, case):
        """
        Visualizes and saves a 3D model of the airway tree with branches.

        Parameters:
            st: The start time to compute the duration of airway tree parsing.
            save_dir: Directory to save the output visualization and animation.
            case: The case identifier used to load the relevant files.
        
        This method generates a 3D visualization of the airway tree with its branches 
        and saves the result as an STL file and GIF animation.
        """
        if self.colors == []:
            color_list = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
                'C', 'D', 'E', 'F'
            ]
            for i in range(400):
                color = ''
                for i in range(6):
                    color_number = color_list[random.randint(0, 15)]
                    color += color_number
                color = '#' + color
                self.colors.append(color)

        mesh_airway = pv.read(
            os.path.join(save_dir,
                         case.split('.mha')[0] + '.stl'))
        # pl = pv.Plotter(off_screen=True)
        # pl.add_mesh(mesh_airway, color='white', style='surface', opacity=0.4)
        skeleton_parse = cd = np.zeros(self.label.shape, dtype=np.int32)
        num = len(self.Bi)
        iii = 1
        for i in self.Bi:
            bi = i['member'].copy()
            bi.insert(0, i['start'])
            if 'end' in i:
                bi.append(i['end'])
            bi = np.array(bi)
            for u in range(bi.shape[0]):
                if cd[bi[u, 0], bi[u, 1], bi[u, 2]] == 0:
                    cd[bi[u, 0], bi[u, 1], bi[u, 2]] = iii
            iii += 1

        skeleton_parse = cd.copy()
        skeleton_parse[np.where(skeleton_parse != 0)] = 1
        tree_parsing = tree_parsing_func(skeleton_parse, self.label, cd)
        end_time = time.time()


        pl = pv.Plotter(off_screen=True)
        pl.open_gif(os.path.join(save_dir, case.split('.mha')[0] + '.gif'))
        points = []
        labels = []
        for k in range(1, num + 1):
            if np.sum(tree_parsing == k)==0:
                continue
            points.append(self.Bi_resize[
                k - 1]['member'][len(self.Bi_resize[k - 1]['member']) //
                                 2] if self.Bi_resize[k - 1]['member'] !=
                          [] else self.Bi_resize[k - 1]['end'])
            labels.append(self.Bi_resize[k - 1]['index'])
            iso = 0.95
            verts, faces, _, _ = marching_cubes(tree_parsing == k, iso)
            verts[:, 0] = verts[:, 0] - self.o[0]
            verts[:, 1] = verts[:, 1] - self.o[1]
            verts[:, 2] = verts[:, 2] - self.o[2]
            verts[:, 0] = verts[:, 0] * self.psize[0]
            verts[:, 1] = verts[:, 1] * self.psize[1]
            verts[:, 2] = verts[:, 2] * self.psize[2]
            faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
            mesh_seg = pv.PolyData(verts, faces)
            mesh_seg = mesh_seg.smooth(relaxation_factor=0.15)
            pl.add_mesh(mesh_seg, color=self.colors[k - 1], style='surface')  #

        n_frames = 60
        for i in range(n_frames):
            pl.camera_position = 'yz'   # 冠状面: YXZ 数据沿 axis0(物理AP)看, up=axis2(物理HF), 树竖直
            pl.camera.azimuth = 360 * (i / n_frames)
            pl.render()
            pl.write_frame()
        pl.close()

        pl = pv.Plotter(off_screen=True)
        points = []
        labels = []
        for k in range(1, num + 1):
            if np.sum(tree_parsing == k)==0:
                continue
            points.append(self.Bi_resize[
                k - 1]['member'][len(self.Bi_resize[k - 1]['member']) //
                                 2] if self.Bi_resize[k - 1]['member'] !=
                          [] else self.Bi_resize[k - 1]['end'])
            labels.append(self.Bi_resize[k - 1]['index'])
            iso = 0.95
            verts, faces, _, _ = marching_cubes(tree_parsing == k, iso)
            verts[:, 0] = verts[:, 0] - self.o[0]
            verts[:, 1] = verts[:, 1] - self.o[1]
            verts[:, 2] = verts[:, 2] - self.o[2]
            verts[:, 0] = verts[:, 0] * self.psize[0]
            verts[:, 1] = verts[:, 1] * self.psize[1]
            verts[:, 2] = verts[:, 2] * self.psize[2]
            faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
            mesh_seg = pv.PolyData(verts, faces)
            mesh_seg = mesh_seg.smooth(relaxation_factor=0.15)
            pl.add_mesh(mesh_seg, color=self.colors[k - 1], style='surface')  #


        pl.camera_position = 'yz'   # 冠状面(YXZ 数据: 沿 axis0=AP 看, HF 竖直)
        pl.show(screenshot=os.path.join(save_dir, case.split('.mha')[0] + '_model.png'))

        return end_time

    # 把所有分支的中心线折线叠加在半透明气道表面上截图保存（俯视 yz 方向）。
    # 维度细节：Bi_resize 的坐标已是物理空间 float；add_lines 要求顶点数为偶数（两点成线段），
    #   故对奇数长度的 bi 先丢掉最后一个点。colors[kk] 与 Bi_resize[kk] 按下标一一对应。
    def show_line1(self, save_dir, case):
        """
        Visualizes and saves the 3D airway mesh and segmented branches.

        Draws segmented branches with unique colors and adds them to the plot.
        Saves a screenshot of the visualization in the given directory.

        Parameters:
            save_dir (str): Directory to save the output screenshot.
            case (str): Case name used for the file paths.
        """
        if self.colors == []:
            color_list = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
                'C', 'D', 'E', 'F'
            ]
            for i in range(600):
                color = ''
                for i in range(6):
                    color_number = color_list[random.randint(0, 15)]
                    color += color_number
                color = '#' + color
                self.colors.append(color)

        mesh_airway = pv.read(
            os.path.join(save_dir,
                         case.split('.mha')[0] + '.stl'))
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(mesh_airway, color='white', style='surface', opacity=0.4)
        for kk in range(len(self.Bi_resize)):
            i = self.Bi_resize[kk]
            bi = i['member'].copy()
            bi.insert(0, i['start'])
            if 'end' in i:
                bi.append(i['end'])
            bi = np.array(bi)
            if bi.shape[0] % 2 != 0:
                bi = bi[:-1, :]
            pl.add_lines(bi, color=self.colors[kk], width=5)
        pl.background_color = "white"
        pl.view_yz()   # 冠状面: YXZ 数据沿 axis0(物理AP)看, up=axis2(物理HF); 注意 ZYX 数据会俯视, 相机需随轴序变
        screenshot_filename = os.path.join(save_dir,
                                           case.split('.mha')[0] + '.png')
        pl.screenshot(screenshot_filename)
        pl.close()
