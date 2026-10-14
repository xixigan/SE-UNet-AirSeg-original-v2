# -*- coding: utf-8 -*-
"""
气道“树解析”模块
================
输入：一张气道二值分割掩码 pred(0/1) + 体素间距 spacing。
输出：把气道从“一堆体素”还原成一棵“带层级拓扑的树”——
     即主干/分支的父子关系、分级(grade)、中心线、三维重建面片。
本文件提供两套解析算法，供对比：
  ours_skel_parse  : 本项目自研，基于 ours_skel_parse.Topology_Tree 类
  atm22_skel_parse : ATM22 官方基线，基于 skeleton_parsing + tree_parsing_func
两者都会产出可视化(.stl/.png/.gif)与计时文本，便于人工检查与效率对比。
"""
import argparse
import os
import numpy as np
import skimage.measure as measure
import time
import pyvista as pv                        # 三维网格/中心线可视化
from skimage.measure import marching_cubes  # 从体素提取等值面(表面网格)
from skimage.morphology import skeletonize_3d  # 3D 骨架化(细化成单像素中心线)
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path

from util import *                        # load_itk_image / large_connected_domain 等工具
from atm22_skel_parse import *            # skeleton_parsing / tree_parsing_func 等基线函数
from ours_skel_parse import Topology_Tree # 自研树对象(核心数据结构)


def ours_skel_parse(pred, spacing, merge_t, save_dir, case):
    """
    自研树解析主流程。

    :param pred    : 气道二值掩码, 形状 (Z,Y,X)
    :param spacing : 体素物理间距(毫米), 仅用于 resize 把体素坐标换算成真实长度(可视化/建模)
    :param merge_t : 分支合并阈值, 单位是【骨架点数】而非毫米: merging() 里直接比较
                     len(start+member+end) <= merge_t, 全程不乘 spacing, 体素各向异性时
                     对应的物理长度因分支走向而异
    :param save_dir: 产物输出目录
    :param case    : 病例文件名(带 .mha), 内部用 split 去掉后缀作为输出名
    :return airway_topo: 已解析完成的 Topology_Tree 对象
    """
    start_time = time.time()
    # spacing 是 ITK (x,y,z) 顺序; 而 B/verts 的列是 numpy 轴索引(GetArrayFromImage 反序)。
    # 恒等关系: numpy axis_i ↔ spacing[2-i], 与是否 permuteAxes 无关, 故必须反序传入 resize
    px, py, pz = spacing[2], spacing[1], spacing[0]   # 按 numpy 轴序排好的间距, resize 换毫米坐标用
    # 找出气道在 z 轴(切片方向)上的起止切片: 只统计 pred==1 的体素所在层
    minz = np.min(np.where(pred == 1)[2])   # 最靠“上端”的气道层
    maxz = np.max(np.where(pred == 1)[2])   # 最靠“下端”的气道层
    cha = maxz - minz                        # 气道在 z 方向跨越的层数(总高度)
    # ---- 判定树的“朝向/根端” order ----
    # 原理: 靠近气管/隆突的一端, 单层切片里气道往往是“少数粗大连通块”(面积大);
    #       靠近肺外周的一端则被分成“许多细小分支”(最大连通块面积小)。
    #       比较两端切片的最大连通块大小, 就能推断哪端是“根”, 供后续从根开始分级。
    oneslice = pred[:, :, int(0.8 * cha + minz)]   # 取靠 maxz 端(80% 高度处)的单层切片
    imLabel, _ = measure.label(oneslice, background=0, return_num=True)  # 2D 连通域标记
    noback = np.bincount(imLabel.reshape(-1))      # 统计每个标签(连通块)的体素数
    noback[0] = 0                                  # 标签 0 是背景, 置 0 排除
    max_num8 = np.max(noback)                      # 该层最大连通块面积
    oneslice = pred[:, :, int(0.2 * cha + minz)]   # 取靠 minz 端(20% 高度处)的单层切片
    imLabel, _ = measure.label(oneslice, background=0, return_num=True)
    noback = np.bincount(imLabel.reshape(-1))
    noback[0] = 0
    max_num2 = np.max(noback)
    # 20% 端最大块更大 -> 认为根在 20% 侧, order=0; 否则根在 80% 侧, order=1
    if max_num2 > max_num8:
        order = 0
    else:
        order = 1

    remerge_l=['000']   # 供 remerge 阶段使用的标记/配置(占位初值)
    # 用掩码、朝向、合并阈值构造树对象; Bi 是内部“分支集合(branch info)”的核心结构
    airway_topo = Topology_Tree(pred, order, merge_t,remerge_l)
    airway_topo.sub(
    )  #骨架化+分割: 得到中心线并切出各分支, 填充 airway_topo.Bi
    airway_topo.merge(
    )  #合并/剔除过短的叶片与碎枝, 更新 airway_topo.Bi(减少假分支)

    airway_topo.grade()    # 依据父子关系给分支分级(气管=0级, 逐级递增)
    airway_topo.regrade()  # 对异常层级做校正重分级
    # rb23/rb12 是重合并开关标志; 若被置 1, 说明检测到需要二次合并的情形, 再合并并重分级
    if airway_topo.rb23==1 or airway_topo.rb12==1 :
        airway_topo.remerge()
        airway_topo.regrade()

    end_time = time.time()
    centerline_time = end_time - start_time
    print('Centerline segment time %d seconds' % centerline_time)   # 中心线阶段耗时

    # resize: 把分支顶点从体素坐标乘以 spacing 换算成毫米坐标, 得到 Bi_resize 并存 .npy
    # 【潜在问题】此处直接向 save_dir 写 .npy, 但目录存在性检查/创建写在了它之后;
    #   且 os.mkdir 只能建单层, 若父目录缺失会抛 FileNotFoundError。
    #   目前靠 __main__ 里提前 Path(...).mkdir(parents=True) 兜底才不会出错。
    # Save airway_topo.Bi_resize a .npy file
    airway_topo.resize(px, py, pz,os.path.join(save_dir, case.split('.mha')[0] + '_parse.npy'))  #Perform resizing operation using input pixel size to obtain airway_topo.Bi_resize
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dir_r = os.path.join(save_dir, case.split('.mha')[0] + '.stl')
    airway_topo.recons(dir_r)          # 三维表面重建, 存 .stl 网格
    airway_topo.show_line1(save_dir, case)  #可视化中心线分段
    start_time = time.time()

    end_time=airway_topo.sub_model(start_time, save_dir, case)  #逐分支建模(树解析产物)
    tree_parse_time = end_time - start_time
    print('Airway tree parse time %d seconds' % tree_parse_time)
    print('Number of branches %d ' % (len(airway_topo.Bi)))     # 最终分支数量

    # 把两阶段耗时与分支数写入 _time.txt, 便于批量对比算法效率
    # Save time information to a text file
    time_filename = os.path.join(save_dir, case.split('.mha')[0] + '_time.txt')
    with open(time_filename, 'w') as f:
        f.write('Centerline segment time %d seconds\n' % centerline_time)
        f.write('Airway tree parse time %d seconds\n' % tree_parse_time)
        f.write('Number of branches %d\n' % (len(airway_topo.Bi)))

    return airway_topo   # 返回已解析完成的树对象, 供外部进一步取用 Bi/Bi_resize

def atm22_skel_parse(label, spacing, olddir, case):
    """
    ATM22 官方基线树解析: 表面重建 + 骨架化 + 迭代式 tree_refinement。

    注意形参名是 label, 但 __main__ 里实际传进来的是 pred(预测掩码),
    即本函数同样可用于“评估预测的树结构”, 而非仅用于金标准标注。

    :param olddir: 产物输出目录(命名沿用官方脚本, 实为 save 目录)
    :return tree_parsing: 逐体素标注“属于第几号分支”的标签图
    """
    # 预生成 400 个随机十六进制颜色, 给每个分支一种独特色, 便于可视化区分
    # 【小瑕疵】内外两层循环变量都叫 i, 外层 i 被内层覆盖; 因外层只用于计数
    #   400 次、循环体内不读取 i, 故不影响结果, 但属易误读写法
    colors = []
    color_list = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
        'E', 'F'
    ]
    for i in range(400):
        color = ''
        for i in range(6):
            color_number = color_list[random.randint(0, 15)]   # 随机取一个十六进制位
            color += color_number
        color = '#' + color          # 拼成 #RRGGBB
        colors.append(color)

    start_time = time.time()

    # ---- 表面重建(recons): 把体素掩码变成三维网格 ----
    label = large_connected_domain(label)   # 先只保留最大连通域, 去掉孤立假阳碎块
    iso = 0.95                              # 等值面阈值: 取接近实体的位置提取表面
    verts, faces, _, _ = marching_cubes(label, iso)   # 移动立方体法提取表面顶点/三角面
    # 顶点坐标从体素索引换算成毫米(逐轴乘 spacing), 使网格尺寸符合真实解剖
    # verts 列是 numpy 轴序, 逐轴需配 spacing 反序 (axis_i ↔ spacing[2-i])
    verts[:, 0] = verts[:, 0] * spacing[2]
    verts[:, 1] = verts[:, 1] * spacing[1]
    verts[:, 2] = verts[:, 2] * spacing[0]
    # pyvista 要求每个面前面带“顶点数”列(三角面恒为 3), 这里在 faces 前补一列 3
    faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
    mesh_airway = pv.PolyData(verts, faces)
    mesh_airway_smooth = mesh_airway.smooth(relaxation_factor=0.2)  # 拉普拉斯平滑, 让表面更光滑
    pl = pv.Plotter()
    pl.add_mesh(mesh_airway_smooth, color='#E96C6F', style='surface')
    if not os.path.exists(olddir):
        os.mkdir(olddir)
    dir_r = os.path.join(olddir, case.split('.mha')[0] + '.stl')
    mesh_airway_smooth.save(dir_r)

    # ---- 中心线提取与解析 ----
    #Centerline segment/ tuning
    skeleton = skeletonize_3d(label)                 # 3D 细化: 实体气道 -> 单像素宽中心线
    skeleton_parse, cd, num = skeleton_parsing(skeleton)  # 把骨架切成 num 段分支
    #   skeleton_parse: 骨架的分段标签图; cd: 每条中心线点对应的分支编号; num: 分支总数
    end_time = time.time()
    centerline_time = end_time - start_time
    print('Centerline segment time %d seconds' % centerline_time)
    mesh_airway_smooth = pv.read(dir_r)   # 读回刚存的 .stl, 作为半透明底图
    # off_screen=True: 无窗口环境下离屏渲染(服务器/远程跑批必需)
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh_airway_smooth,
                color='white',
                style='surface',
                opacity=0.4)              # 气道表面用 40% 透明白底, 便于叠加看中心线
    for i in range(1, num + 1):           # 逐个分支画一条彩色中心线
        bi = np.array(np.where(cd == i))          # 取属于第 i 号分支的所有体素坐标 (3,N)
        bi = bi[:, bi[2].argsort()]               # 按 z 坐标排序, 保证折线沿走向连续不断裂
        bi = bi.T                                 # 转成 (N,3) 的点序列
        bi = np.array(bi)
        if bi.shape[0] % 2 != 0:
            bi = bi[:-1, :]               # add_lines 要求点是“成对”的(每 2 点一段线),
                                          # 故去掉末尾落单的一个点
        bi = bi.astype(np.float32)
        # bi 列来自 np.where, 是 numpy 轴序; 必须与上面 verts 用同一套反序配对, 否则中心线脱离底图
        bi[:, 0] *= spacing[2]            # 体素坐标 -> 毫米, 与底图网格对齐
        bi[:, 1] *= spacing[1]
        bi[:, 2] *= spacing[0]

        pl.add_lines(bi, color=colors[i - 1], width=5)   # 每条分支一种颜色, 线宽 5
    pl.background_color = "white"
    pl.view_yz()                          # 冠状面: YXZ 数据沿 axis0(物理AP)看, HF 竖直; ZYX 数据需换 'xz'
    screenshot_filename = os.path.join(olddir,
                                       case.split('.mha')[0] + '.png')
    pl.screenshot(screenshot_filename)    # 截图保存中心线可视化
    pl.close()

    # ---- 树精修(Tree parse): 反复重建拓扑直到无需修正 ----
    #Tree parse/ tuning
    start_time = time.time()
    tree_parsing = tree_parsing_func(skeleton_parse, label, cd)  # 由骨架分段+实体推出分支邻接关系
    trachea = loc_trachea(tree_parsing, num)                     # 定位气管(树根)所在的分支编号
    ad_matric = adjacent_map(tree_parsing, num)                  # 构建分支邻接矩阵
    parent_map, children_map, generation = parent_children_map(
        ad_matric, trachea, num)     # 以气管为根做 BFS, 得到父/子映射与代数(层级)
    # 只要判定还需要修正(如层级错乱、父子关系异常), 就合并/拆分分支后重建拓扑, 直到收敛
    while whether_refinement(parent_map, children_map, tree_parsing, num,
                             trachea) is True:
        tree_parsing, num = tree_refinement(parent_map, children_map,
                                            tree_parsing, num, trachea)
        # 分支数 num 可能因合并而变化, 故根/邻接/父子关系全部重新计算
        trachea = loc_trachea(tree_parsing, num)
        ad_matric = adjacent_map(tree_parsing, num)
        parent_map, children_map, generation = parent_children_map(
            ad_matric, trachea, num)

    end_time = time.time()
    tree_parse_time = end_time - start_time
    print('Airway tree parse time %d seconds' % tree_parse_time)
    print('Number of branches %d ' % (num))
    # ---- 逐分支上色, 输出 60 帧旋转 GIF 动画 ----
    pl = pv.Plotter(off_screen=True)
    pl.open_gif(os.path.join(olddir, case.split('.mha')[0] + '.gif'))
    for k in range(1, num + 1):
        iso = 0.95
        # 对“第 k 号分支的掩码”单独提表面 -> 每个分支成为独立可上色的网格
        verts, faces, _, _ = marching_cubes(tree_parsing == k, iso)
        verts[:, 0] = verts[:, 0] * spacing[2]   # 同样换算成毫米(numpy 轴序配 spacing 反序)
        verts[:, 1] = verts[:, 1] * spacing[1]
        verts[:, 2] = verts[:, 2] * spacing[0]
        faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
        mesh_seg = pv.PolyData(verts, faces)
        mesh_seg = mesh_seg.smooth(relaxation_factor=0.15)
        pl.add_mesh(mesh_seg, color=colors[k - 1], style='surface')  #
    n_frames = 60
    for i in range(n_frames):
        pl.camera_position = 'yz'
        pl.camera.azimuth = 360 * (i / n_frames)   # 方位角均匀转一圈 -> 旋转展示效果
        pl.render()
        pl.write_frame()
    pl.close()


    # ---- 静态模型图(单帧, 与 GIF 同一套逐分支上色网格) ----
    pl = pv.Plotter(off_screen=True)
    for k in range(1, num + 1):
        iso = 0.95
        verts, faces, _, _ = marching_cubes(tree_parsing == k, iso)
        verts[:, 0] = verts[:, 0] * spacing[2]   # numpy 轴序配 spacing 反序
        verts[:, 1] = verts[:, 1] * spacing[1]
        verts[:, 2] = verts[:, 2] * spacing[0]
        faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
        mesh_seg = pv.PolyData(verts, faces)
        mesh_seg = mesh_seg.smooth(relaxation_factor=0.15)
        pl.add_mesh(mesh_seg, color=colors[k - 1], style='surface')  #

    pl.camera_position = 'yz'   # 冠状面(YXZ 数据: 沿 axis0=AP 看); ZYX 数据会俯视
    # show(screenshot=...) 渲染一帧并存成 _model.png(注意与上面的 .png 中心线图不同名)
    pl.show(screenshot=os.path.join(olddir, case.split('.mha')[0] + '_model.png'))


    # 保存两阶段耗时与分支数, 供批量效率/规模对比
    # Save time information to a text file
    time_filename = os.path.join(olddir, case.split('.mha')[0] + '_time.txt')
    with open(time_filename, 'w') as f:
        f.write('Centerline segment time %d seconds\n' % centerline_time)
        f.write('Airway tree parse time %d seconds\n' % tree_parse_time)
        f.write('Number of branches %d\n' % num)


    return tree_parsing   # 返回“体素->分支编号”的标签图, 是后续指标计算(BD 等)的输入


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process airway segmentation and tree parsing.")

    parser.add_argument(
        '--pred_mask_path',
        type=str,
        default='./demo_mask/',
        help="Path to the directory containing predicted mask files.")
    parser.add_argument('--save_path',
                        type=str,
                        default=None,
                        help="Directory where the Ours output will be saved.")
    parser.add_argument('--save_ATM22_path',
                        type=str,
                        default=None,
                        help="Directory where the ATM22 output will be saved.")
    parser.add_argument(
        '--merge_t',
        type=int,
        default=5,
        help="Threshold for merging branches during airway skeleton parsing.")

    args = parser.parse_args()

    pred_mask_path = args.pred_mask_path
    save_path = args.save_path
    save_ATM22_path = args.save_ATM22_path
    merge_t = args.merge_t

    # 也可写死路径直接跑(调试用), 默认走上面的命令行参数
    # pred_mask_path = './demo_mask/'
    # save_path ='./demo_output_Ours/'
    # save_ATM22_path = './demo_output_ATM22/'
    # merge_t = 5

    flist = os.listdir(pred_mask_path)   # 列出预测掩码目录下所有病例
    flist.sort()                          # 按名排序, 保证处理顺序稳定可复现

    for case in flist:
        # 读入预测掩码: 返回体素数组 + 几何信息(origin/spacing/direction)
        # 这里只用 pred 和 spacing; spacing 仅决定 resize/建模时体素->毫米的换算,
    # merge_t 判定用的是骨架点数, 与 spacing 无关
        pred, origin, spacing, direction = load_itk_image(os.path.join(pred_mask_path, case))

        # 产出目录用 mkdir(parents=True, exist_ok=True) 递归创建, 已存在也不报错
        # (这也正好兜底了 ours_skel_parse 内部 os.mkdir 早于建目录的潜在问题)
        if save_path is not None:
            if not Path(save_path).exists():
                Path(save_path).mkdir(parents=True, exist_ok=True)
            airway_topo = ours_skel_parse(pred, spacing, merge_t, save_path,case)

        # 两套算法二选一/都跑, 由是否传对应 --save 参数决定, 便于横向对比
        if save_ATM22_path is not None:
            if not Path(save_ATM22_path).exists():
                Path(save_ATM22_path).mkdir(parents=True, exist_ok=True)
            tree_parsing = atm22_skel_parse(pred, spacing, save_ATM22_path,case)
