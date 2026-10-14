import os
import numpy as np
import torch
import pyvista as pv
from skimage.morphology import skeletonize_3d
from skimage.measure import marching_cubes_lewiner
from stl import mesh
from SE_UNet import SE_UNet
from preprocessing import preprocess_CT
from util import *
import time

def double_threshold_iteration(pred, h_thresh, l_thresh):
    """
    双阈值迭代(DTI)后处理：把被单一阈值“腰斩”的细小气道末梢重新接回主干。
    与 train.py/test.py 中的同名函数逻辑一致(此处用 float64, 那两处用 float32)。

    思路：高阈值定“可靠种子”，低阈值定“候选区”，但候选只有与种子连通才被吸收。
    :param h_thresh: 高阈值(种子), 本文件用 0.5
    :param l_thresh: 低阈值(生长候选), 本文件用 0.4
    :return: 二值化预测 {0,1}
    """
    # 26 邻域偏移表(3x3x3 去中心), 用于判断候选体素是否与已生长区域相邻
    neigb = np.array(
    [[-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0], [0, 1, 0], [1, -1, 0], [1, 0, 0], [1, 1, 0], [-1, -1, -1],
    [-1, 0, -1], [-1, 1, -1], [0, -1, -1], [0, 0, -1], [0, 1, -1], [1, -1, -1], [1, 0, -1], [1, 1, -1],
    [-1, -1, 1], [-1, 0, 1], [-1, 1, 1], [0, -1, 1], [0, 0, 1], [0, 1, 1], [1, -1, 1], [1, 0, 1],[1, 1, 1]])
    h, w,z = pred.shape
    pred = np.array(pred*255, dtype=np.float64)   # 概率缩放到 0~255, 仅为比较直观
    bin = np.where(pred >= h_thresh*255, 255, 0).astype(np.float64)  # 种子集(高置信)
    gbin = bin.copy()      # 当前生长结果, 初值为种子
    gbin_pre = gbin-1      # 上一轮快照; -1 造“必然不等”初值, 保证 while 至少进入一次
    # 【已知缺陷】gbin_pre = gbin 是引用赋值而非拷贝, 两者指向同一对象,
    #   导致 gbin_pre.all() != gbin.all() 恒为 False, 区域生长实际只扩展【一层】就退出。
    #   正确写法应为 gbin_pre = gbin.copy()。
    while(gbin_pre.all() != gbin.all()):
        gbin_pre = gbin
        for i in range(h):              # 纯 Python 三重循环遍历整卷, 极慢, 应向量化
            for j in range(w):
                for k in range(z):
                    # 仅处理候选体素: 未入选 且 概率落在 [低阈值, 高阈值) 之间
                    if gbin[i][j][k] == 0 and pred[i][j][k] < h_thresh*255 and pred[i][j][k] >= l_thresh*255:
                        for n in range(0, 26):   # 逐个查 26 邻域
                            inn = i + neigb[n, 0]
                            jnn = j + neigb[n, 1]
                            knn = k + neigb[n, 2]
                            # max(min(...)) 夹持索引避免越界(等价边缘重复填充)
                            if gbin[max(min(inn,h-1),0)][max(min(jnn,w-1),0)][max(min(knn,z-1),0)]:
                                gbin[i][j][k] = 255   # 与已入选区域相邻 -> 吸收
                                break                 # 有一个邻居入选即可

    return gbin/255   # 还原为 {0,1}

def two_channel(data):
    """
    把单通道 CT(HU 值)转成两路不同窗宽窗位的归一化输入, 对应 SE_UNet 的 2 输入通道。
    两路窗宽各有侧重, 让网络同时看到“软组织对比”和“含气腔对比”:
      data2(纵隔窗): 窗位宽 [-1000,500] -> 突出血管/软组织等密度差异
      data (肺窗)  : 范围 [-1024,1024]  -> 突出肺内气体与气道腔的低 HU 结构
    各自截断(clear)后线性缩放到 [0,1]。
    """
    data = data.astype(float)
    data2 = data.copy()
    # ---- 第一路: 纵隔窗 [-1000, 500], 跨度 1500 ----
    data2[data2 > 500] = 500        # 上截断
    data2[data2 < -1000] = -1000    # 下截断
    data2 = (data2 + 1000) / 1500   # 平移到 [0,1]
    # ---- 第二路: 肺窗 [-1024, 1024], 跨度 2048 ----
    data[data > 1024] = 1024
    data[data < -1024] = -1024
    data = (data + 1024) / 2048

    return data, data2

def network_prediction(model_path,patient,data_path, save_path,flip=False,gpu='0'):
    """
    对单个病例做完整推理：读入预处理后的 CT -> 滑窗预测 -> 后处理 -> 存掩码 + 导出 STL。

    :param model_path: 权重文件, 如 saved_model/stage_three/SE_UNet_43.pth
    :param patient   : 病例名(含 'data' 字样), 内部用 split('data')[0] 还原出真实前缀
    :param data_path : 预处理结果目录(如 ./processed_cases/)
    :param save_path : 输出目录(如 ./predicted_airways/)
    :param flip      : True 时对结果做一次翻转再导出 STL(注意: 掩码在翻转前已保存)
    :return: (result 二值掩码, spacing)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu   # 必须在 .cuda() 之前设置才生效

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)   # 遗留未使用(本函数无 DataLoader)

    start_time = time.time()
    print ('starting segmenting lung CT')

    model = SE_UNet(in_channel=2, n_classes=1)   # 2 通道(双窗宽)输入, 1 通道前景概率输出
    weights_dict = torch.load(model_path)
    model.load_state_dict(weights_dict, strict=False)   # strict=False 容忍少量键不匹配
    model = torch.nn.DataParallel(model).cuda()          # 包一层以统一 .module 存取习惯
    model.eval()   # 推理必须 eval(): 否则 DropLayer 会按 self.training 随机丢弃 30% 通道,
                   # 使每个 patch 拿到不同随机掩码, 滑窗平均结果不可复现
    cube_size = 128   # 每个滑窗块的边长(体素)
    step = 64         # 步长 = 边长一半 -> 相邻块重叠 50%, 用于消除拼接缝

    # 读入预处理后的裁剪体; 同时拿到几何信息以便原样写回
    img, origin, spacing, direction = load_itk_image(data_path+patient.split('data')[0]+'data_cut.mha')
    img=img-1024    # 整体下移 1024: 把 CT 的 0~3071 范围平移回以空气为负的 HU 惯例,
                    # 使下面 two_channel 的窗宽阈值(-1024/500 等)语义成立
    
    # imput ct img
    # ---- 构造网络输入张量, 最终形状为 (N,C,D,H,W) = (1,2,Z,Y,X) ----
    img=np.expand_dims(img,axis=0)      # (Z,Y,X) -> (1,Z,Y,X), 补出 batch 维
    img, img2 = two_channel(img)        # 两路窗宽, 各自仍是 (1,Z,Y,X)
    img = np.array([img, img2])         # -> (2,1,Z,Y,X): 第0维是“通道来源”, 第1维是 batch
    x=torch.from_numpy(img.astype(np.float32))
    x = x.transpose(0, 1)               # -> (1,2,Z,Y,X): 把 batch 提到第0维、通道放第1维,torch这个函数是交换轴
                                        #   这才是 PyTorch 的 NCDHW 布局(不是改数据, 只是换维序)
    x = x.cuda()
    # 两个累加器: pred 存各体素被预测的概率之和, pred_num 存被覆盖的次数
    pred = np.zeros((1,1,x.shape[2],x.shape[3],x.shape[4]))
    pred_num = np.zeros((1,1,x.shape[2],x.shape[3],x.shape[4]))
    # 每个轴向需要的块数 = 向上取整((长度-块长)/步长 + 1);
    # 用三元式手写是因为整除时不该多切一块, 非整除时要多切一块才能覆盖末端
    xnum = (x.shape[2]-cube_size)//step + 1 if (x.shape[2]-cube_size)%step==0 else (x.shape[2]-cube_size)//step + 2
    ynum = (x.shape[3]-cube_size)//step + 1 if (x.shape[3]-cube_size)%step==0 else (x.shape[3]-cube_size)//step + 2
    znum = (x.shape[4]-cube_size)//step + 1 if (x.shape[4]-cube_size)%step==0 else (x.shape[4]-cube_size)//step + 2
    for xx in range(xnum):        # 沿 Z(第一轴)滑窗
        xl = step*xx
        xr = step*xx + cube_size
        if xr > x.shape[2]:       # 末端越界时: 把窗口右端贴到体积末尾, 左端回退一个块长
            xr = x.shape[2]       # (这样最后一块与倒数第二块重叠更多, 但保证块尺寸恒为 128)
            xl = x.shape[2]-cube_size
        for yy in range(ynum):    # 沿 Y
            yl = step*yy
            yr = step*yy + cube_size
            if yr > x.shape[3]:
                yr = x.shape[3]
                yl = x.shape[3] - cube_size
            for zz in range(znum):   # 沿 X
                zl = step*zz
                zr = step*zz + cube_size
                if zr > x.shape[4]:
                    zr = x.shape[4]
                    zl = x.shape[4] - cube_size

                x_input = x[:,:,xl:xr,yl:yr,zl:zr]   # (1,2,128,128,128)
                p0, p = model(x_input)               # 深监督双头: 只取解码端 p, p0 忽略
                p = torch.sigmoid(p)                 # logits -> 概率(阈值判断前必须做)
                p = p.cpu().detach().numpy()         # 转回 CPU 累加, 省显存
                pred[:,:,xl:xr,yl:yr,zl:zr] += p        # 把该块概率加回它在整卷中的位置
                pred_num[:,:,xl:xr,yl:yr,zl:zr] += 1     # 记录覆盖次数(重叠区多为 4 次)

    # ---- 后处理链: 融合 -> 二值化 -> 边缘裁剪 -> 取最大连通域 ----
    pred = pred/pred_num          # 重叠区求平均 -> 每个体素的融合概率
    pred = np.squeeze(pred)       # 去掉补出来的 (1,1) 前导维 -> (Z,Y,X)
    pred=double_threshold_iteration(pred,h_thresh=0.5,l_thresh=0.4)   # DTI 受约束生长
    # 把最外 15% 直接清零: 预处理已按气道范围裁剪, 边缘基本不可能是气道,
    # 而滑窗在边缘处缺少上下文、假阳最多, 故用先验强行切掉
    pred[0:int(0.15 * pred.shape[0]), :, :] = 0
    pred[int(0.85 * pred.shape[0]):, :, :] = 0
    pred[:, 0:int(0.15 * pred.shape[1]), :] = 0
    pred[:, int(0.85 * pred.shape[1]):, :] = 0
    pred_img=maximum_3d(pred)     # 只保留最大连通域(气道是连通的树, 碎块必为假阳)

    result=pred_img               # 引用赋值(非拷贝), 下面对 result 的原地修改会同时作用于 pred_img
    # 用原 CT 的 origin/spacing/direction 写回, 保证掩码与影像在同一坐标系下可叠加;
    # 若用单位仿射矩阵(如某些脚本里的 np.eye(4))会丢失几何信息, 下游按 spacing 换算毫米就会错
    save_itk(result.astype(np.byte),origin,spacing,direction, save_path+patient+'_pred_mask.mha')

    # 【隐患】翻转发生在掩码保存【之后】:
    #   存下的 _pred_mask.mha 是未翻转的, 而后面导出的 .stl 是翻转后的,
    #   两者坐标系不一致。若 flip=True 是为对齐原始 DICOM 方向,
    #   则 save_itk 应移到翻转之后(或翻转提前到 save_itk 之前)。
    if flip==True:
        for y in range(result.shape[1]):
            result[ :,y, :] = np.flipud(result[ :,y, :])   # 沿 Z 轴逐行上下翻转

    # ---- 导出 STL 三维模型(供 CAD/浏览软件查看) ----
    iso = 0.95
    verts, faces, _, _ = marching_cubes_lewiner(result, iso)   # 从二值掩码提取表面顶点与三角面
    skel = skeletonize_3d(result)                              # 骨架化, 只为求几何中心做平移基准
    B = np.array(np.where(skel != 0))                          # 骨架点坐标 (3,N)
    B = B[:, B[2].argsort()]                                   # 按 z 排序(此处仅为整理, 不影响求均值)
    B = B.T      #为啥转置
    B=B.astype(np.float32)
    ox=np.mean(B[:,0]); oy=np.mean(B[:,1]); oz=np.mean(B[:,2])  # 骨架质心
    # 把模型平移到以质心为原点(居中), 便于第三方软件查看时不用处理大坐标偏移
    verts[:, 0] = verts[:, 0]-ox
    verts[:, 1] = verts[:, 1]-oy
    verts[:, 2] = verts[:, 2]-oz
    # 体素间距换算成毫米后再 /10 整体缩小(纯显示尺度, 不改变形状)
    verts[:, 0] = verts[:, 0] * spacing[0]/10
    verts[:, 1] = verts[:, 1] * spacing[1]/10
    verts[:, 2] = verts[:, 2] * spacing[2]/10
    # numpy-stl 要求每个三角面存为 3x3 向量数组, 逐面逐顶点填入
    cube_a = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)) 
    for i, f in enumerate(faces):
        for j in range(3):
            cube_a.vectors[i][j] = verts[f[j], :]
    cube_a.save(save_path + "/" +patient+'.stl')
    # 再用 pyvista 读回并平滑一次后覆盖保存(先写后读再写, 属冗余但能美化表面)
    mesh_a = pv.read(save_path + "/" +patient+'.stl') 
    mesh_a_smooth = mesh_a.smooth(relaxation_factor=0.2)
    mesh_a_smooth.plot(background='w')   # 会弹出交互窗口; 服务器/无显示环境需注释掉此行
    mesh_a_smooth.save(save_path + "/" +patient+'.stl')

    end_time = time.time()
    print ('end segmenting lung CT, time %d seconds'%(end_time-start_time))

    return result,spacing

if __name__ == '__main__':
    # ---- 自动挑空闲 GPU: 轮询直到有足够显存空闲的卡才继续, 适合多人共用服务器 ----
    gpu_all = 1        # 机器总卡数
    gpu_need = 1       # 本任务需要几张卡
    lm_per_gpu = 10000 # 单卡最低空闲显存(MB)门槛
    while 1:
        free = np.zeros((gpu_all))
        for i in range(gpu_all):
            free[i] = get_gpu_mem_info(i)      # 查每张卡的空闲显存
        if len(np.where(free > lm_per_gpu)[0]) >= gpu_need:
            break                              # 空闲卡数够了才跳出轮询
    gpu = ','.join(
        [str(x) for x in list(np.where(free > lm_per_gpu)[0][0:gpu_need])])



    model_path = 'saved_model/stage_three/SE_UNet_43.pth'   # 用 stage3 第 43 epoch 权重
    DCM_Path = 'example_dcm'                                 # 原始 DICOM 目录
    flist = os.listdir(DCM_Path)
    flist.sort()
    for patient in flist:      # 逐病例: 先预处理再推理
        print('ct：', patient)
        data_path = os.path.join(DCM_Path , patient)
        save_path = './processed_cases/'
        # mode='prediction' 走推理用的预处理分支(不做训练相关的裁剪/增强), 输出 *_data_cut.mha
        preprocess_CT(data_path,save_path,format='nii.gz',mode='prediction',multides=False)

        data_path = './processed_cases/'
        save_path = './predicted_airways/'
        # 注意 save_path 目录需已存在, 本脚本未做 makedirs, 首次运行需手动创建
        pred, spacing = network_prediction(model_path,
                                            patient,
                                            data_path,
                                            save_path,
                                            flip=True)   # 翻转以对齐原始 DICOM 方向


    sfd = 9   # 无意义占位变量(遗留代码), 可删除
