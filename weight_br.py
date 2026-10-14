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
    file_list = load_json_file( file_path,
                                folder='0',
                                mode=['train', 'val'])
    file_list.sort()
    for idx in range(len(file_list)):
        name = file_list[idx]
        img0, origin, spacing, direction = load_itk_image(data_root + '/data/' + name + 'data_cut' +
                             '.mha')

        img0=img0-1024

        label, origin, spacing, direction = load_itk_image(data_root + '/mask/' + name + 'mask_cut' +
                               '.mha')
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
    # 修复: 用 os.makedirs 递归创建多级输出目录(os.mkdir 遇到缺失的父目录会直接报错)
    os.makedirs(savepath2, exist_ok=True)
    os.makedirs(savepath3, exist_ok=True)

    file_list = load_json_file(file_path, folder='0', mode=['train', 'val'])
    file_list.sort()
    for i in range(len(file_list)):
        name = file_list[i]
        label, origin, spacing, direction = load_itk_image(data_root+'/mask/'+name+ 'mask_cut' + '.mha')
        # 统一用 load_itk_image 读取 save_itk 写出的 3D 标量体数据,
        # 显式接收 origin/spacing/direction; pred 由 mask 的同一套元数据写出,
        # 此处重复赋值与上一行取到的值相同
        pred, origin, spacing, direction = load_itk_image(os.path.join(savepath, name+'.mha'))
        fn = ((label.astype(np.float16) - pred)>0).astype(np.uint8) 
        skeleton = skeletonize_3d(label)
        fn_skel = fn*skeleton 

        ################################### w_hm
        edt, inds = ndimage.distance_transform_edt(1-skeleton, return_indices=True)
        hard_mining = fn_skel[inds[0,...], inds[1,...], inds[2,...]] * label 
        loc = (hard_mining>0).astype(np.uint8)
        f = loc * edt 
        f = f * (1. - skeleton)
        maxf = np.amax(f) 
        if np.max(maxf)==0:
            w_br=np.zeros(label.shape,dtype=np.float16)
            br_skel=np.zeros(label.shape)
            print(np.min(w_br),np.max(w_br))
            np.save(os.path.join(savepath2, name+'.npy'), w_br)
            np.save(os.path.join(savepath3, name+'.npy'), br_skel)
            print(name)
            continue
        D = -((1./(maxf)) * f) + 1
        D = D * loc

        w_hm = (hard_mining**2)*(D**2)
        w_hm = w_hm.astype(np.float16)

        ################################### w_br
        cd=cc3d.connected_components(fn_skel, connectivity=26)
        br_skel=np.zeros(cd.shape)
        for i in range(1,np.max(cd)+1):
            t=cd==i
            t=t.astype(np.int8)
            neighbor_filter = ndimage.generate_binary_structure(3, 3)
            skeleton_filtered = ndimage.convolve(skeleton, neighbor_filter) * t
            if np.sum(skeleton_filtered==2):
                continue
            # print(i)
            br_skel+=t
        br_label = br_skel[inds[0,...], inds[1,...], inds[2,...]] * label 
        edt, inds = ndimage.distance_transform_edt(1-(binary_dilation(br_label)-br_label), return_indices=True)
        w_br=br_label*edt
        w_br[w_br>=2]=2

        w_br = w_br.astype(np.float16)
        lamda=0.7
        w_br=(w_br+w_hm)*lamda+1-lamda
        w_br=w_br*hard_mining

        print(np.min(w_br),np.max(w_br))
        np.save(os.path.join(savepath2, name+'.npy'), w_br)
        np.save(os.path.join(savepath3, name+'.npy'),np.where(br_skel==1))
        print(name)



def valid_recall(log_path):

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
    SCORE=[]
    for l in m_lines:
        TD.append(float(re.findall(r'TD: (.*?) ', l)[0]))
        BD.append(float(re.findall(r'BD: (.*?) ', l)[0]))
        DSC.append(float(re.findall(r'DSC: (.*?) ', l)[0]))
        Pre.append(float(re.findall(r'Pre: (.*?) ', l)[0]))
        Sen.append(float(re.findall(r'Sen: (.*?) ', l)[0]))
    for l in range(len(TD)):
        SCORE.append((TD[l]+BD[l])*0.15+(DSC[l]+Pre[l])*0.2+Sen[l]*0.3)

    print('第',SCORE.index(max(SCORE)),'个epoch验证效果最佳，分数为',max(SCORE))
    return SCORE.index(max(SCORE))


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



    log_path = './LOG/log_stage_two.txt'
    epoch = valid_recall(log_path)
    whichepoch = './saved_model/stage_two/SE_UNet_{}.pth'.format(epoch)
    savepath='./data/pred_2'
    file_root='./data'
    data_root='/mnt/yby/AFTER_PREPROCESS'
    save_s2_pred(data_root,whichepoch, savepath, file_root, gpu)


    savepath2='./data/BR_weight'
    savepath3='./data/br_skel'
    save_weight_break(data_root,savepath, savepath2, savepath3, file_root)
