import os
import numpy as np
import time
import SimpleITK as sitk
import warnings
from glob import glob
import re
import pydicom
import pylab
from util import *

def crop_origin_yxz(origin, spacing_yxz, direction_yxz, start_yxz):
    """将 YXZ 数组裁剪起点换算为物理 XYZ 体素数，并计算裁剪后的物理原点。

    注意各输入的内部顺序不同，还原置换也不同：
      spacing_yxz = (pz, px, py)，ZXY 序 -> 还原 XYZ 用 [1,2,0]
      direction_yxz 列 = (Z, X, Y)，ZXY 序 -> 还原用 [:,[1,2,0]]
      start_yxz   = (iy, ix, iz)，数组 YXZ 序 -> 还原 XYZ 体素数须用 [1,0,2]
                    (旧代码误用 [1,2,0]，导致裁剪原点 Y/Z 分量互换)
    origin 是物理点，分量恒为世界 (x,y,z)，不随轴序重排，直接使用。
    已对照 SimpleITK 官方 TransformIndexToPhysicalPoint 实测等价。
    """
    direction_xyz = np.array(direction_yxz).reshape(3, 3)[:,[1,2,0]]
    spacing_xyz = np.array([spacing_yxz[1], spacing_yxz[2], spacing_yxz[0]])
    start_xyz = np.array([start_yxz[1], start_yxz[0], start_yxz[2]])
    origin_xyz = np.array(origin)
    cropped_origin = origin_xyz + direction_xyz @ (spacing_xyz * start_xyz)
    return tuple(cropped_origin.tolist())
    
def transpose_zyx_image(image: sitk.Image) -> sitk.Image:
    """
    ZYX 数组 -> YXZ 数组：与 convert_zyx_to_yxz.py 中同名函数同一形式，
    直接用 sitk.PermuteAxes 让 SimpleITK 内部同步处理三件事——
      数组:      (sz, sy, sx) -> (sy, sx, sz)   (等价 numpy transpose(1,2,0))
      direction: 列随轴循环平移 (old[:, [2, 0, 1]], 即 ZXY)
      spacing:   (px, py, pz) -> (pz, px, py)
      origin:    不变
    转完后 GetArrayFromImage 得到的就是本项目约定的 YXZ 数组序，
    且 GetSpacing/GetDirection 已是"按 YXZ 存放"的配套值，
    与下游 crop_origin_yxz / save_itk 的 YXZ 约定严格对齐。
    """
    # 防御: 只支持标量 3D 体数据(CT/mask)，多分量或 2D 直接报错
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Only scalar 3D images are supported.")
    # (2, 0, 1): 新数组的第 0 轴取旧第 2 轴... 与 convert 脚本验证过的等价写法
    return sitk.PermuteAxes(image, (2, 0, 1))


def load_itk_image(filename):
    """纯读取: 返回数组(原始 ZYX 轴序)与 origin/spacing/direction，不做任何转置。"""
    itkimage = sitk.ReadImage(filename)
    image = sitk.GetArrayFromImage(itkimage)
    origin = itkimage.GetOrigin()
    spacing = itkimage.GetSpacing()
    direction = itkimage.GetDirection()
    return image, origin, spacing, direction

def save_itk(image, origin, spacing, direction, filename):

    itkimage = sitk.GetImageFromArray(image)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, filename)

def savenpy(data_path, prep_folder, format='nii.gz', mode='prepro', multides=True):

    resolution = np.array([1, 1, 1])
    # name = data_path.split('/')[-1].split('.nii')[0]
    # name = data_path.split('/')[-1].split('data.nii')[0]
    name = os.path.basename(data_path).split('data.nii')[0]# 取病例名, 如 'CASE002'
    assert (os.path.exists(data_path) is True)
    if format == 'nii.gz':
        # 先正常读取原始 ZYX 数据，再立即转成 YXZ 数组序，之后全部按 YXZ 处理。
        # PermuteAxes 会同步平移 spacing/direction，使三者与 YXZ 数组配套，
        # 与下游 crop_origin_yxz / save_itk 的 YXZ 约定严格对齐。
        itkimage = sitk.ReadImage(data_path)
        itkimage = transpose_zyx_image(itkimage)          # ZYX -> YXZ
        case_pixels = sitk.GetArrayFromImage(itkimage)    # 形状 (sy, sx, sz)
        origin = itkimage.GetOrigin()
        spacing = itkimage.GetSpacing()                   # 已随轴平移: (pz, px, py)
        direction = itkimage.GetDirection()               # 列已循环平移: ZXY
        # if spacing[0] != spacing[1] and spacing[1] == spacing[2]:
        #     px, py, pz = spacing[1], spacing[2], spacing[0]
        #     spacing = (px, py, pz)
        #     ox, oy, oz = origin[1], origin[2], origin[0]
        #     origin = (ox, oy, oz)
        # a = case_pixels.shape[0]
        # b = case_pixels.shape[1]
        # c = case_pixels.shape[2]
        # if b == c:
        #     case_pixels = case_pixels.transpose(1, 2, 0)
        # elif a == c:
        #     case_pixels = case_pixels.transpose(0, 2, 1)

    case_pixels=case_pixels+1024
    cmin = np.min(case_pixels)
    cmax = np.max(case_pixels)
    # hist = pylab.hist(case_pixels.flatten(), 300)
    hist =np.histogram(case_pixels.flatten(), 300)

    th = -800
    if cmin <= th:
        hhh1 = hist[1][np.where(hist[1] >= th)[0][0]:]
        hhh0 = hist[0][np.where(hist[1] >= th)[0][0]:]
        maxloc = np.where(hhh0 == np.max(hhh0))
        # print(maxLoc)
        firstPeak = hhh1[maxloc[0][0]] 
        measureDists = np.zeros([300], np.float32)
        for k in range(hhh0.shape[0]):
            measureDists[k] = pow(hhh1[k + 1] - firstPeak,
                                  2) * hhh0[k]  
        maxloc2 = np.where(measureDists == np.max(measureDists))
        secondPeak = hhh1[maxloc2[0][0]] 
        aaa = firstPeak
        if secondPeak < firstPeak:
            aaa = secondPeak
        case_pixels[np.where(case_pixels <= th)] = aaa
        cmax = np.max(case_pixels)

    if mode == 'prepro':
        T = th_2t(case_pixels)
        L = get_l(case_pixels, T)
        L1 = maximum_3d(L)
        L2 = maximum_3d(L ^ L1)
        L = L1 + L2

        Mask = L
        xx, yy, zz = np.where(Mask)
        box = np.array([[np.min(xx), np.max(xx)], [np.min(yy),
                                                np.max(yy)],
                        [np.min(zz), np.max(zz)]])
        margin = 5
        box = np.vstack([
            np.max([[0, 0, 0], box[:, 0] - margin], 0),
            np.min([np.array(Mask.shape), box[:, 1] + margin], axis=0).T
        ]).T

        # save the lung mask
        data_savepath = os.path.join(prep_folder, name + '_lung_mask.mha')
        Mask_crop = Mask[box[0, 0]:box[0, 1], box[1, 0]:box[1, 1],
                        box[2, 0]:box[2, 1]]
        crop_origin = crop_origin_yxz(origin, spacing, direction, box[:, 0])
        save_itk(Mask_crop.astype(dtype='uint8'), crop_origin, spacing, direction, data_savepath)

        sliceim_hu = case_pixels
        shapeorg = sliceim_hu.shape
        box_shape = np.array([[0, shapeorg[0]], [0, shapeorg[1]], [0,
                                                                shapeorg[2]]])
        sliceim2_hu = sliceim_hu[box[0, 0]:box[0, 1], box[1, 0]:box[1, 1],
                                box[2, 0]:box[2, 1]]
        box = np.concatenate([box, box_shape], axis=0)
        np.save(os.path.join(prep_folder, name + '_box.npy'), box)

    # save processed image
    data_savepath = os.path.join(prep_folder, name + 'data_cut.mha')
    if mode == 'prediction':
        # 不裁剪，直接保存原始图像
        save_itk(case_pixels, origin, spacing, direction, data_savepath)
    else:
        #剪图像
        save_itk(sliceim2_hu, crop_origin, spacing, direction, data_savepath)
    return

def cutmask(data_path, prep_folder):
    
    # name = data_path.split('CASE')[-1].split('mask')[0]
    name = os.path.basename(data_path).split('CASE')[-1].split('mask')[0]
    assert (os.path.exists(data_path) is True)
    # 与 savenpy 一致: 先读原始 ZYX，再转 YXZ，box 切片与保存均按 YXZ 口径
    itkimage = sitk.ReadImage(data_path)
    itkimage = transpose_zyx_image(itkimage)          # ZYX -> YXZ
    mask_p = sitk.GetArrayFromImage(itkimage)         # 形状 (sy, sx, sz)
    origin = itkimage.GetOrigin()
    spacing = itkimage.GetSpacing()
    direction = itkimage.GetDirection()
    # save box (image original shape and cropped window region)
    box = np.load(os.path.join(prep_folder[:-4]+'data','CASE'+name+'_box.npy'),allow_pickle=True)
    # mask_p = large_connected_domain(mask_p)
    mask_p = large_connected_domain26(mask_p)
    sliceim2_hu = mask_p[box[0, 0]:box[0, 1],
         box[1,0]:box[1,1],
         box[2,0]:box[2,1]]
    crop_origin = crop_origin_yxz(origin, spacing, direction, box[0:3, 0])
    data_savepath = os.path.join(prep_folder, 'CASE'+name+'mask_cut.mha')
    save_itk(sliceim2_hu.astype(dtype='uint8'), crop_origin, spacing, direction, data_savepath)

    return

def preprocess_CT(inputpath=None, savepath=None, format='nii.gz',mode='prepro',multides=True):
    """
	Preprocess the CT images in the input path to extract lung field
	Save the processed images in the save path
	:param inputpath: input data path
	:param savepath: output save path
	:return: save directory path
	"""
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    if format=='nii.gz':
        if mode=='prepro':
            filelist = glob(os.path.join(inputpath, '*.nii*'))  # default nifty format
            filelist.sort()
            print (inputpath, filelist)
            for curfilepath in filelist:
                start_time = time.time()
                print ('starting preprocessing lung CT', curfilepath)
                savenpy(data_path=curfilepath, prep_folder=savepath,format=format,mode=mode)
                end_time = time.time()
                print ('end preprocessing lung CT', curfilepath, ' time %d seconds'%(end_time-start_time))
        if mode=='prediction':
            start_time = time.time()
            print ('starting preprocessing lung CT', inputpath)
            savenpy(data_path=inputpath, prep_folder=savepath,format=format,mode=mode)
            end_time = time.time()
            print ('end preprocessing lung CT', ' time %d seconds'%(end_time-start_time))


    return savepath

def preprocess_mask(inputpath=None, savepath=None):
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    filelist = glob(os.path.join(inputpath, '*.nii*'))  # default nifty format
    print (inputpath, filelist)
    for curfilepath in filelist:
        start_time = time.time()
        print ('starting preprocessing lung mask', curfilepath)
        cutmask(data_path=curfilepath, prep_folder=savepath)
        end_time = time.time()
        print ('end preprocessing lung mask', ' time %d seconds'%(end_time-start_time))

    return savepath


if __name__ == '__main__':

    inputpath = r'F:\datasets\ATM22\TrainBatch1\before/data'
    savepath = r'F:\datasets\ATM22\TrainBatch1\after/data'
    inputpath2 = r'F:\datasets\ATM22\TrainBatch1\before/mask'
    savepath2 = r'F:\datasets\ATM22\TrainBatch1\after/mask'

    preprocess_CT(inputpath, savepath)
    preprocess_mask(inputpath2, savepath2)
