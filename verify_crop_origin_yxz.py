# -*- coding: utf-8 -*-
"""
crop_origin_yxz 与 SimpleITK 官方 API 的对照验证脚本
====================================================
用途: 验证 preprocessing.crop_origin_yxz 计算的"裁剪后物理原点"
     与官方 TransformIndexToPhysicalPoint 逐点一致。
     修改 crop_origin_yxz / transpose_zyx_image / 轴序约定后都应重跑本脚本。

背景: 数组按 YXZ 序存放, spacing/direction 元数据按 ZXY 序,
     start 是数组(YXZ)索引 (iy0, ix0, iz0) -> 官方 ITK index 为反序 (iz0, ix0, iy0)。
     旧版 crop_origin_yxz 对 start 误用 spacing 的 [1,2,0] 循环置换,
     导致裁剪原点 Y/Z 分量互换; 正确置换是 [1,0,2]。

运行(需 SimpleITK 的环境, 本机为 sam):
    E:\\code_software\\miniconda\\envs\\sam\\python.exe verify_crop_origin_yxz.py
"""
import sys
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, r'd:\code_project\medical\SE-UNet-AirSeg-original-v2')
from preprocessing import transpose_zyx_image, crop_origin_yxz

# ===== 构造非对称 ZYX 体: spacing/origin/direction 三者都不同, 排除巧合 =====
img = sitk.Image(400, 500, 300, sitk.sitkUInt8)       # ITK size (x=400,y=500,z=300)
img.SetSpacing((0.7, 0.8, 1.5))                       # (px, py, pz) 三轴互不相等
img.SetOrigin((10.0, -20.0, 30.0))
ang = np.deg2rad(23)
rot = np.array([[np.cos(ang), -np.sin(ang), 0.],
                [np.sin(ang),  np.cos(ang), 0.],
                [0., 0., 1.]])                        # 带旋转, 排除方向=I 的特例巧合
img.SetDirection(tuple(rot.ravel()))

t = transpose_zyx_image(img)                          # 真实 PermuteAxes(2,0,1) -> YXZ
print('转置后 array.shape =', sitk.GetArrayFromImage(t).shape,
      ' spacing =', t.GetSpacing(), ' origin =', t.GetOrigin())

# ===== 示例 1: 单点详解(与修复时的验算数字对应) =====
iy0, ix0, iz0 = 100, 80, 20
p_official = np.array(t.TransformIndexToPhysicalPoint((iz0, ix0, iy0)))
p_truth = np.array(img.GetOrigin()) + rot @ np.array([0.7*ix0, 0.8*iy0, 1.5*iz0])
p_func = np.array(crop_origin_yxz(t.GetOrigin(), t.GetSpacing(),
                                  t.GetDirection(), (iy0, ix0, iz0)))
np.set_printoptions(precision=4, suppress=True)
print('\n单点 (iy0,ix0,iz0) =', (iy0, ix0, iz0))
print('官方 TransformIndexToPhysicalPoint :', p_official)
print('物理真值(原 ZYX 图直接算)          :', p_truth)
print('crop_origin_yxz(修复后)            :', p_func)
print('官方 == 真值 ?', np.allclose(p_official, p_truth, atol=1e-4))
print('函数 == 官方 ?', np.allclose(p_func, p_official, atol=1e-4))

# ===== 示例 2: 20 组随机裁剪起点逐一与官方比对 =====
rng = np.random.RandomState(1)
ok = True
for _ in range(20):
    # int() 转换: numpy int64 不能直接进 SimpleITK 接口
    iy, ix, iz = [int(v) for v in rng.randint(0, [500, 400, 300])]
    po = np.array(t.TransformIndexToPhysicalPoint((iz, ix, iy)))
    pf = np.array(crop_origin_yxz(t.GetOrigin(), t.GetSpacing(),
                                  t.GetDirection(), (iy, ix, iz)))
    if not np.allclose(po, pf, atol=1e-4):
        ok = False
        print('MISMATCH', (iy, ix, iz), po, pf)
print('\n20 组随机起点全部与官方一致?', ok)
