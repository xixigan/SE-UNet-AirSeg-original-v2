import skimage
print(skimage.__version__)
import SimpleITK as sitk
import numpy as np

def load_itk_image(filename):
    """
    读取 NIfTI (.nii.gz) 或 DICOM 图像。
    返回: numpy数组, 原点, 间距, 方向矩阵
    """
    # 1. 读取图像，得到一个 SimpleITK 的 Image 对象
    itk_image = sitk.ReadImage(filename)
    
    # 2. 将 Image 对象转换为 NumPy 数组
    # 【核心知识点 1】：返回的数组维度顺序永远是 (Z, Y, X)
    # 即：shape[0]是切片数(Z)，shape[1]是高度(Y)，shape[2]是宽度(X)
    numpy_image = sitk.GetArrayFromImage(itk_image)
    
    # 3. 获取物理空间参数
    # 【核心知识点 2】：SimpleITK 返回的物理参数顺序永远是 (X, Y, Z)！
    # 这与 NumPy 数组的 (Z, Y, X) 顺序是相反的！
    origin = itk_image.GetOrigin()       # 返回 tuple: (X, Y, Z)
    spacing = itk_image.GetSpacing()     # 返回 tuple: (X, Y, Z)
    direction = itk_image.GetDirection() # 返回 tuple: 9个元素，代表3x3方向矩阵
    
    return numpy_image, origin, spacing, direction

def save_itk_image(numpy_image, origin, spacing, direction, filename):
    """
    将 NumPy 数组保存为 NIfTI (.nii.gz) 图像。
    注意：传入的 origin, spacing, direction 必须是原始的 (X, Y, Z) 顺序。
    """
    # 1. 将 NumPy 数组转换为 SimpleITK 的 Image 对象
    # 【核心知识点 3】：GetImageFromArray 是个“死脑筋”，它永远默认你传入的数组是 (Z, Y, X) 顺序。
    # 如果你传入了 (Y, X, Z)，它会把 Y 当成 Z，导致图像物理空间彻底崩溃！
    itk_image = sitk.GetImageFromArray(numpy_image)
    
    # 2. 将物理参数重新绑定到 Image 对象上
    # 因为数组是 (Z, Y, X)，所以这里必须传入原始的 (X, Y, Z) 参数，才能完美对应。
    itk_image.SetOrigin(origin)
    itk_image.SetSpacing(spacing)
    itk_image.SetDirection(direction) # 必须设置 Direction，否则图像方向会错乱！
    
    # 3. 写入磁盘
    sitk.WriteImage(itk_image, filename)

# ================= 1. 读取 =================
file_path = r"F:\datasets\ATM22\TrainBatch1\before_yxz\data\CASE001data.nii.gz"
img_array, origin, spacing, direction = load_itk_image(file_path)

print(f"读取到更改前的数组: {img_array.shape}, 原点: {origin}, 间距: {spacing}, 方向: {direction}") 
# 输出应该是类似 (100, 512, 512)，代表 (Z, Y, X)

# ================= 2. 中间处理 (强烈建议不要转置) =================
# 假设你要做阈值处理、连通域、骨架提取等。
# 请始终记住，此时 img_array 的第0维是Z，第1维是Y，第2维是X。
# 例如：取中间切片 -> middle_slice = img_array[50, :, :] (取第50层Z)

# 【如果你非要转置成 (Y, X, Z) 进行处理】：
# img_array = img_array.transpose(1, 2, 0) 
# 此时 shape 变成了 (512, 512, 100)。
# 注意：此时 origin, spacing, direction 【绝对不要动】，保持原样！

# ================= 3. 保存 =================
# save_path = r"F:\datasets\ATM22\TrainBatch1\after_yxz\data\output.nii.gz"

# # 【如果你中间转置了，保存前必须转回来！】
# # if img_array.shape == (512, 512, 100): # 假设是 (Y, X, Z)
# #     img_array = img_array.transpose(2, 0, 1) # 转回 (Z, Y, X)

# # 调用保存函数，传入【原始的】物理参数
# img_array = img_array.transpose(1, 2, 0)#yxz
# origin = (origin[2], origin[0], origin[1])
# spacing = (spacing[2], spacing[0], spacing[1])
# direction = tuple([0,0,1,1,0,0,0,1,0])

# # 保存
# save_path = "output.nii.gz"
# save_itk_image(img_array, origin, spacing, direction, save_path)
# img_array, origin, spacing, direction = load_itk_image(save_path)

# print(f"读取到更改后的数组: {img_array.shape}, 原点: {origin}, 间距: {spacing}, 方向: {direction}") 
