import numpy as np
import SimpleITK as sitk

def analyze(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    size = img.GetSize()                              # ITK (x,y,z)
    spacing = np.array(img.GetSpacing())              # ITK (x,y,z)
    print("=" * 60)
    print(path)
    print("array.shape =", arr.shape, " GetSize =", size)
    print("spacing (x,y,z) =", spacing)
    print("各向同性?" , np.allclose(spacing, spacing[0]))

    # ground truth: 沿每个 numpy 轴走 1 步的真实物理距离
    # numpy axis a 对应 ITK axis (2-a)
    print("-" * 60)
    for a in range(3):
        itk_axis = 2 - a
        i0 = [0, 0, 0]; i0[itk_axis] = 0
        i1 = [0, 0, 0]; i1[itk_axis] = 1
        p0 = np.array(img.TransformContinuousIndexToPhysicalPoint(i0))
        p1 = np.array(img.TransformContinuousIndexToPhysicalPoint(i1))
        step = np.linalg.norm(p1 - p0)
        k = int(np.argmin([abs(step - spacing[j]) for j in range(3)]))
        print(f"numpy axis{a}: 真实步长={step:.4f}mm  ->  应配 spacing[{k}]={spacing[k]:.4f}")

    # 检查 resize 的假设: numpy axis a 用 spacing[a]
    print("-" * 60, "\nresize 假设 axis_a 用 spacing[a] 是否成立:")
    for a in range(3):
        itk_axis = 2 - a
        i1 = [0, 0, 0]; i1[itk_axis] = 1
        step = np.linalg.norm(np.array(img.TransformContinuousIndexToPhysicalPoint(i1))
                              - np.array(img.TransformContinuousIndexToPhysicalPoint([0,0,0])))
        k = int(np.argmin([abs(step - spacing[j]) for j in range(3)]))
        ok = (k == a) or np.isclose(spacing[a], spacing[k])
        print(f"  axis{a}: 代码用 spacing[{a}]={spacing[a]:.4f}, 物理应配 spacing[{k}]={spacing[k]:.4f}  ->  {'OK' if ok else 'MISMATCH!'}")

    # round-trip: 前景体素 numpy索引 -> 物理点 -> 回 numpy索引
    print("-" * 60, "\nround-trip 几何一致性:")
    nz = np.argwhere(arr > 0)
    if len(nz):
        na = nz[len(nz)//2]                            # 一个 numpy 索引 (axis0,axis1,axis2)
        itk = [int(na[2]), int(na[1]), int(na[0])]     # -> ITK (x,y,z)
        p = np.array(img.TransformContinuousIndexToPhysicalPoint(itk))
        back = img.TransformPhysicalPointToContinuousIndex(p)
        na_back = [back[2], back[1], back[0]]
        print(f"  numpy{tuple(na)} -> phys{np.round(p,2)} -> back{tuple(np.round(na_back,3))}  "
              f"{'OK' if np.allclose(na, na_back, atol=1e-3) else 'MISMATCH!'}")

if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        analyze(p)