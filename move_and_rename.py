import os, re, shutil
from glob import glob

SRC_IMAGES = r"F:\datasets\ATM22\TrainBatch1\before\data"
SRC_LABELS = r"F:\datasets\ATM22\TrainBatch1\before\mask"
DST_ROOT   = r"F:\datasets\ATM22\TrainBatch1\before"

os.makedirs(os.path.join(DST_ROOT,"data"), exist_ok=True)
os.makedirs(os.path.join(DST_ROOT,"mask"), exist_ok=True)

# imagesTr 里的是 CT
ct_files   = sorted(glob(os.path.join(SRC_IMAGES, "*.nii.gz")))
# labelsTr 里的是 Mask（同名配对）
mask_files = sorted(glob(os.path.join(SRC_LABELS, "*.nii.gz")))

print(f"找到 {len(ct_files)} 个 CT, {len(mask_files)} 个 Mask")

ok = 0
for ct_path, mask_path in zip(ct_files, mask_files):
    m = os.path.basename(ct_path)
    if not m: continue
    a=m.split(".nii.gz")[0]
    shutil.move(ct_path,   os.path.join(DST_ROOT, "data", "CASE"+a+"data.nii.gz"))
    shutil.move(mask_path, os.path.join(DST_ROOT, "mask", "CASE"+a+"mask.nii.gz"))
    ok += 1

print(f"\n✅ 完成！{ok} 例 → {DST_ROOT}")
print(f"   data/  有 {len(os.listdir(os.path.join(DST_ROOT,'data')))} 个文件")
print(f"   mask/  有 {len(os.listdir(os.path.join(DST_ROOT,'mask')))} 个文件")
print(f"\n原始 imagesTr / labelsTr 已空，可以删了（压缩包保留即可）")