"""Convert paired CT and mask NIfTI volumes from array order ZYX to YXZ."""

import argparse
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def nifti_files(folder):
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz"))
    )


def case_key(path):
    name = path.name
    if name.lower().endswith(".nii.gz"):
        name = name[:-7]
    elif name.lower().endswith(".nii"):
        name = name[:-4]
    return re.sub(r"(?i)(?:[_-]?(?:data|mask|ct|label))$", "", name)


def pair_by_case(ct_files, mask_files):
    def index_files(files, role):
        indexed = {}
        for path in files:
            key = case_key(path)
            if key in indexed:
                raise ValueError(
                    f"Duplicate {role} case key {key!r}: "
                    f"{indexed[key]} and {path}"
                )
            indexed[key] = path
        return indexed

    ct_by_case = index_files(ct_files, "CT")
    mask_by_case = index_files(mask_files, "mask")
    if ct_by_case.keys() != mask_by_case.keys():
        missing_masks = sorted(ct_by_case.keys() - mask_by_case.keys())
        missing_cts = sorted(mask_by_case.keys() - ct_by_case.keys())
        raise ValueError(
            "CT/mask cases do not match. "
            f"Missing masks for: {missing_masks}; missing CTs for: {missing_cts}"
        )
    return [(key, ct_by_case[key], mask_by_case[key]) for key in sorted(ct_by_case)]


def assert_same_geometry(ct_image, mask_image, case):
    if ct_image.GetSize() != mask_image.GetSize():
        raise ValueError(
            f"{case}: CT/mask sizes differ: "
            f"{ct_image.GetSize()} vs {mask_image.GetSize()}"
        )
    for field, ct_value, mask_value in (
        ("origin", ct_image.GetOrigin(), mask_image.GetOrigin()),
        ("spacing", ct_image.GetSpacing(), mask_image.GetSpacing()),
        ("direction", ct_image.GetDirection(), mask_image.GetDirection()),
    ):
        if not np.allclose(ct_value, mask_value, rtol=1e-5, atol=1e-5):
            raise ValueError(
                f"{case}: CT/mask {field} differ: {ct_value} vs {mask_value}"
            )


def transpose_zyx_image(image: sitk.Image) -> sitk.Image:
    """
    ZYX 数组 -> YXZ 数组，direction 列与 spacing 项随 numpy 轴循环平移。

    数组:       (sz, sy, sx) --transpose(1,2,0)--> (sy, sx, sz)
    direction:  old[:, [2, 0, 1]]     (ZXY)
    spacing:    (pz, px, py)
    origin:     不变
    """
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Only scalar 3D NIfTI images are supported.")

    # arr_zyx = sitk.GetArrayFromImage(image)                          # (sz, sy, sx)
    # arr_yxz = np.ascontiguousarray(arr_zyx.transpose(1, 2, 0))       # (sy, sx, sz)
    # out = sitk.GetImageFromArray(arr_yxz)

    # old_dir     = np.asarray(image.GetDirection()).reshape(3, 3)
    # old_spacing = image.GetSpacing()        # (px, py, pz)
    # old_origin  = image.GetOrigin()         # (ox, oy, oz)
    # new_dir     = old_dir[:, [2, 0, 1]]                              # ZXY 循环
    # new_spacing = (old_spacing[2], old_spacing[0], old_spacing[1])
    # new_origin  = old_origin
    # out.SetOrigin(new_origin)
    # out.SetSpacing(new_spacing)
    # out.SetDirection(tuple(new_dir.ravel()))
    out = sitk.PermuteAxes(image, (2, 0, 1))
    return out


def convert_dataset(ct_dir, mask_dir, output_dir):
    ct_dir = Path(ct_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    for folder in (ct_dir, mask_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {folder}")

    ct_files = nifti_files(ct_dir)
    mask_files = nifti_files(mask_dir)
    if not ct_files or not mask_files:
        raise FileNotFoundError("No .nii or .nii.gz files found in CT/mask inputs.")

    pairs = pair_by_case(ct_files, mask_files)
    ct_output_dir = output_dir / "data"
    mask_output_dir = output_dir / "mask"
    ct_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    for case, ct_path, mask_path in pairs:
        ct_image = sitk.ReadImage(str(ct_path))
        mask_image = sitk.ReadImage(str(mask_path))
        assert_same_geometry(ct_image, mask_image, case)

        ct_result = transpose_zyx_image(ct_image)
        mask_result = transpose_zyx_image(mask_image)

        ct_destination = ct_output_dir / (ct_path.name[:-7] + ".nii.gz")
        mask_destination = mask_output_dir / (mask_path.name[:-7] + ".nii.gz")

        sitk.WriteImage(ct_result, str(ct_destination))
        sitk.WriteImage(mask_result, str(mask_destination))
        print(
            f"{case}: {tuple(reversed(ct_image.GetSize()))} -> "
            f"{tuple(reversed(ct_result.GetSize()))}"
        )

    print(f"Converted {len(pairs)} CT/mask pairs to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transpose paired NIfTI arrays from SimpleITK array order "
            "ZYX to YXZ (direction columns cycle with numpy axes)."
        )
    )
    parser.add_argument("--ct-dir", default=r"F:\datasets\ATM22\TrainBatch1\before\data")
    parser.add_argument("--mask-dir", default=r"F:\datasets\ATM22\TrainBatch1\before\mask")
    parser.add_argument("--output-dir", default=r"F:\datasets\ATM22\TrainBatch1\before")
    args = parser.parse_args()
    convert_dataset(args.ct_dir, args.mask_dir, args.output_dir)


if __name__ == "__main__":
    main()