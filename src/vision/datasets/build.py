from pathlib import Path

from vision.datasets.cityscapes import CityscapesMultiTaskDataset


def get_cityscapes_multitask_pairs(data_root, split):
    data_root = Path(data_root)

    image_root = data_root / "leftImg8bit" / split
    mask_root = data_root / "gtFine" / split
    disparity_root = data_root / "disparity" / split

    image_files = []
    mask_files = []
    disparity_files = []

    for image_path in image_root.rglob("*_leftImg8bit.png"):
        city = image_path.parent.name

        mask_name = image_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")

        disparity_name = image_path.name.replace("_leftImg8bit.png", "_disparity.png")

        mask_path = mask_root / city / mask_name
        disparity_path = disparity_root / city / disparity_name

        if mask_path.exists() and disparity_path.exists():
            image_files.append(str(image_path))
            mask_files.append(str(mask_path))
            disparity_files.append(str(disparity_path))

    return image_files, mask_files, disparity_files


def build_cityscapes_multitask_train(data_root):
    images, masks, disparities = get_cityscapes_multitask_pairs(data_root, "train")
    return CityscapesMultiTaskDataset(images,masks,disparities)

def build_cityscapes_multitask_val(data_root):
    images, masks, disparities = get_cityscapes_multitask_pairs(data_root, "val")

    return CityscapesMultiTaskDataset(images,masks,disparities)

def build_cityscapes_multitask_test(data_root):
    images, masks, disparities = get_cityscapes_multitask_pairs(data_root, "test")
    return CityscapesMultiTaskDataset(images,masks,disparities)