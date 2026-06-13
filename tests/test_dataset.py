from vision.datasets.build import (build_cityscapes_multitask_train, build_cityscapes_multitask_val)

train_dataset = build_cityscapes_multitask_train("data/cityscapes")
val_dataset = build_cityscapes_multitask_val("data/cityscapes")

print("Train:", len(train_dataset))
print("Val:", len(val_dataset))

sample = train_dataset[0]

print("Image:", sample["image"].shape, sample["image"].dtype)
print("Sky mask:", sample["sky_mask"].shape, sample["sky_mask"].dtype)
print("Disparity:", sample["disparity"].shape, sample["disparity"].dtype)

print("Sky unique values:", sample["sky_mask"].unique())
print("Disparity min:", sample["disparity"].min())
print("Disparity max:", sample["disparity"].max())