from torch.utils.data import DataLoader
from vision.datasets.build import build_cityscapes_multitask_train

dataset = build_cityscapes_multitask_train("data/cityscapes")

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

batch = next(iter(loader))

print("Images:", batch["image"].shape, batch["image"].dtype)
print("Sky:", batch["sky_mask"].shape, batch["sky_mask"].dtype)
print("Disparity:", batch["disparity"].shape, batch["disparity"].dtype)