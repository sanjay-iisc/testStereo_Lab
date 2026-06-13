from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class CityscapesMultiTaskDataset(Dataset):
    def __init__(self, image_files, mask_files, disparity_files, size=(256, 512)):
        self.image_files = image_files
        self.mask_files = mask_files
        self.disparity_files = disparity_files
        self.size = size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx])
        disparity = Image.open(self.disparity_files[idx])

        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        disparity = F.resize(disparity, self.size, interpolation=F.InterpolationMode.NEAREST)

        image = F.to_tensor(image)

        mask = torch.as_tensor(list(mask.getdata()), dtype=torch.long)
        mask = mask.view(self.size[0], self.size[1])

        # Cityscapes 19-class trainId: sky = 10
        sky_mask = (mask == 23).long()

        disparity = torch.as_tensor(list(disparity.getdata()), dtype=torch.float32)
        disparity = disparity.view(self.size[0], self.size[1])

        # Normalize disparity roughly to [0, 1]
        disparity = disparity.float()
        valid = disparity > 0
        if valid.any():
            disparity[valid] = (disparity[valid] - disparity[valid].min()) / (disparity[valid].max() - disparity[valid].min() + 1e-8)
        # disparity = disparity / 256.0
        # disparity = disparity.unsqueeze(0)
        disparity[~valid] = 0.0
        disparity = disparity.unsqueeze(0)

        return {"image": image, "sky_mask": sky_mask, "disparity": disparity}