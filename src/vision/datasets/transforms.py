import torch
import torchvision.transforms.functional as F


class SegmentationTransform:
    def __call__(self, image, mask):
        image = F.resize(image, (256, 512))
        mask = F.resize(mask, (256, 512), interpolation=F.InterpolationMode.NEAREST)

        image = F.to_tensor(image)
        mask = torch.as_tensor(list(mask.getdata()), dtype=torch.long)
        mask = mask.view(256, 512)

        return image, mask