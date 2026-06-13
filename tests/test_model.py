import torch

from vision.models.multitask_unet import build_model

model = build_model().cuda()
model.eval()

x = torch.randn(2, 3, 256, 512).cuda()

with torch.no_grad():
    outputs = model(x)

print("Sky:", outputs["sky_logits"].shape)
print("Depth:", outputs["depth"].shape)
print("Depth min:", outputs["depth"].min())
print("Depth max:", outputs["depth"].max())