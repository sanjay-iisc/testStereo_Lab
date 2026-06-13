import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from vision.datasets.build import (build_cityscapes_multitask_train, build_cityscapes_multitask_val)
from vision.models.multitask_unet import build_model

device = "cuda"

dataset = build_cityscapes_multitask_train("data/cityscapes")

loader = DataLoader(dataset,batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

model = build_model().to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

sky_loss_fn = nn.CrossEntropyLoss()
depth_loss_fn = nn.L1Loss()

batch = next(iter(loader))

images = batch["image"].to(device)
sky_mask = batch["sky_mask"].to(device)
disparity = batch["disparity"].to(device)

outputs = model(images)

loss_sky = sky_loss_fn(outputs["sky_logits"], sky_mask)
loss_depth = depth_loss_fn(outputs["depth"], disparity)

loss = loss_sky + 0.5 * loss_depth

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Sky loss:", loss_sky.item())
print("Depth loss:", loss_depth.item())
print("Total loss:", loss.item())