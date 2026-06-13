import matplotlib.pyplot as plt

from vision.datasets.build import build_cityscapes_multitask_train

dataset = build_cityscapes_multitask_train("data/cityscapes")

sample = dataset[0]

image = sample["image"].permute(1, 2, 0).numpy()
sky = sample["sky_mask"].numpy()
disp = sample["disparity"][0].numpy()


print("Min:", sample["disparity"].min())
print("Max:", sample["disparity"].max())

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(image)
plt.title("RGB")

plt.subplot(132)
plt.imshow(sky)
plt.title("Sky Mask")

plt.subplot(133)
plt.imshow(disp)
plt.title("Disparity")

plt.show()