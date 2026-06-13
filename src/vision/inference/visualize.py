from pathlib import Path

import matplotlib.pyplot as plt
import torch

from vision.datasets.build import build_cityscapes_multitask_test
from vision.models.multitask_unet import build_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "outputs/checkpoints/best_model.pth"
    output_path = "outputs/predictions/prediction.png"

    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)

    dataset = build_cityscapes_multitask_test("data/cityscapes")
    sample = dataset[0]

    image = sample["image"].unsqueeze(0).to(device)

    gt_sky = sample["sky_mask"].cpu().numpy()
    gt_disp = sample["disparity"][0].cpu().numpy()

    model = build_model().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    with torch.no_grad():
        outputs = model(image)

    pred_sky = outputs["sky_logits"].argmax(dim=1)[0].cpu().numpy()
    pred_depth = outputs["depth"][0, 0].cpu().numpy()

    rgb = sample["image"].permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 5, 1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(gt_sky)
    plt.title("GT Sky")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(pred_sky)
    plt.title("Pred Sky")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(gt_disp)
    plt.title("GT Disparity")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(pred_depth)
    plt.title("Pred Depth")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()