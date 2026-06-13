import argparse
from pathlib import Path

import json
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from vision.training.metrics import sky_metrics, depth_metrics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from vision.datasets.build import (build_cityscapes_multitask_train, build_cityscapes_multitask_val)
from vision.models.multitask_unet import build_model



def train_one_epoch(model, loader, optimizer, device, depth_weight):
    model.train()

    sky_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()

    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        sky_mask = batch["sky_mask"].to(device)
        disparity = batch["disparity"].to(device)

        outputs = model(images)

        loss_sky = sky_loss_fn(outputs["sky_logits"], sky_mask)
        loss_depth = depth_loss_fn(outputs["depth"], disparity)

        loss = loss_sky + depth_weight * loss_depth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, depth_weight):
    model.eval()

    sky_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()

    total_loss = 0.0
    total_sky_iou = 0.0
    total_sky_dice = 0.0
    total_sky_acc = 0.0
    total_depth_mae = 0.0
    total_depth_rmse = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        sky_mask = batch["sky_mask"].to(device)
        disparity = batch["disparity"].to(device)

        outputs = model(images)

        loss_sky = sky_loss_fn(outputs["sky_logits"], sky_mask)
        loss_depth = depth_loss_fn(outputs["depth"], disparity)

        loss = loss_sky + depth_weight * loss_depth

        sky_dice, sky_iou, sky_acc = sky_metrics(outputs["sky_logits"], sky_mask)
        depth_mae, depth_rmse = depth_metrics(outputs["depth"], disparity)

        total_loss += loss.item()
        total_sky_iou += sky_iou
        total_sky_dice += sky_dice
        total_sky_acc += sky_acc
        total_depth_mae += depth_mae
        total_depth_rmse += depth_rmse

    n = len(loader)

    return {"loss": total_loss / n,
            "sky_dice": total_sky_dice / n,
            "sky_iou": total_sky_iou / n,
            "sky_acc": total_sky_acc / n,
            "depth_mae": total_depth_mae / n,
            "depth_rmse": total_depth_rmse / n}

def save_training_curves(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["sky_iou"], label="Sky IoU")
    plt.plot(epochs, history["sky_dice"], label="Sky Dice")
    plt.plot(epochs, history["sky_acc"], label="Sky Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Sky Segmentation Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "sky_metrics.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["depth_mae"], label="Depth MAE")
    plt.plot(epochs, history["depth_rmse"], label="Depth RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Depth Estimation Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "depth_metrics.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/cityscapes")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--depth-weight", type=float, default=0.5)
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/checkpoints")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dataset = build_cityscapes_multitask_train(args.data_root)
    val_dataset = build_cityscapes_multitask_val(args.data_root)

    if args.subset > 0:
        train_dataset = Subset(train_dataset, range(min(args.subset, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(50, len(val_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(device)

    optimizer = torch.optim.AdamW( model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")

    history = { "train_loss": [], "val_loss": [], "sky_iou": [], "sky_acc": [], "depth_mae": [], "depth_rmse": [], "sky_dice": []}


    for epoch in range(args.epochs):
        train_loss = train_one_epoch( model, train_loader, optimizer, device, args.depth_weight)

        val_metrics = validate(model, val_loader, device, args.depth_weight)

        val_loss = val_metrics["loss"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["sky_iou"].append(val_metrics["sky_iou"])
        history["sky_acc"].append(val_metrics["sky_acc"])
        history["depth_mae"].append(val_metrics["depth_mae"])
        history["depth_rmse"].append(val_metrics["depth_rmse"])
        history["sky_dice"].append(val_metrics["sky_dice"])

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"sky_iou={val_metrics['sky_iou']:.4f} "
            f"sky_acc={val_metrics['sky_acc']:.4f} "
            f"depth_mae={val_metrics['depth_mae']:.4f} "
            f"depth_rmse={val_metrics['depth_rmse']:.4f} "
            f"sky_dice={val_metrics['sky_dice']:.4f} ")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(args.output_dir) / "best_model.pth"
            torch.save({"model_state_dict": model.state_dict(), "val_loss": val_loss, "epoch": epoch + 1,}, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
    save_training_curves(history, "outputs/metrics")

    with open("outputs/metrics/history.json", "w") as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    main()