import torch


@torch.no_grad()
def sky_metrics(logits, target):
    pred = logits.argmax(dim=1)

    # Pixel accuracy
    accuracy = (pred == target).float().mean()

    pred_sky = pred == 1
    target_sky = target == 1

    intersection = (pred_sky & target_sky).sum().float()

    union = (pred_sky | target_sky).sum().float()

    pred_area = pred_sky.sum().float()
    target_area = target_sky.sum().float()

    # IoU
    if union == 0:
        iou = torch.tensor(1.0, device=logits.device)
    else:
        iou = intersection / (union + 1e-8)

    # Dice
    if pred_area + target_area == 0:
        dice = torch.tensor(1.0, device=logits.device)
    else:
        dice = (2.0 * intersection) / (pred_area + target_area + 1e-8)

    return (dice.item(), iou.item(), accuracy.item())


@torch.no_grad()
def depth_metrics(pred, target):
    valid = target > 0

    if valid.sum() == 0:
        return 0.0, 0.0

    pred_valid = pred[valid]
    target_valid = target[valid]

    mae = torch.mean(torch.abs(pred_valid - target_valid))
    rmse = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2))

    return mae.item(), rmse.item()

