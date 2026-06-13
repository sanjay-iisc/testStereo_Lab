from pathlib import Path

import torch

from vision.models.multitask_unet import build_model


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)

        sky_logits = outputs["sky_logits"]
        depth = outputs["depth"]

        return sky_logits, depth


def main():
    checkpoint_path = "outputs/checkpoints/best_model.pth"
    onnx_path = "models/multitask.onnx"

    Path("models").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    wrapped_model = ONNXWrapper(model).to(device)
    wrapped_model.eval()

    dummy_input = torch.randn(1, 3, 256, 512).to(device)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["sky_logits", "depth"],
        dynamic_axes=None,
    )

    print(f"ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    main()