from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.network import CrocodileCrltgaNet
from src.utils.config import load_config
from src.utils.io import ensure_dir


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def overlay_heatmap(image: Image.Image, heatmap: torch.Tensor, output_path: Path) -> None:
    image = image.resize((heatmap.shape[-1], heatmap.shape[-2])).convert("RGB")
    image_np = torch.from_numpy(__import__("numpy").array(image)).float() / 255.0
    cmap = plt.get_cmap("jet")
    colored = torch.from_numpy(cmap(heatmap.cpu().numpy())[:, :, :3]).float()
    overlay = 0.55 * image_np + 0.45 * colored
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay.clamp(0, 1).numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    model = CrocodileCrltgaNet(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0].detach())

    handle_forward = model.backbone.stem[-1][-1].conv3.register_forward_hook(forward_hook)
    handle_backward = model.backbone.stem[-1][-1].conv3.register_full_backward_hook(backward_hook)

    raw_image = Image.open(args.image).convert("RGB")
    tensor = build_transform(config["image_size"])(raw_image).unsqueeze(0).to(device)
    outputs = model(tensor)
    probs = torch.sigmoid(outputs["disease_logits"])[0]
    class_index = int(torch.argmax(probs).item())
    warning_score = float(outputs["spurious_warning_score"][0].item())

    model.zero_grad(set_to_none=True)
    outputs["disease_logits"][0, class_index].backward()
    grad = gradients[-1][0]
    act = activations[-1][0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * act).sum(dim=0)
    cam = F.relu(cam)
    cam = cam / cam.max().clamp_min(1e-6)
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(config["image_size"], config["image_size"]), mode="bilinear", align_corners=False)[0, 0]

    output_dir = ensure_dir(Path(config["output_root"]) / "inference")
    output_path = output_dir / (Path(args.image).stem + "_gradcam.png")
    overlay_heatmap(raw_image, cam, output_path)

    print("Predictions:")
    for label_name, prob in zip(config["label_names"], probs.tolist()):
        print(f"{label_name}: {prob:.4f}")
    print(f"Spurious warning score: {warning_score:.4f}")
    if warning_score >= 0.6:
        print("Warning: decision may be influenced by domain-specific or spurious cues.")
    print(f"Grad-CAM saved to: {output_path}")

    handle_forward.remove()
    handle_backward.remove()


if __name__ == "__main__":
    main()
