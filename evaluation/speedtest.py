import time
import os
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from data_preparation.dataloader import SegmentationDataset



# -------------------------
# Config
# -------------------------
DATA_DIR = "data"
NUM_CLASSES = 19  
TARGET_SIZE = (512, 512) # (W, H)

# -------------------------
# Transforms
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "model_state" in checkpoint:
            return checkpoint["model_state"]
    return checkpoint  


def evaluate_fps(model, dataloader, device="cpu", warmup=15, max_batches=None):
    """
    Evaluate FPS of a segmentation model.

    Args:
        model: PyTorch model
        dataloader: DataLoader returning images (and optionally targets)
        device: "cpu", "cuda", or "mps"
        warmup: number of warmup iterations to skip from timing
        max_batches: max number of batches to run (for faster testing)

    Returns:
        avg_fps: average frames per second
        avg_time: average inference time per image (seconds)
    """
    model.eval()
    model.to(device)
    times = []

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dataloader, desc="FPS evaluation")):
            # Optional: stop early
            if max_batches is not None and i >= max_batches:
                break

            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)

            start_time = time.time()
            _ = model(images)  # forward pass
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.time() - start_time

            # skip warmup iterations
            if i >= warmup:
                times.append(elapsed / images.shape[0])  # time per image

    if not times:
        return None, None

    avg_time = sum(times) / len(times)
    avg_fps = 1.0 / avg_time
    print(f"Average FPS: {avg_fps:.2f}, Average inference time: {avg_time:.4f}s/image")

    return avg_fps, avg_time

# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)  
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    val_dataset = SegmentationDataset(
            image_dir=os.path.join(DATA_DIR, args.image_dir),
            mask_dir=os.path.join(DATA_DIR, args.mask_dir),
            target_size=TARGET_SIZE,
            transform=transform
        )

    val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            pin_memory=(device.type == "cuda")
        )

    model = deeplabv3_resnet50(weights=None)

    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = extract_model_state(checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    avg_fps, avg_time = evaluate_fps(model, val_loader, device="mps")
