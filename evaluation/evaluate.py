import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from evaluation.metrics import evaluate_segmentation
from data_preparation.dataloader import SegmentationDataset


CS = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}
IGNORE_LABEL = 255
NUM_CLASSES = 19  # Cityscapes trainIds: 0–18
TARGET_SIZE = (512, 512) # (W, H)

# ---------------------
# Preprocessing
# ---------------------
preprocess = transforms.Compose([
    transforms.ToTensor(),  # HWC [0,255] -> CHW [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ---------------------
# Tools
# ---------------------
def colorize_mask(mask: np.ndarray, colormap: dict) -> np.ndarray:
    """
    mask: (H, W) uint8 trainId mask
    returns: (H, W, 3) uint8 RGB image
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for train_id, color in colormap.items():
        rgb[mask == train_id] = color

    return rgb

def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "model_state" in checkpoint:
            return checkpoint["model_state"]
    return checkpoint  

# ---------------------
# Evaluation Functions
# ---------------------
def aggregate_metrics(metrics_list):
    """
    Average metrics over dataset
    """
    pixel_acc = np.mean([m["pixel_accuracy"] for m in metrics_list])
    mean_iou = np.mean([m["mean_iou"] for m in metrics_list])

    per_class = {}
    for cls_id in range(NUM_CLASSES):
        vals = [
            m["per_class_iou"].get(cls_id)  
            for m in metrics_list
            if not np.isnan(m["per_class_iou"].get(cls_id, np.nan))
        ]
        per_class[CS[cls_id]] = float(np.mean(vals)) if vals else None 
        
    metrics = {
        "pixel_accuracy": float(pixel_acc),
        "mean_iou": float(mean_iou),
        "per_class_iou": per_class,
    }

    losses = [m["loss"] for m in metrics_list if "loss" in m]
    if losses:
        avg_val_loss = np.mean(losses)
        metrics["avg_val_loss"] = float(avg_val_loss)

    return metrics

def evaluate_dataset(model, loader, output_dir, device, save_predictions=True, training=False, save_predictions_interval=1):
    os.makedirs(output_dir, exist_ok=True)
    if save_predictions:
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    
    all_metrics = []
    model.eval()
    with torch.no_grad():
        for imgs, targets, fnames in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            pred_logits = model(imgs)['out']  # [B,C,H,W]
            preds = torch.argmax(pred_logits, dim=1).cpu().numpy()  # [B,H,W]
            
            for i in range(len(fnames)):
                pred = preds[i]
                target = targets[i]

                metrics = evaluate_segmentation(
                    pred=pred,
                    target=target,
                    num_classes=NUM_CLASSES,
                )

                # Loss
                if training:
                    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
                    target_tensor = target.unsqueeze(0).to(device).long()
                    loss = criterion(pred_logits[i:i+1], target_tensor)
                    metrics["loss"] = loss.item()

                all_metrics.append(metrics)

                if save_predictions and i % save_predictions_interval == 0:
                    pred_u8 = pred.astype(np.uint8)          # (H, W)
                    pred_rgb = np.repeat(
                        pred_u8[..., None], 3, axis=2        # (H, W, 3)
                    )

                    Image.fromarray(pred_rgb, mode="RGB").save(
                        os.path.join(output_dir, "predictions", fnames[i])
                    )


    return aggregate_metrics(all_metrics)

# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default=None)  # optional
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_predictions", action='store_true')
    parser.add_argument("--save_predictions_interval", default=None, type=int)
    args = parser.parse_args()

    # ---------------------
    # Device setup
    # ---------------------
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


    # ---------------------
    # Model setup
    # ---------------------
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = extract_model_state(checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    model.to(device)

    # ---------------------
    # Dataset & Loader
    # ---------------------
    dataset = SegmentationDataset(args.image_dir, args.mask_dir, target_size=TARGET_SIZE, transform=preprocess)
    if args.max_samples:
        dataset.files = dataset.files[:args.max_samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device != "cpu" else False,
    )

    # ---------------------
    # Evaluation
    # ---------------------
    results = evaluate_dataset(
        model=model,
        loader=loader,
        output_dir=args.output_dir,
        device=device,
        save_predictions=args.save_predictions,
        save_predictions_interval=args.save_predictions_interval
    )

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete.")
    print(json.dumps(results, indent=2))
