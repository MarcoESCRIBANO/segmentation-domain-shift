import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
from data_preparation.dataloader import SegmentationDataset
from evaluation.evaluate import evaluate_dataset

# -------------------------
# Config
# -------------------------
DATA_DIR = "data"
NUM_CLASSES = 19 
LR = 1e-4
EPOCHS = 50
TARGET_SIZE = (512, 512) # (W, H)


# -------------------------
# Transforms
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", required=True)
    parser.add_argument("--train_mask_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--val_mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default=None)  # optional
    parser.add_argument("--device", default=None)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--train_num_workers", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--val_num_workers", type=int, default=4)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--save_checkpoint_interval", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------
    # Device setup
    # -------------------------
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

    # -------------------------
    # Dataloaders
    # -------------------------
    train_dataset = SegmentationDataset(
        image_dir=os.path.join(DATA_DIR, args.train_image_dir),
        mask_dir=os.path.join(DATA_DIR, args.train_mask_dir),
        target_size=TARGET_SIZE,
        transform=transform
    )

    val_dataset = SegmentationDataset(
        image_dir=os.path.join(DATA_DIR, args.val_image_dir),
        mask_dir=os.path.join(DATA_DIR, args.val_mask_dir),
        target_size=TARGET_SIZE,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=(device.type == "cuda")
    )


    # -------------------------
    # Model
    # -------------------------
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model = model.to(device)

    # -------------------------
    # Loss and Optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # assuming 255 is ignore_label
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    start_epoch = 0

    # -------------------------
    # Resume checkpoint
    # -------------------------
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0

        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f}")

        # -------------------------
        # Validation
        # -------------------------
        if (epoch + 1) % args.val_interval == 0:
            print("Running validation...")

            val_results = evaluate_dataset(
                model=model,
                loader=val_loader,
                output_dir=os.path.join(args.output_dir, "val"),
                device=device,
                save_predictions=False,  # avoid saving images during training
                training=True # Add loss to metrics
            )

            print(
                f"[VAL] Epoch {epoch+1} | "
                f"Val Loss: {val_results['avg_val_loss']:.4f} | "
                f"mIoU: {val_results['mean_iou']:.4f} | "
                f"Pixel Acc: {val_results['pixel_accuracy']:.4f}"
            )

            # Save metrics 
            metrics_to_save = {
                "epoch": epoch + 1,
                **val_results
            }

            metrics_path = os.path.join(args.output_dir, "val_metrics.jsonl")

            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics_to_save) + "\n")
        
        # -------------------------
        # Checkpointing
        # -------------------------
        if (epoch + 1) % args.save_checkpoint_interval == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"deeplab_epoch_{epoch+1}.pth"
            )
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Always save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, os.path.join(args.output_dir, "last.pth"))
