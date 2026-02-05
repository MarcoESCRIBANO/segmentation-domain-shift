import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path

# ---------------------
# Dataset Class
# ---------------------
class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        target_size=(512, 512),
        transform=None,
        recursive=True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.target_size = target_size

        # ---------------------
        # Collect image files
        # ---------------------
        if recursive:
            self.files = sorted(
                p.relative_to(self.image_dir)
                for p in self.image_dir.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
        else:
            self.files = sorted(
                p.name
                for p in self.image_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rel_path = Path(self.files[idx])
        img_path = self.image_dir / rel_path

        # ---------------------
        # Default mask path (CamVid-style)
        # ---------------------
        mask_path = (self.mask_dir / rel_path).with_suffix(".png")

        # ---------------------
        # Cityscapes fallback
        # ---------------------
        if not mask_path.exists():
            base = img_path.stem.replace("_leftImg8bit", "")
            mask_path = (
                self.mask_dir
                / rel_path.parent
                / f"{base}_gtFine_labelIds.png"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"No matching mask for image:\n"
                f"  image: {img_path}\n"
                f"  tried: {mask_path}"
            )

        # ---------------------
        # Load image & mask
        # ---------------------
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # ---------------------
        # Resize
        # ---------------------
        if self.target_size is not None:
            img = img.resize(self.target_size, Image.BILINEAR)
            mask = mask.resize(self.target_size, Image.NEAREST)

        # ---------------------
        # To tensor
        # ---------------------
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()
    
        if self.transform:
            img = self.transform(img)

        return img, mask, img_path.name
