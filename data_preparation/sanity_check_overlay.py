import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# -------------------------------------------------
# Cityscapes colormap (trainId -> RGB)
# -------------------------------------------------
CITYSCAPES_COLORMAP = {
    0:  (128, 64, 128),   # road
    1:  (244, 35, 232),   # sidewalk
    2:  (70, 70, 70),     # building
    3:  (102, 102, 156),  # wall
    4:  (190, 153, 153),  # fence
    5:  (153, 153, 153),  # pole
    6:  (250, 170, 30),   # traffic light
    7:  (220, 220, 0),    # traffic sign
    8:  (107, 142, 35),   # vegetation
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
}

IGNORE_LABEL = 255

# -------------------------------------------------
# Mask colorization
# -------------------------------------------------
def colorize_mask(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert 3-channel grayscale mask to Cityscapes color visualization
    """
    h, w, _ = mask_rgb.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    train_ids = mask_rgb[..., 0]

    for train_id, color in CITYSCAPES_COLORMAP.items():
        color_mask[train_ids == train_id] = color

    color_mask[train_ids == IGNORE_LABEL] = (0, 0, 0)

    return color_mask

def overlay(image: np.ndarray, mask_color: np.ndarray, alpha=0.5):
    return (image * (1 - alpha) + mask_color * alpha).astype(np.uint8)

# -------------------------------------------------
# Main processing
# -------------------------------------------------
def process(image_dir, mask_dir, output_dir, max_samples=30):
    random.seed(42)
    
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    mask_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        mask_files.extend(mask_dir.rglob(ext))


    if max_samples is not None:
        mask_files = random.sample(
            mask_files,
            min(max_samples, len(mask_files))
        )
    

    for mask_path in tqdm(mask_files, desc="Generating overlays"):
        # preserve city subfolder
        rel_path = mask_path.relative_to(mask_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Default: same filename (CamVid-style)
        img_path = (image_dir / rel_path).with_suffix(".png")

        # Cityscapes fallback
        if not img_path.exists():
            base = img_path.stem.replace("_gtFine_labelIds", "")
            img_path = (
                image_dir
                / rel_path.parent
                / f"{base}_leftImg8bit.png"
            )

        # Mapillary fallback
        if not img_path.exists():
            img_path = (image_dir / rel_path).with_suffix(".jpg")

        if not img_path.exists():
            raise FileNotFoundError(f"No matching image for {rel_path}")

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path)

        img = np.array(img_pil)
        mask = np.array(mask_pil)

        # -------------------------------------------------
        # Resize mask to image size if needed
        # -------------------------------------------------
        if mask.shape[:2] != img.shape[:2]:
            mask_pil = mask_pil.resize(
                (img.shape[1], img.shape[0]),  # (W, H)
                resample=Image.NEAREST
            )
            mask = np.array(mask_pil)


        if mask.ndim == 3 and mask.shape[2] == 3:
            mask_color = colorize_mask(mask)
        else:
            raise ValueError("Expected 3-channel grayscale mask")

        blended = overlay(img, mask_color)

        Image.fromarray(blended).save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=30)

    args = parser.parse_args()
    process(
        args.image_dir,
        args.mask_dir,
        args.output_dir,
        args.max_samples
    )
