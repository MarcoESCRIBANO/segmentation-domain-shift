import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

IGNORE_LABEL = 255  # will encode as (255, 255, 255)

MAPILLARY_RGB_TO_CITYSCAPES_TRAINID = {
    (128, 64, 128): 0,   # Road
    (244, 35, 232): 1,   # Sidewalk
    (70, 70, 70): 2,     # Building
    (102,102,156): 3,   # Wall
    (190,153,153): 4,   # Fence
    (153,153,153): 5,   # Pole
    (250,170,30): 6,    # Traffic light (all variants)
    (220,220,0): 7,     # Traffic sign (front)
    (107,142,35): 8,    # Vegetation
    (152,251,152): 9,   # Terrain
    (70,130,180): 10,   # Sky
    (220,20,60): 11,    # Person
    (255,0,0): 12,      # Bicyclist
    (255,0,100): 12,    # Motorcyclist
    (255,0,200): 12,   # Other rider
    (0,0,142): 13,      # Car
    (0,0,70): 14,       # Truck
    (0,60,100): 15,     # Bus
    (0,80,100): 16,     # On rails → train
    (0,0,230): 17,      # Motorcycle
    (119,11,32): 18,    # Bicycle
}


# CamVid RGB -> Cityscapes trainId grayscale
CAMVID_RGB_TO_CITYSCAPES_TRAINID = {
    # Flat
    (128, 64,128): 0,    # Road
    (0, 0,192): 1,       # Sidewalk
    (128, 0,192): 0,     # LaneMkgsDriv
    (192, 0,64): 0,      # LaneMkgsNonDriv

    # Construction
    (128,0,0): 2,        # Building
    (64,192,0): 3,       # Wall
    (64,64,128): 4,      # Fence
    (192,0,128): 2,      # Archway -> building

    # Nature
    (128,128,0): 8,      # Tree
    (192,192,0): 8,      # VegetationMisc
    (128,128,128): 10,   # Sky

    # Objects
    (192,192,128): 5,    # Column_Pole
    (0,64,64): 6,        # TrafficLight
    (192,128,128): 7,    # SignSymbol

    # Humans
    (64,64,0): 11,       # Pedestrian
    (192,128,64): 11,    # Child
    (0,128,192): 12,     # Bicyclist
    (192,0,192): 12,     # MotorcycleScooter

    # Vehicles
    (64,0,128): 13,      # Car
    (64,128,192): 13,    # SUVPickupTruck
    (192,128,192): 14,   # Truck_Bus
    (192,64,128): 16,    # Train
    (0,0,230): 17,       # Motorcycle
    (119,11,32): 18,     # Bicycle
}


CITYSCAPES_ID_TO_TRAINID = {
   # id   trainId
     0  : 255 ,
     1  : 255 ,
     2  : 255 ,
     3  : 255 ,
     4  : 255 ,
     5  : 255 ,
     6  : 255 ,
     7  :   0 ,
     8  :   1 ,
     9  : 255 ,
     0  : 255 ,
     11 :   2 ,
     12 :   3 ,
     13 :   4 ,
     14 : 255 ,
     15 : 255 ,
     16 : 255 ,
     17 :   5 ,
     18 : 255 ,
     19 :   6 ,
     20 :   7 ,
     21 :   8 ,
     22 :   9 ,
     23 :  10 ,
     24 :  11 ,
     25 :  12 ,
     26 :  13 ,
     27 :  14 ,
     28 :  15 ,
     29 : 255 ,
     30 : 255 ,
     31 :  16 ,
     32 :  17 ,
     33 :  18 ,
}

def remap_to_grayscale(mask_rgb: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Convert RGB mask to 3-channel grayscale trainId mask
    """
    h, w, _ = mask_rgb.shape
    remapped = np.full((h, w), IGNORE_LABEL, dtype=np.uint8)

    for rgb, train_id in mapping.items():
        matches = np.all(mask_rgb == rgb, axis=-1)
        remapped[matches] = train_id

    # Convert to 3-channel grayscale for visualization
    remapped_rgb = np.stack([remapped]*3, axis=-1)
    return remapped_rgb

def process_directory(input_dir: str, output_dir: str, mapping: dict):
    os.makedirs(output_dir, exist_ok=True)
    
    # Recursively gather all image files
    mask_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):  # include more formats if needed
                mask_files.append(os.path.join(root, f))

    for in_path in tqdm(mask_files, desc="Remapping masks"):
        # Preserve relative folder structure
        rel_path = os.path.relpath(in_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        mask_rgb = np.array(Image.open(in_path).convert("RGB"))
        remapped = remap_to_grayscale(mask_rgb, mapping)

        Image.fromarray(remapped).save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", choices=["camvid", "cityscapes", "mapillary"], default="camvid")

    args = parser.parse_args()

    if args.dataset == "camvid":
        mapping = CAMVID_RGB_TO_CITYSCAPES_TRAINID
    elif args.dataset == "mapillary":
        mapping = MAPILLARY_RGB_TO_CITYSCAPES_TRAINID
    else:
        mapping = CITYSCAPES_ID_TO_TRAINID

    process_directory(args.input_dir, args.output_dir, mapping)