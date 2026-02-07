import torch
import numpy as np
from segmentation_models_pytorch.metrics.functional import (
    get_stats,
    iou_score,
    accuracy,
)

IGNORE_LABEL = 255

def _to_train_id(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        return mask[..., 0]
    return mask

def evaluate_segmentation(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    # Convert masks
    pred = _to_train_id(pred)
    target = _to_train_id(target)

    # Mask IGNORE_LABEL
    valid = target != IGNORE_LABEL
    pred = pred[valid]
    target = target[valid]

    # Convert to tensors
    pred_tensor = torch.tensor(pred, dtype=torch.long).unsqueeze(0)  # shape [1, H*W]
    if isinstance(target, torch.Tensor):
        target_tensor = target.clone().detach().long().unsqueeze(0)
    else:
        target_tensor = torch.from_numpy(target).long().unsqueeze(0)

    # Compute stats for multi-class
    tp, fp, fn, tn = get_stats(
        pred_tensor,
        target_tensor,
        mode="multiclass",
        ignore_index=None,   # already masked
        num_classes=num_classes,
    )

    # IoU for each class (reduction=None → tensor [1, C])
    iou_per_class = iou_score(tp, fp, fn, tn, reduction=None).squeeze(0)

    # Pixel accuracy
    pix_acc = accuracy(tp, fp, fn, tn, reduction="micro").item()

    

    # Format results
    per_class = {cls: float(iou_per_class[cls]) for cls in range(num_classes)}
    mean_iou = float(iou_per_class.mean().item())

    metrics = {
        "pixel_accuracy": pix_acc,
        "mean_iou": mean_iou,
        "per_class_iou": per_class,
    }


    return metrics

