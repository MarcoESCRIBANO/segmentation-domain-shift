# Generalization of Semantic Segmentation Models Under Domain Shift

## 1. Motivation
Semantic segmentation models often achieve strong performance when evaluated on data drawn from the same distribution as their training set. However, in real-world applications, models are frequently deployed in environments that differ from the training domain, leading to performance degradation.

This project investigates the **generalization ability of semantic segmentation models under domain shift**, with a focus on **low-compute experimental settings**. The goal is not to achieve state-of-the-art performance, but to analyze how architectural choices and training strategies affect robustness when models are evaluated on a different dataset than the one used for training.

---

## 2. Datasets

### Training Dataset
**CamVid**
- Urban driving scenes
- ~700 images
- Pixel-level semantic annotations
- Used for training and in-domain validation

### Evaluation Dataset (Domain Shift)
**Cityscapes (validation subset)**
- High-resolution urban street scenes
- Different geographic locations, camera setups, and annotation styles
- Used *only for evaluation*, without any fine-tuning

To reduce complexity and training time, classes were remapped to a reduced set of common semantic categories (e.g., road, building, vehicle, pedestrian, sky, vegetation).

---

## 3. Models
The following segmentation architectures were evaluated:

- **U-Net with ResNet-34 encoder**
- **DeepLabV3+ with ResNet-50 backbone**
- **SegFormer-B0 (lightweight transformer-based model)**

All encoders were initialized with ImageNet pretrained weights. Training from scratch was not performed.

---

## 4. Experimental Setup

### Input Resolution
All images were resized to **512 × 256** to reduce computational cost.

### Training Details
- Framework: PyTorch
- Optimizer: Adam
- Learning rate:
  - Decoder: 1e-3
  - Encoder (when unfrozen): 1e-4
- Batch size: 2–4
- Epochs: 20–30
- Loss function: Cross-entropy loss
- Early stopping based on validation mIoU

For selected experiments, the encoder was either:
- Fully frozen
- Partially unfrozen during later epochs

---

## 5. Experiments

### Experiment A — In-Domain Performance
Models were trained on CamVid and evaluated on the CamVid validation split to establish baseline performance.

### Experiment B — Cross-Domain Generalization
Without any retraining or fine-tuning, the trained models were evaluated on the Cityscapes validation subset to assess performance under domain shift.

### Experiment C — Effect of Encoder Freezing
For the U-Net model, generalization performance was compared between:
- Fully frozen encoder
- Partially fine-tuned encoder

---

## 6. Results

### Quantitative Results
Evaluation metrics:
- Mean Intersection over Union (mIoU)
- Pixel Accuracy

Results show a consistent performance drop when models are evaluated on Cityscapes, highlighting the impact of domain shift. Lightweight transformer-based models exhibit slightly improved robustness compared to CNN-based architectures.

*(Detailed tables are provided in `results/quantitative/`.)*

---

## 7. Qualitative Analysis
Qualitative results reveal common failure modes under domain shift, including:
- Misclassification of visually similar classes
- Reduced boundary precision
- Sensitivity to differences in lighting and camera perspective

Representative examples are provided in `results/qualitative/`.

---

## 8. Compute Constraints
All experiments were conducted on a **CPU-based environment (Mac Intel)** without access to GPU acceleration. As a result:
- Training was performed using reduced image resolution
- Pretrained encoders were used
- Dataset sizes and number of epochs were limited

Despite these constraints, the experiments provide meaningful insights into relative model behavior and generalization trends.

---

## 9. Limitations and Future Work
This study is limited by dataset size, compute constraints, and the use of reduced-resolution inputs. Future work could include:
- Training on larger datasets
- Full fine-tuning with GPU resources
- Exploring domain adaptation techniques
- Extending the study to additional transformer-based architectures

---

## 10. Conclusion
This project demonstrates that domain shift significantly affects semantic segmentation performance, even for modern architectures. While pretrained models mitigate some of the performance loss, robustness remains a challenge, particularly under limited compute settings. These findings emphasize the importance of evaluation beyond in-domain benchmarks.
