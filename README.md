# Generalization of Semantic Segmentation Models Under Domain Shift

## Overview
Semantic segmentation models often report strong performance when evaluated on data drawn from the same distribution as their training set. However, real-world deployment typically involves **domain shift**, such as changes in geographic location, camera hardware, lighting conditions, or annotation styles, which can significantly degrade performance.

This project investigates the **generalization behavior of semantic segmentation models under domain shift**, with a particular focus on **low-compute experimental settings**. Rather than optimizing for state-of-the-art accuracy, the objective is to analyze how **model architecture and training dynamics** influence robustness when models are evaluated on unseen datasets.

---

## 1. Motivation
Most semantic segmentation benchmarks emphasize in-domain evaluation, providing limited insight into how models behave under distributional changes. This work aims to:
- Quantify the performance gap between in-domain and cross-domain evaluation
- Compare CNN-based and transformer-based architectures under identical constraints
- Analyze generalization behavior without relying on large-scale pretraining

All experiments are intentionally designed to reflect **realistic compute limitations**.

---

## 2. Datasets

### Training Dataset
**CamVid**
- Urban driving scenes
- ~700 images
- Pixel-level semantic annotations
- Used for training and in-domain validation

### Evaluation Datasets (Domain Shift)
- **Cityscapes (validation subset)**
- **Mapillary Vistas 2.0 (validation subset)**

Both evaluation datasets differ from CamVid in terms of:
- Geographic locations
- Camera hardware and viewpoints
- Scene composition
- Annotation conventions

### Class Remapping
All three datasets (CamVid, Cityscapes, and Mapillary Vistas) were **remapped to the Cityscapes 19-class label set** to ensure label consistency and enable direct cross-dataset evaluation.

---

## 3. Models
The following semantic segmentation architectures were evaluated:

- **U-Net**
- **DeepLabV3+**
- **SegFormer-B0**

### Training From Scratch
Models were trained from scratch to isolate the effects of domain shift without introducing biases from large-scale pretraining datasets such as ImageNet. This design choice ensures that any observed cross-domain generalization arises from the model architecture and training dynamics rather than prior exposure to external visual domains. Despite the absence of pretraining, models converged reliably within a limited number of epochs under the imposed compute constraints.

No pretrained weights or external datasets were used.

---

## 4. Experimental Setup

### Input Resolution
All images were resized to **512 × 512**.

### Training Configuration
- Framework: PyTorch
- Optimizer: Adam
- Learning rate: `1e-3`
- Batch size: **4**
- Epochs: **30**
- Loss function: Cross-entropy loss
- Training schedule: Fixed number of epochs (no early stopping)

---

## 5. Experiments

### Experiment A — In-Domain Evaluation
Models were trained and evaluated on CamVid to establish baseline segmentation performance.

### Experiment B — Cross-Domain Generalization
Trained models were evaluated on Cityscapes and Mapillary Vistas validation subsets **without any retraining or fine-tuning**, in order to assess performance degradation under domain shift.

---

## 6. Results

### Quantitative Evaluation
Evaluation metrics:
- Mean Intersection over Union (mIoU)
- Pixel Accuracy

All models exhibit a consistent performance drop when evaluated on out-of-domain datasets. Transformer-based architectures demonstrate slightly improved robustness compared to CNN-based models, though a substantial generalization gap remains.

Detailed quantitative results are provided in: `results/quantitative/`

---

## 7. Qualitative Analysis
Qualitative inspection reveals common failure modes under domain shift, including:
- Confusion between visually similar semantic classes
- Reduced boundary precision
- Sensitivity to changes in lighting, camera perspective, and scene layout

Representative visual results are available in: `results/qualitative/`

---

## 8. Compute Constraints
All experiments were conducted on a **Mac Intel** system using **MPS acceleration** with an **AMD Radeon Pro 560X GPU**.

As a result:
- Training was performed at a fixed input resolution
- Model capacity and training duration were constrained
- Architectural choices prioritized computational feasibility

Despite these constraints, experiments produced stable training behavior and consistent cross-domain trends.

---

## 9. Limitations and Future Work
This study is limited by dataset size, compute resources, and the use of reduced-resolution inputs. Potential directions for future work include:
- Training and evaluation at higher resolutions
- Scaling experiments with more powerful GPU hardware
- Incorporating domain adaptation or test-time adaptation techniques
- Extending the comparison to larger transformer-based architectures

---

## 10. Conclusion
This project demonstrates that **domain shift remains a significant challenge for semantic segmentation**, even when using modern architectures. Training from scratch enables a controlled analysis of generalization behavior, revealing that architectural differences alone are insufficient to fully bridge the cross-domain performance gap under constrained settings. These findings highlight the importance of evaluating segmentation models beyond standard in-domain benchmarks, particularly for real-world deployment scenarios.
