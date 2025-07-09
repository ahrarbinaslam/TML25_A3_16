# # Robustness (TML Assignment 3 – Team 16)

This repository contains our implementation of a robust image classification model developed for **Assignment 3** of the **Trustworthy Machine Learning** course (Summer Semester 2025) at Saarland University.

---

## Assignment Goal

The goal of this assignment was to train an image classification model that performs well on both clean inputs and adversarial examples. The adversarial examples are crafted using:

- **Fast Gradient Sign Method (FGSM)**
- **Projected Gradient Descent (PGD)**

These perturbations are small and visually imperceptible but can cause machine learning models to make incorrect predictions. The task required us to build a model that is **robust** against such attacks while maintaining **high clean accuracy**.

---

## Our Approach

### Dataset

- Provided as a serialized PyTorch `.pt` file using a custom `TaskDataset` class.
- RGB images labeled from 0 to 9.
- All images resized to `32×32` and converted into tensors using standard PyTorch transforms.
- No additional data augmentation (flipping, cropping) was applied to keep adversarial perturbations meaningful and interpretable.

### Model

- Architecture: `ResNet34`
- Output Layer: Modified to output 10 logits (one for each class)
- We use Resnet34 because:
  - Better generalization than ResNet18. Moreover, it is less computationally demanding than ResNet50.
### Training Setup

- Optimizer**: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 20
- Loss Function: Cross-entropy loss averaged over:
  - Clean inputs
  - FGSM adversarial inputs
  - PGD adversarial inputs

### Adversarial Training

- **FGSM**:
  - One-step attack
  - Epsilon = 0.03

- **PGD**:
  - 3 iterative steps
  - Step size (α) = 0.01
  - Epsilon = 0.03
  - Pixel values clipped after each step

We chose this configuration to balance attack strength and training efficiency within Google Colab’s resource constraints.

---

## Evaluation

- We did not use a separate validation set, instead, we monitored clean and adversarial accuracy on the training data.
- Training was stable across all 20 epochs without using a learning rate scheduler.

### Final Results

| Metric           | Score     |
|------------------|-----------|
| Clean Accuracy   | 52.1%     |
| FGSM Accuracy    | 37.1%     |
| PGD Accuracy     | 36.8%     |

