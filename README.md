# Face Frontalization using GANs – EE655 Course Project

This repository contains the codebase and documentation for Group 26's final project in **EE655: Computer Vision & Deep Learning**, titled:

> **Frontalization of Profile Face Images Using a Generative Adversarial Network**

## Overview

Face frontalization refers to generating a front-facing image of a person given a profile view. Our project formulates this as a conditional image-to-image translation task using a **U-Net-based Generator** and **PatchGAN Discriminator**. We train a conditional GAN (cGAN) from scratch without relying on 3D priors or identity labels. The model achieves realistic and identity-preserving frontal views, even under extreme head poses.

## Repository Contents

- 📄 [Project Writeup (PDF)](https://github.com/Face-Frontalization/Face-Frontalization/blob/main/Face-Frontalization-Writeup.pdf)  
- 🎞️ [Project Presentation (PPTX)](https://github.com/Face-Frontalization/Face-Frontalization/blob/main/Face_Frontalization_Presentation.pptx)  
- 💻 [Model Code](https://github.com/Face-Frontalization/Face-Frontalization/blob/main/Model.txt)

## Dataset

We utilize the [300W-LP Dataset](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?resourcekey=0-WT5tO4TOCbNZY6r6z6WmOA), a large-scale collection of synthesized face images with a variety of poses derived from the Multi-PIE dataset.

- **Images per subject:** 1 frontal + 23 profile variants
- **Preprocessing:** Resize to `128×128`, normalize to `[-1, 1]`
- **Augmentation:** Horizontal flips for pose diversity
- **Split:** 80% training / 20% testing

## Model Architecture

### Generator (U-Net)
- **Input:** 128×128×3 RGB profile face
- **Encoder:** Convolutional layers with downsampling
- **Decoder:** Transposed convolutions with skip connections
- **Output:** 128×128×3 frontal face
- **Activation:** Tanh

### Discriminator (PatchGAN)
- Classifies N×N patches instead of full images
- Takes concatenated (profile, frontal) pairs as input
- Outputs patch-wise real/fake predictions

### Loss Functions
We use a combined loss:
``` L_total = L_adv + 100 × L_L1 ```

- **Adversarial Loss (Binary Crossentropy):** Encourages realism
- **L1 Loss (Mean Absolute Error):** Ensures pixel-level accuracy

## Training Details

| Parameter       | Value          |
|----------------|----------------|
| Optimizer      | Adam           |
| Learning Rate  | 2 × 10⁻⁴       |
| Beta₁          | 0.5            |
| Batch Size     | 32             |
| Epochs         | 10             |
| Framework      | TensorFlow (Keras) |
| Hardware       | NVIDIA RTX GPU (12GB VRAM) |

## Sample Output

Example of frontalization from profile input:

```python
frontalized_img = frontalize_image("path/to/profile.jpg")

plt.imshow(frontalized_img)
plt.title("Frontalized Output")
plt.axis("off")

