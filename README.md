# ImagesHRM: A Faithful Hierarchical Reasoning Model (HRM) on Small Image Benchmarks

**Does HRM work for natural-image classification without augmentation?**  
This repo contains a faithful JAX/Flax implementation of the **Hierarchical Reasoning Model (HRM)**—two Transformer-style modules \(f_L,f_H\) trained with a **one-step (DEQ-style)** gradient and **deep supervision**—and **convolutional baselines** for MNIST, CIFAR-10, and CIFAR-100. Experiments are run in a deliberately *raw* regime (no augmentation) to isolate architectural inductive bias.

> TL;DR: HRM optimizes stably and is competitive on MNIST, but **overfits and underperforms** on CIFAR-10/100 relative to a small CNN that trains ~30× faster per epoch.

---

## Table of Contents
- [Results (Summary)](#results-summary)
- [Environment](#environment)
- [Datasets](#datasets)
- [Training & Evaluation](#training--evaluation)
  - [HRM on CIFAR-10](#hrm-on-cifar10)
  - [CNN on CIFAR-10](#cnn-on-cifar10)
  - [HRM on CIFAR-100](#hrm-on-cifar100)
  - [CNN on CIFAR-100](#cnn-on-cifar100)
  - [MNIST (HRM)](#mnist-hrm)
- [Reproducing Figures](#reproducing-figures)
- [Project Structure](#project-structure)
- [Cite](#cite)
- [License](#license)

---

## Results (Summary)

- **MNIST (HRM)**: ~**98.0%** test accuracy.  
- **CIFAR-10**: HRM **65.0%** vs **77.2%** (CNN); HRM ~**5:53/epoch** vs CNN ~**12 s/epoch**.  
- **CIFAR-100**: HRM **29.7%** vs **45.3%** (CNN) with HRM train acc ~**91.5%** vs CNN train acc ~**50.5%**.

Loss traces show stable optimization for both models, but CNN generalizes substantially better on small natural images under the raw regime.

---

## Environment

- **Language/Frameworks**: Python 3.10+, JAX, Flax, Optax, NumPy, Matplotlib, scikit-learn, tqdm.


### Datasets

Code will auto-download/verify files for MNIST and CIFAR-10/100 into ./data/.
Please cite datasets in papers using the canonical references (MNIST, CIFAR tech report).