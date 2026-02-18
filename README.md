<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&duration=4000&pause=1000&color=A855F7&center=true&vCenter=true&multiline=true&repeat=true&width=900&height=100&lines=%F0%9F%91%97+Deep+Virtual+Try-On+with+Clothes+Transform;Context-Aware+GANs+%7C+TPS+Warping+%7C+Multi-Model+Architecture" alt="Typing SVG" />
</p>

<p align="center">
  <em>🔬 A production-grade Deep Learning pipeline that virtually dresses people in new garments using Generative Adversarial Networks, Thin-Plate Spline geometric warping, and cascaded refinement — built for real-world e-commerce fashion AI.</em>
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch_2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://en.wikipedia.org/wiki/Generative_adversarial_network"><img src="https://img.shields.io/badge/Generative_AI-GANs-A855F7?style=for-the-badge&logo=openai&logoColor=white" alt="GANs"/></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/Computer_Vision-SOTA-10B981?style=for-the-badge&logo=opencv&logoColor=white" alt="CV"/></a>
  <a href="https://kornia.readthedocs.io/"><img src="https://img.shields.io/badge/Kornia-Geometric_DL-F59E0B?style=for-the-badge" alt="Kornia"/></a>
  <a href="https://torchmetrics.readthedocs.io/"><img src="https://img.shields.io/badge/TorchMetrics-Evaluation-06B6D4?style=for-the-badge" alt="TorchMetrics"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA_11.8-GPU_Accelerated-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/torchvision-VGG16_%7C_DeepLabV3-EE4C2C?style=flat-square" alt="torchvision"/>
  <img src="https://img.shields.io/badge/MediaPipe-Pose_Estimation-4285F4?style=flat-square&logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/SciPy-Scientific_Computing-8CAAE6?style=flat-square&logo=scipy&logoColor=white" alt="SciPy"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

---

## 👩‍💻 Author

<table>
<tr>
<td width="120" align="center">
  <a href="https://github.com/bhavanareddy19">
    <img src="https://github.com/bhavanareddy19.png" width="100" height="100" style="border-radius:50%;" alt="Bhavana Vippala"/>
  </a>
</td>
<td>

**Bhavana Vippala**  
*AI Engineer · Data Scientist · ML Researcher*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bhavanareddy19)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/bhavanareddy19)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-A855F7?style=flat-square&logo=vercel&logoColor=white)](https://data-girl-s-portfolio.vercel.app/)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:Bhavana.Vippala@colorado.edu)

</td>
</tr>
</table>

---

## 🎯 Executive Summary

> **Problem:** Traditional e-commerce requires expensive photo shoots for every garment × model combination.  
> **Solution:** This project synthesizes photorealistic images of *any person wearing any garment* using deep generative models — reducing cost by orders of magnitude while enabling personalized shopping experiences.

This system implements a **complete ML lifecycle** — from raw data ingestion and preprocessing through model training, evaluation, ablation studies, and inference — comparing **4 state-of-the-art GAN architectures** with rigorous quantitative benchmarking.

---

## 🏗️ System Architecture

<p align="center">
  <img src="Virtual%20Try%20on%20Architecture.png" alt="System Architecture" width="900"/>
</p>

---

## 🧠 Model Architectures Deep Dive

<table>
<tr>
<th width="25%">🟣 PRGAN</th>
<th width="25%">🔵 CAGAN</th>
<th width="25%">🟢 CRN</th>
<th width="25%">🟠 VITON</th>
</tr>
<tr>
<td>

**Pose-Guided Residual GAN**

- UNet Generator
- PatchGAN Discriminator
- Residual connections
- Single-stage synthesis

</td>
<td>

**Context-Aware GAN**

- Context-conditioned UNet
- Conditional discriminator
- Agnostic + garment fusion
- End-to-end adversarial

</td>
<td>

**Cascaded Refinement Network**

- 3-stage progressive:
  - `64×64` → `128×128` → `256×256`
- Coarse-to-fine synthesis
- Multi-resolution fusion

</td>
<td>

**Virtual Try-On Network**

- 🌀 **TPS Warping** (Stage 0)
- 🎨 **Coarse UNet** (Stage 1)
- ✨ **Refine UNet** (Stage 2)
- Geometric deformation aware

</td>
</tr>
</table>

---

## 🛠️ Tech Stack & Skills Demonstrated

### 🤖 **AI/ML Engineering** `Core Competency`

| Technology | Application in Project | Proficiency |
|:--|:--|:--|
| **`PyTorch 2.2+`** | End-to-end deep learning framework — model definition, autograd, DataLoader, distributed training | ⭐⭐⭐⭐⭐ |
| **`torchvision`** | VGG-16 feature extraction (perceptual loss), DeepLabV3-ResNet50 (semantic segmentation), image transforms | ⭐⭐⭐⭐⭐ |
| **`Kornia`** | Differentiable geometric vision — Thin-Plate Spline warping, spatial transformations | ⭐⭐⭐⭐ |
| **`torchmetrics`** | SSIM, Inception Score, Jaccard Index (IoU) — rigorous model benchmarking | ⭐⭐⭐⭐ |
| **`einops`** | Einstein notation for tensor operations — clean, readable reshaping | ⭐⭐⭐⭐ |
| **`MediaPipe`** | Real-time pose estimation — 33 landmark detection, OpenPose keypoint mapping | ⭐⭐⭐⭐ |
| **`GANs`** | Adversarial training with hinge loss, PatchGAN discriminators, generator-discriminator interplay | ⭐⭐⭐⭐⭐ |

### 📊 **Data Science & Analytics** `Quantitative Rigor`

| Skill | Implementation Details |
|:--|:--|
| **Experiment Design** | A/B comparison of 4 architectures under identical conditions; YAML-based hyperparameter configs |
| **Statistical Metrics** | SSIM (structural fidelity), Inception Score (generation quality), IoU (alignment accuracy) |
| **Ablation Studies** | Systematic component knockout — disabling refinement stage and/or TPS warper to isolate contributions |
| **Data Visualization** | Comparison grids, training loss curves, per-model qualitative outputs via `matplotlib` + `torchvision.utils` |
| **EDA Pipeline** | Image pair analysis, distribution checks on mask coverage, garment-person alignment statistics |

### ⚙️ **Data Engineering & MLOps** `Production Practices`

| Practice | Implementation |
|:--|:--|
| **Data Pipeline** | Custom `VITONPairSet` PyTorch Dataset with lazy loading, on-the-fly augmentation, multi-format support |
| **Config Management** | Centralized `config.yaml` — learning rates, epochs, batch size, device, per-model hyperparams |
| **Environment Reproducibility** | Conda `env.yml` with pinned versions (Python 3.10, PyTorch ≥2.2, CUDA 11.8) |
| **Modular Architecture** | Clean separation: `models/` · `data/` · `utils/` · `scripts/` — plug-and-play model swapping |
| **Checkpoint Management** | Epoch-level model state persistence with `torch.save` / `torch.load` (e.g., `viton_020.pth`) |
| **GPU Optimization** | `cudnn.benchmark=True`, mixed-precision ready, efficient `pin_memory` DataLoaders |
| **CLI Interface** | Unified `run.py` CLI — `test`, `demo`, `train`, `evaluate` commands with argument parsing |

### 🧬 **Computer Vision** `Domain Expertise`

| Technique | Details |
|:--|:--|
| **Semantic Segmentation** | DeepLabV3 (ResNet-50 backbone) for human body parsing — COCO person class extraction |
| **Pose Estimation** | MediaPipe 33-landmark → OpenPose 18-keypoint mapping; Gaussian heatmap generation |
| **Geometric Warping** | Thin-Plate Spline (TPS) with RBF interpolation; Affine transform fallback |
| **Image-to-Image Translation** | Conditional generation: agnostic person + garment → dressed person |
| **Feature Extraction** | VGG-16 `relu_2_2` features for perceptual similarity computation |
| **Agnostic Representation** | Body region masking using segmentation maps + clothing region priors |

---

## 📸 Results & Capabilities

| Capability | Status | Description |
|:--|:--:|:--|
| 🧥 **Garment Warping** | ✅ | TPS-based non-rigid deformation to match body contour |
| 🎨 **Texture Preservation** | ✅ | Logos, patterns, and fabric details maintained through perceptual loss |
| 🖐️ **Skin Reconstruction** | ✅ | Generates realistic skin in previously occluded body regions |
| 🕺 **Multi-Pose Support** | ✅ | Skeleton-aware synthesis handles diverse body poses |
| 🔄 **Multi-Model Comparison** | ✅ | Side-by-side evaluation of PRGAN, CAGAN, CRN, VITON |
| 📏 **Multi-Scale Synthesis** | ✅ | CRN cascaded refinement: 64→128→256 progressive generation |

<p align="center">
  <img src="output/comparison_grid.png" alt="Comparison Grid" width="700"/>
  <br/>
  <em>Generated comparison: Input Person → Garment → Agnostic Representation → Final Try-On Result</em>
</p>

---

## 📂 Project Structure

```
Virtual-Try-On/
│
├── 📁 models/                        # 🧠 Neural Network Definitions
│   ├── __init__.py                   # Model registry & exports
│   ├── _base.py                      # UNet Generator, ResBlk, Conv blocks
│   ├── prgan.py                      # Pose-Guided Residual GAN
│   ├── cagan.py                      # Context-Aware GAN
│   ├── crn.py                        # Cascaded Refinement Network (3-stage)
│   ├── viton.py                      # VITON orchestrator (Coarse → Refine)
│   ├── viton_coarse.py               # Coarse stage with TPS warping
│   ├── viton_refine.py               # Refinement stage UNet
│   └── warper_tps.py                 # Thin-Plate Spline & Affine warpers
│
├── 📁 data/                          # 💾 Data Pipeline & ETL
│   ├── __init__.py
│   ├── dataset.py                    # VITONPairSet & VITONInferenceSet
│   ├── train/                        # Training split
│   │   ├── image/                    #   Person images
│   │   ├── cloth/                    #   Garment images
│   │   ├── agnostic-v3.2/            #   Agnostic representations
│   │   └── pairs.txt                 #   Person-garment mappings
│   └── test/                         # Test split (same structure)
│
├── 📁 utils/                         # 🔧 Utilities & Metrics
│   ├── __init__.py                   # Unified utility exports
│   ├── losses.py                     # GANLoss, VGG Perceptual, L1/L2, TV
│   ├── metrics.py                    # SSIM, Inception Score, IoU
│   ├── pose.py                       # MediaPipe pose → heatmaps
│   ├── seg.py                        # DeepLabV3 segmentation pipeline
│   └── vis.py                        # Grid visualization with torchvision
│
├── 📁 scripts/                       # 🔄 ML Workflow Scripts
│   ├── __init__.py
│   ├── train.py                      # Training loop with AdamW + Perceptual loss
│   ├── evaluate.py                   # Test-set evaluation with metrics
│   └── ablate.py                     # Ablation studies (disable refine/warper)
│
├── 📁 output/                        # 📤 Generated Results
├── 📁 Images/                        # 📸 Architecture & Report Figures
├── 📁 Research Paper/                # 📜 Related Literature
│
├── config.yaml                       # ⚙️ Hyperparameters & experiment config
├── env.yml                           # 🐍 Conda environment spec
├── demo.py                           # 🎮 Interactive inference & training demo
├── run.py                            # 🚀 Unified CLI entry point
└── LICENSE                           # 📄 MIT License
```

---

## 🚀 Quick Start

### Prerequisites

```
✅ Python 3.10+        ✅ CUDA 11.8 (GPU recommended)        ✅ Conda (recommended)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/bhavanareddy19/Virtual-Try-On.git
cd Virtual-Try-On

# Create environment (recommended)
conda env create -f env.yml
conda activate viton

# OR install via pip
pip install torch torchvision kornia torchmetrics einops scikit-image scipy matplotlib tqdm pyyaml pillow
```

### Usage

```bash
# ✅ Test all models load correctly
python run.py test

# 🎮 Run inference demo
python run.py demo prgan          # PRGAN model
python run.py demo viton          # VITON with TPS warping

# 🏋️ Train a model
python run.py train prgan          # Train PRGAN
python run.py train viton          # Train VITON (dual learning rates)

# 📊 Evaluate on test set
python run.py evaluate prgan checkpoints/prgan_020.pth

# 🔬 Ablation study
python scripts/ablate.py --disable_refine    # Without refinement stage
python scripts/ablate.py --disable_warper    # Without TPS warping

# 🎨 Custom image inference
python demo.py --model viton --person my_photo.jpg --cloth my_shirt.jpg
```

---

## ⚡ Engineering Workflow

```mermaid
flowchart LR
    A["📥 Data Ingestion<br/><code>data/dataset.py</code><br/>VITONPairSet"]
    B["🏋️ Training Loop<br/><code>scripts/train.py</code><br/>AdamW + Losses"]
    C["📊 Validation<br/><code>scripts/evaluate.py</code><br/>SSIM · IS · IoU"]
    D["🔬 Ablation<br/><code>scripts/ablate.py</code><br/>Component Analysis"]
    E["🎮 Inference<br/><code>demo.py</code><br/>Real-time Try-On"]

    A --> B --> C --> D --> E

    style A fill:#1e1b4b,stroke:#818cf8,color:#e0e7ff
    style B fill:#1e1b4b,stroke:#a855f7,color:#f3e8ff
    style C fill:#1e1b4b,stroke:#10b981,color:#d1fae5
    style D fill:#1e1b4b,stroke:#f59e0b,color:#fef3c7
    style E fill:#1e1b4b,stroke:#06b6d4,color:#cffafe
```

---

## 🏆 Key Technical Highlights

<table>
<tr>
<td width="50%">

### 🌀 Thin-Plate Spline Warping
- **9 control points** (3×3 grid) with learned offsets
- RBF Gaussian interpolation (`σ=0.5`)
- Kornia-accelerated with custom fallback
- Offset clamping via `tanh × 0.3` for stability

</td>
<td width="50%">

### 🎯 Multi-Loss Optimization
- **L1 Reconstruction** — pixel-level fidelity
- **VGG-16 Perceptual** — `relu_2_2` feature matching
- **Hinge Adversarial** — PatchGAN stability
- **Total Variation** — smoothness regularization
- **SSIM** — structural similarity awareness

</td>
</tr>
<tr>
<td>

### 🦴 Skeleton-Aware Processing
- MediaPipe 33→18 keypoint mapping (OpenPose format)
- Gaussian heatmap generation (`σ=7`)
- Pose-conditioned agnostic representation
- Fallback to heuristic body priors

</td>
<td>

### 🎭 Semantic Body Parsing
- DeepLabV3 (ResNet-50) for COCO segmentation
- Person class extraction (class 15)
- Clothing region isolation via spatial priors
- Morphological cleanup (fill → dilate → erode)

</td>
</tr>
</table>

---

## 📦 Dependencies & Environment

```yaml
# Core ML Stack
pytorch >= 2.2          # Deep learning framework
torchvision             # Pretrained models (VGG-16, DeepLabV3)
kornia                  # Differentiable geometric vision
torchmetrics            # SSIM, Inception Score, IoU
einops                  # Einstein tensor operations

# Scientific Computing
scipy                   # Morphological ops, ndimage filters
scikit-image            # Advanced image processing
numpy                   # Numerical computing
pillow                  # Image I/O

# Visualization & Tooling
matplotlib              # Training curves, analysis plots
tqdm                    # Progress bars
pyyaml                  # Configuration management
cudatoolkit = 11.8      # GPU acceleration
```

---

## 🎓 Relevant For These Roles

<table>
<tr>
<td align="center" width="25%">
<h3>🤖 AI Engineer</h3>

`GANs` `PyTorch`  
`Computer Vision`  
`TPS Warping`  
`Pose Estimation`  
`Segmentation`

</td>
<td align="center" width="25%">
<h3>📊 Data Scientist</h3>

`Experiment Design`  
`Ablation Studies`  
`SSIM / IS / IoU`  
`Model Comparison`  
`Statistical Analysis`

</td>
<td align="center" width="25%">
<h3>⚙️ Data Engineer</h3>

`ETL Pipelines`  
`Data Loaders`  
`Config Management`  
`Environment Repro.`  
`CLI Tooling`

</td>
<td align="center" width="25%">
<h3>🔬 Data Analyst</h3>

`Metric Analysis`  
`Visualization`  
`A/B Testing`  
`Report Generation`  
`Performance Tracking`

</td>
</tr>
</table>

---

## 📜 License

This project is open-source under the [MIT License](LICENSE) and available for educational and portfolio purposes.

---

<p align="center">
  <strong>Built with ❤️ by <a href="https://github.com/bhavanareddy19">Bhavana Vippala</a></strong><br/>
  <sub>University of Colorado Boulder · MS in Data Science</sub><br/><br/>
  <a href="https://www.linkedin.com/in/bhavanareddy19"><img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
  <a href="https://data-girl-s-portfolio.vercel.app/"><img src="https://img.shields.io/badge/Portfolio-Visit-A855F7?style=for-the-badge&logo=vercel&logoColor=white" alt="Portfolio"/></a>
  <a href="mailto:Bhavana.Vippala@colorado.edu"><img src="https://img.shields.io/badge/Email-Reach_Out-EA4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/></a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,15,20,24&height=80&section=footer" width="100%"/>
</p>
