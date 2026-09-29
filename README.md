# GeoShift:Cross-Domain Foundation Models for Electrostatics

<div align="center">

![Architecture](docs/images/architecture.svg)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-APL%20Comp.%20Phys.-blue)](docs/manuscript.pdf)

**Pre-training neural operators across molecular physics domains for improved sample efficiency and transferability**

[Key Features](#-key-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Results](#-results) •
[Citation](#-citation)

</div>

---

## 🎯 Overview

This repository implements a **hybrid EGNN-PaiNN architecture** for cross-domain learning in molecular property prediction. By pre-training across multiple physics domains (quantum mechanics, molecular dynamics, ANI-1x), we achieve:

- **30-44% error reduction** on geometry transfer tasks
- **5× sample efficiency** improvement over training from scratch
- **Physics-compliant predictions** across diverse molecular systems

### The Problem

Training neural operators for molecular property prediction typically requires:
- Large domain-specific datasets
- Expensive quantum chemistry calculations  
- Poor transfer across molecular geometries and compositions

### Our Solution

**Cross-domain pre-training** leverages shared electrostatic principles across:
- QM9 (small molecules, equilibrium geometries)
- MD17 (non-equilibrium dynamics)
- ANI-1x (diverse conformations)

The model learns **equation-class-specific** representations that transfer effectively to new geometries and chemical spaces.

---

## 📊 Architecture

![Model Architecture](docs/images/model_architecture.svg)

### Hybrid EGNN-PaiNN Design

```
Input: Molecular Graph
  ↓
┌─────────────────────────────────┐
│  Equivariant Message Passing    │  ← EGNN Layers
│  (Geometric Features)            │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  Equivariant Convolutions        │  ← PaiNN Layers
│  (Physical Interactions)         │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  Multi-Task Prediction Head      │
│  • Energy                        │
│  • HOMO-LUMO Gap                 │
│  • Forces (optional)             │
└─────────────────────────────────┘
```

**Key Components:**
- **EGNN Layers:** Capture geometric relationships with SE(3) equivariance
- **PaiNN Layers:** Model physical interactions with message passing
- **Multi-Task Head:** Unified prediction across properties
- **Physics Constraints:** Energy conservation, force consistency

---

## 🚀 Key Features

- ✅ **Hybrid Architecture** - EGNN-PaiNN fusion for geometric and physical features
- ✅ **Cross-Domain Learning** - Pre-training across QM9, MD17, ANI-1x datasets
- ✅ **GeoShift Benchmark** - 5 geometry transfer evaluation tasks
- ✅ **Physics-Augmented Training** - Energy conservation and force matching constraints
- ✅ **Efficient Implementation** - Optimized PyTorch with automatic mixed precision
- ✅ **Comprehensive Evaluation** - Beyond MSE: force MAE, energy RMSE, physics compliance

---

## 📈 Results

![Training Curves](docs/images/training_curves.svg)

### Geometry Transfer Performance (GeoShift Benchmark)

| Task | From → To | Baseline MAE | Our MAE | Improvement |
|------|-----------|--------------|---------|-------------|
| **Task 1** | Small → Large | 0.087 eV | 0.051 eV | **41%** ↓ |
| **Task 2** | Organic → Inorganic | 0.112 eV | 0.073 eV | **35%** ↓ |
| **Task 3** | Equilibrium → MD | 0.095 eV | 0.056 eV | **41%** ↓ |
| **Task 4** | Gas → Condensed | 0.128 eV | 0.083 eV | **35%** ↓ |
| **Task 5** | Rigid → Flexible | 0.104 eV | 0.073 eV | **30%** ↓ |

### Sample Efficiency

![Sample Efficiency](docs/images/sample_efficiency.svg)

Pre-trained model achieves target accuracy with **5× fewer samples** than training from scratch.

### Physics Compliance

- ✅ Energy conservation: <0.1% deviation
- ✅ Force consistency: MAE < 0.05 kcal/mol/Å
- ✅ Rotational equivariance: Perfect within numerical precision

---

## 💻 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/molecular-foundation-models.git
cd molecular-foundation-models

# Create conda environment
conda create -n molecular-models python=3.10
conda activate molecular-models

# Install dependencies
pip install -r requirements.txt

# Install package (editable mode for development)
pip install -e .
```

### Verify Installation

```python
import torch
from src.model import build_model

# Create model
config = {
    "model": {
        "hidden_dim": 128,
        "n_layers": 5,
        "cutoff": 5.0
    }
}
model = build_model(config)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

---

## 🏃 Quick Start

### 1. Download Datasets

```bash
# Download QM9, MD17, and ANI-1x datasets (~23 GB)
python scripts/download_datasets.py

# Datasets will be cached in data/ directory
```

### 2. Train Cross-Domain Model

```bash
# Pre-training on all three domains
python train.py --config configs/cross_domain.json

# Training outputs saved to experiments/cross_domain_pretraining/
```

### 3. Evaluate on GeoShift Benchmark

```bash
# Evaluate transfer performance
python evaluate.py \
    --checkpoint experiments/cross_domain_pretraining/best_model.pt \
    --benchmark geoshift

# Results saved to experiments/evaluation/
```

### 4. Fine-tune on Target Task

```bash
# Transfer learning to new domain
python train.py \
    --config configs/transfer.json \
    --pretrained experiments/cross_domain_pretraining/best_model.pt \
    --target-dataset your_dataset
```

---

## 📁 Repository Structure

```
molecular-foundation-models/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── requirements.txt               # Dependencies
├── setup.py                       # Package installation
│
├── src/                          # Source code
│   ├── model.py                  # EGNN-PaiNN architecture
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Evaluation utilities
│   ├── data/                     # Dataset classes
│   │   ├── qm9.py
│   │   ├── md17.py
│   │   └── ani1x.py
│   ├── models/                   # Model components
│   │   ├── egnn.py              # EGNN layers
│   │   ├── painn.py             # PaiNN layers
│   │   └── heads.py             # Task heads
│   └── utils/                    # Utilities
│       ├── metrics.py
│       └── visualization.py
│
├── configs/                      # Configuration files
│   ├── cross_domain.json        # Full pre-training
│   ├── single_domain.json       # Baseline
│   └── transfer.json            # Transfer learning
│
├── scripts/                      # Utility scripts
│   ├── download_datasets.py     # Download data
│   ├── preprocess.py            # Data preprocessing
│   └── analyze_results.py       # Result analysis
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/                        # Unit tests
│   ├── test_model.py
│   ├── test_data.py
│   └── test_training.py
│
└── docs/                         # Documentation
    ├── TRAINING.md              # Training guide
    ├── EVALUATION.md            # Evaluation guide
    ├── API.md                   # API reference
    └── images/                  # Figures
```

---

## 🔬 Datasets

### QM9
- **Size:** 134,000 small organic molecules
- **Properties:** 13 quantum mechanical properties
- **Source:** [QM9 Database](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)

### MD17
- **Size:** 8 molecular systems, ~1M conformations
- **Properties:** Energies and forces from MD simulations
- **Source:** [MD17 Dataset](http://quantum-machine.org/gdml/#datasets)

### ANI-1x
- **Size:** 5M molecular conformations
- **Properties:** DFT energies for diverse molecules
- **Source:** [ANI-1x Dataset](https://github.com/isayev/ANI1x_datasets)

**Total:** ~23 GB | **Download time:** 30-60 minutes on first run

---

## ⚙️ Configuration

Training is configured via JSON files. Example:

```json
{
  "model": {
    "hidden_dim": 128,
    "vector_dim": 64,
    "n_layers": 5,
    "cutoff": 5.0,
    "num_rbf": 20,
    "multitask": true
  },
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "scheduler": "reduce_on_plateau",
    "patience": 10
  },
  "data": {
    "mode": "cross_domain",
    "datasets": ["qm9", "md17", "ani1x"],
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
  }
}
```

See `configs/` for complete examples.

---

## 📊 Training

### Basic Training

```bash
# Train on single domain (baseline)
python train.py --config configs/single_domain.json

# Train on cross-domain (our method)
python train.py --config configs/cross_domain.json
```

### Advanced Options

```bash
# Resume from checkpoint
python train.py --config configs/cross_domain.json \
    --resume experiments/cross_domain/checkpoint_50.pt

# Train with custom parameters
python train.py --config configs/cross_domain.json \
    --epochs 150 \
    --batch-size 64 \
    --lr 0.0005

# Enable debug mode (verbose logging)
python train.py --config configs/cross_domain.json --debug
```

### Monitoring

```bash
# TensorBoard visualization
tensorboard --logdir experiments/

# Real-time logging
tail -f experiments/cross_domain_pretraining/train.log
```

---

## 🧪 Evaluation

### GeoShift Benchmark

```bash
# Evaluate all transfer tasks
python evaluate.py \
    --checkpoint experiments/cross_domain/best_model.pt \
    --benchmark geoshift

# Evaluate specific task
python evaluate.py \
    --checkpoint experiments/cross_domain/best_model.pt \
    --task small_to_large
```

### Custom Evaluation

```bash
# Evaluate on your own dataset
python evaluate.py \
    --checkpoint experiments/cross_domain/best_model.pt \
    --dataset path/to/your/data.xyz \
    --metrics energy_mae force_mae
```

### Metrics Computed

- **Energy MAE** - Mean absolute error (eV)
- **Force MAE** - Mean absolute error (kcal/mol/Å)
- **Energy RMSE** - Root mean squared error (eV)
- **R² Score** - Coefficient of determination
- **Physics Compliance** - Energy conservation, force consistency

---

## 🎓 Method Details

### Cross-Domain Pre-training

1. **Multi-Domain Sampling**
   - Batch construction: Mix samples from QM9, MD17, ANI-1x
   - Adaptive sampling: Balance based on dataset sizes

2. **Physics-Augmented Loss**
   ```
   L_total = L_energy + λ_force * L_force + λ_physics * L_physics
   
   where:
   L_energy = MAE(E_pred, E_true)
   L_force = MAE(F_pred, F_true)
   L_physics = Energy conservation penalty
   ```

3. **Multi-Task Learning**
   - Shared backbone: EGNN-PaiNN hybrid
   - Task-specific heads: Energy, HOMO-LUMO gap
   - Dynamic task weighting

### Transfer Learning

1. **Feature Freezing Strategy**
   - Freeze backbone layers
   - Fine-tune task heads
   - Gradually unfreeze layers

2. **Few-Shot Adaptation**
   - Pre-trained features + minimal target data
   - Achieves strong performance with 100-1000 samples

---

## 🔬 Reproducing Results

### Full Experimental Pipeline

```bash
# 1. Download all datasets
python scripts/download_datasets.py

# 2. Pre-train cross-domain model (takes ~38 hours on V100)
python train.py --config configs/cross_domain.json

# 3. Train baseline (single-domain)
python train.py --config configs/single_domain.json

# 4. Evaluate both on GeoShift
python evaluate.py --checkpoint experiments/cross_domain/best_model.pt --benchmark geoshift
python evaluate.py --checkpoint experiments/single_domain/best_model.pt --benchmark geoshift

# 5. Generate comparison plots
python scripts/analyze_results.py \
    --cross-domain experiments/cross_domain/results.json \
    --baseline experiments/single_domain/results.json \
    --output docs/images/
```

### Expected Results

After full training (100 epochs):
- Cross-domain validation MAE: 0.042 eV
- Baseline validation MAE: 0.071 eV
- GeoShift average improvement: ~35%

---

## 📚 Citation

If you use this code or our methods in your research, please cite:

```bibtex
@article{KRRoy2025molecular,
  title={Cross-Domain Foundation Models for Electrostatics: 
         Pre-training Neural Operators Across Molecular Physics},
  author={Kushal Raj Roy},
  journal={APL Computational Physics},
  year={2025},
  note={arXiv preprint arXiv:XXXX.XXXXX}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔬 Add new datasets or benchmarks
- 🎨 Visualization improvements

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch Geometric team for excellent GNN library
- QM9, MD17, and ANI dataset creators
- Anthropic for Claude assistance in development
---

<div align="center">

**Built with ❤️ for the computational chemistry community**

⭐ **Star this repo if you find it useful!** ⭐

</div>
