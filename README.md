# GeoShift: Cross-Domain Foundation Models for Electrostatics

[![CI](https://github.com/kroy3/geoshift/actions/workflows/ci.yml/badge.svg)](https://github.com/kroy3/geoshift/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1063%2F5.0317737-blue.svg)](https://doi.org/10.1063/5.0317737)

Code accompanying the paper

> K. R. Roy, *Cross-Domain Foundation Models for Electrostatics: Pre-training
> Neural Operators Across Molecular Physics*, APL Computational Physics (2025).
> [doi:10.1063/5.0317737](https://doi.org/10.1063/5.0317737)

GeoShift pre-trains an E(3)-equivariant graph neural network jointly on
several molecular datasets (QM9, MD17 and ANI-1x) and transfers it to new
molecular geometries and chemical spaces. This repository contains the model,
the training and evaluation pipeline, and the configurations used in the
paper.

## Contents

- [Installation](#installation)
- [Quick check](#quick-check)
- [Datasets](#datasets)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Model](#model)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Citation](#citation)

## Installation

Python 3.9 or newer is required. Install PyTorch for your platform first
(see [pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
git clone https://github.com/kroy3/geoshift.git
cd geoshift
pip install -e ".[analysis]"      # add ",dev" to run the test suite
```

No compiled PyTorch Geometric extensions (`torch-scatter`, `torch-cluster`,
...) are needed.

## Quick check

The smoke-test configuration trains a small model for three epochs on a
synthetic dataset and exercises the full pipeline (data, training,
checkpointing, evaluation) in under a minute on a CPU:

```bash
geoshift-train --config configs/smoke_test.json
pytest                                # requires the "dev" extra
```

## Datasets

| Dataset | Content | Units in files | How to obtain |
|---|---|---|---|
| QM9 | ~134k small organic molecules at equilibrium (DFT) | eV | automatic |
| MD17 / rMD17 | MD trajectories of small molecules with energies and forces | kcal/mol, kcal/mol/Å | automatic |
| ANI-1x | ~5M off-equilibrium conformations, ωB97x/6-31G(d) energies and forces | Hartree, Hartree/Å | manual |

All quantities are converted to eV and eV/Å on loading. QM9 and (r)MD17 are
downloaded through PyTorch Geometric:

```bash
geoshift-download --data-dir data
```

ANI-1x is distributed as a single HDF5 file (`ani1x-release.h5`, ~5 GB).
Download it from the
[figshare release](https://springernature.figshare.com/articles/dataset/ANI-1x_Dataset_Release/10047041)
and place it at `data/ani1x/ani1x-release.h5`.

## Reproducing the experiments

All experiments are driven by the JSON files in [`configs/`](configs). Each
run writes its outputs to its own directory (see [Outputs](#outputs)).

```bash
# 1. Cross-domain pre-training (QM9 + MD17 + ANI-1x)
geoshift-train --config configs/cross_domain.json

# 2. Single-domain baseline (QM9 only)
geoshift-train --config configs/single_domain.json

# 3. Transfer to a new domain (fine-tuning on revised MD17 aspirin)
geoshift-train --config configs/transfer.json \
    --pretrained experiments/cross_domain/best_model.pt

# 4. Evaluate a checkpoint on a dataset it was not trained on
geoshift-evaluate --checkpoint experiments/cross_domain/best_model.pt \
    --dataset rmd17:aspirin --split all
geoshift-evaluate --checkpoint experiments/single_domain/best_model.pt \
    --dataset rmd17:aspirin --split all

# 5. Compare the two
python scripts/analyze_results.py \
    --model experiments/cross_domain/eval_all_rmd17-aspirin.json \
    --baseline experiments/single_domain/eval_all_rmd17-aspirin.json
```

Useful command-line overrides for `geoshift-train`: `--seed`, `--epochs`,
`--batch-size`, `--lr`, `--output-dir`, `--data-dir`, `--device`,
`--max-samples` (cap molecules per dataset for a quick run) and `--resume`.

To evaluate transfer to larger molecules, restrict a dataset by atom count,
for example `{"name": "qm9", "min_atoms": 20}` in a config's `datasets` list.

Results vary slightly between hardware and library versions even with fixed
seeds, because some GPU scatter operations are non-deterministic. Each run
records the software versions and git commit it used in `environment.json`.

### Outputs

| File | Content |
|---|---|
| `config.json` | Fully resolved configuration, including command-line overrides |
| `environment.json` | Python, PyTorch, PyG and CUDA versions, GPU and git commit |
| `splits.json` | Indices of the train/val/test molecules in each source dataset |
| `metrics.csv` | Per-epoch training and validation losses and learning rate |
| `best_model.pt`, `last.pt` | Best (lowest validation loss) and most recent checkpoints |
| `test_metrics.json` | Test-set metrics of the best checkpoint, overall and per dataset |
| `tensorboard/` | TensorBoard logs (if enabled and installed) |

Reported metrics are MAE, RMSE and R² for energies (eV), HOMO-LUMO gaps (eV)
and force components (eV/Å). Each evaluation also reports the largest change in
predictions under a random rotation and translation of the input, which checks
the model's symmetry numerically.

## Model

The network (`src/geoshift/model.py`) embeds atomic numbers and applies a stack
of interaction blocks on a radius graph (default cutoff 5 Å). Each block
consists of

1. an **equivariant message-passing layer** (EGNN-style): messages are
   computed from the features of both atoms and their distance, weighted by a
   smooth cutoff envelope; they update scalar features and accumulate vector
   features along the interatomic directions;
2. a **scalar–vector mixing layer** (PaiNN-style): vector-feature norms update
   the scalar features, which in turn gate the vector features.

Atom-wise outputs are pooled to molecule-level predictions. The multitask
variant shares the encoder across tasks (energy, HOMO-LUMO gap) with one
output head per task. Forces are computed as the negative gradient of the
predicted energy, so they are conservative and rotate correctly with the
molecule.

The default configuration has 5 blocks, 128 scalar and 64 vector channels, and
about 0.93M parameters.

Training uses a weighted sum of energy, gap and force losses. Each loss term
only uses the molecules that have that label, so datasets with different
labels can be mixed in one batch. Energies are referenced to a per-element
linear fit on the training set and standardised before training.

## Configuration

A configuration has the following sections (see `configs/cross_domain.json`
for a complete example):

| Section | Key | Meaning |
|---|---|---|
| top level | `output_dir`, `seed`, `deterministic` | Output location and reproducibility settings |
| `model` | `hidden_dim`, `vector_dim`, `n_layers`, `cutoff` | Network size and interaction radius (Å) |
| | `readout` | `"mean"` or `"sum"` pooling of atom outputs |
| | `multitask`, `task_dims` | Shared encoder with one head per task |
| `data` | `datasets` | List of dataset specs (below) |
| | `dataset_weights` | Sampling probability of each dataset per batch |
| | `train_split`, `val_split`, `test_split` | Split fractions, applied per dataset |
| | `max_train_samples` | Cap on training molecules per dataset (few-shot runs) |
| | `energy_reference` | `"linear_fit"` (per-element reference energies) or `"none"` |
| `training` | `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `gradient_clip` | Optimisation (AdamW) |
| | `scheduler` | `{"type": "reduce_on_plateau" \| "cosine", ...}` |
| | `early_stopping_patience`, `save_frequency` | Stopping and checkpointing |
| | `freeze_backbone_epochs` | Train only the output heads for the first N epochs |
| | `use_amp` | Mixed precision (CUDA only) |
| `loss` | `criterion`, `weights` | `"mae"` or `"mse"`; weights for `energy`, `homo_lumo_gap`, `forces` |
| `transfer` | `pretrained_checkpoint` | Initialise from a checkpoint; tensors are matched by name and shape |

Dataset specs are either a name or an object with options:

```json
"qm9"
"md17:aspirin"
"rmd17:aspirin"
{"name": "ani1x", "max_samples": 50000}
{"name": "qm9", "min_atoms": 20, "max_atoms": 29}
```

## Repository layout

```
configs/            experiment configurations
scripts/            dataset download and result comparison
src/geoshift/
    model.py        equivariant network
    data.py         dataset loading, splits, normalisation
    train.py        training entry point (geoshift-train)
    evaluate.py     metrics and evaluation entry point (geoshift-evaluate)
    download.py     dataset download (geoshift-download)
    utils.py        configuration, seeding, provenance
tests/              unit and end-to-end tests
```

## Citation

If you use this code, please cite:

```bibtex
@article{roy2025geoshift,
  title   = {Cross-Domain Foundation Models for Electrostatics: Pre-training Neural Operators Across Molecular Physics},
  author  = {Roy, Kushal Raj},
  journal = {APL Computational Physics},
  year    = {2025},
  doi     = {10.1063/5.0317737}
}
```

Please also cite the datasets you use: QM9 (Ramakrishnan et al., *Sci. Data*
2014), MD17 (Chmiela et al., *Sci. Adv.* 2017), revised MD17 (Christensen and
von Lilienfeld, *Mach. Learn.: Sci. Technol.* 2020) and ANI-1x (Smith et al.,
*Sci. Data* 2020).

## License

Released under the [MIT License](LICENSE).
