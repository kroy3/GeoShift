# Cross-Geometry Pretraining for Equivariant Neural Networks: Improving Molecular Property Prediction Under Computational Constraints

[![CI](https://github.com/kroy3/geoshift/actions/workflows/ci.yml/badge.svg)](https://github.com/kroy3/geoshift/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1063%2F5.0317737-blue.svg)](https://doi.org/10.1063/5.0317737)

Code accompanying the paper

> Kushal Raj Roy; Cross-geometry pretraining for equivariant neural networks:
> Improving molecular property prediction under computational constraints.
> *AIP Advances* 1 October 2026; 16 (10): 105006.
> [https://doi.org/10.1063/5.0317737](https://doi.org/10.1063/5.0317737)

The code is distributed as the Python package **`geoshift`** (commands
`geoshift-train`, `geoshift-evaluate` and `geoshift-download`).

It pre-trains an E(3)-equivariant graph neural network across molecular
geometries from QM9 and MD17, trains it jointly on several QM9 properties, and
fine-tunes it on new molecules with very little data (50 rMD17 conformations
per molecule). The whole pipeline is designed to run on a single GPU. This
repository contains the model, the training and evaluation code, and the
configurations for each stage of the paper.

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
| ANI-1x (optional) | ~5M off-equilibrium conformations, ωB97x/6-31G(d) energies and forces | Hartree, Hartree/Å | manual |

The paper uses QM9, MD17 and rMD17. ANI-1x is supported for further
experiments but is not part of the paper's configurations. Energies and forces
are converted to eV and eV/Å on loading. QM9 and (r)MD17 are downloaded through
PyTorch Geometric:

```bash
geoshift-download --data-dir data
```

To use ANI-1x, download the single HDF5 file (`ani1x-release.h5`, ~5 GB).
Download it from the
[figshare release](https://springernature.figshare.com/articles/dataset/ANI-1x_Dataset_Release/10047041)
and place it at `data/ani1x/ani1x-release.h5`.

## Reproducing the experiments

All experiments are driven by the JSON files in [`configs/`](configs). Each
run writes its outputs to its own directory (see [Outputs](#outputs)). The
configurations follow the paper's protocol:

| Stage | Config | Data |
|---|---|---|
| Cross-geometry pretraining | `cross_domain.json` | QM9 (50,000 training molecules) and MD17 aspirin, benzene, ethanol, malonaldehyde (1,000 training / 1,000 test conformations each), sampled with temperature τ = 0.5 |
| Cross-physics multitask learning | `multitask.json` | QM9: U0, HOMO-LUMO gap, H, G and Cv, with gradient-based task balancing |
| Single-task baseline | `single_domain.json` | QM9, one property (set `task_dims`) |
| Few-shot transfer | `transfer.json` | rMD17, 50 training conformations per molecule |

All stages use 5 interaction blocks (128 scalar, 64 vector channels, 5 Å
cutoff), AdamW (learning rate 10⁻³, weight decay 10⁻⁵, gradient clipping at
1.0), mixed precision, an effective batch size of 64 (16 × 4 accumulation
steps), 5 warmup epochs followed by cosine annealing, and early stopping with a
patience of 10 epochs.

```bash
# 1. Cross-geometry pretraining (QM9 + MD17)
geoshift-train --config configs/cross_domain.json

# 2. Cross-physics multitask learning on QM9 (initialised from step 1)
geoshift-train --config configs/multitask.json

# 3. Single-task baseline on QM9 (repeat with each property in task_dims)
geoshift-train --config configs/single_domain.json

# 4. Few-shot transfer to rMD17: pretrained vs. trained from scratch
geoshift-train --config configs/transfer.json
geoshift-train --config configs/transfer.json --pretrained "" \
    --output-dir experiments/transfer_scratch

# 5. Compare
python scripts/analyze_results.py \
    --model experiments/transfer/test_metrics.json \
    --baseline experiments/transfer_scratch/test_metrics.json \
    --metric force_mae
```

For the training-set-size curve, repeat step 4 with
`--max-train-samples N` for N = 10 … 1000 (see [docs/TRAINING.md](docs/TRAINING.md)).

Useful command-line overrides for `geoshift-train`: `--seed`, `--epochs`,
`--batch-size`, `--lr`, `--output-dir`, `--data-dir`, `--device`,
`--max-samples` (cap molecules per dataset for a quick run),
`--max-train-samples`, `--pretrained` and `--resume`.

Splits are random (80/10/10 per dataset, seeded). Caps such as
`max_train_samples` are applied after splitting, so runs with different
amounts of training data are evaluated on the same test molecules.

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

Reported metrics are MAE, RMSE and R² for each predicted property (energies
and gap in eV, Cv in cal/(mol K)) and for force components (eV/Å). Each evaluation also reports the largest change in
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

Atom-wise outputs are summed to molecule-level predictions (`readout: "sum"`).
The multitask variant shares the encoder across tasks with one output head per
task. Forces are computed as the negative gradient of the
predicted energy, so they are conservative and rotate correctly with the
molecule.

The default configuration has 5 blocks, 128 scalar and 64 vector channels, and
about 0.93M parameters.

Training minimises a weighted sum of per-task losses. With
`task_weighting: "gradient"` each task's weight is inversely proportional to
the norm of its gradient, so no single task dominates. Each loss term only uses
the molecules that have that label, so datasets with different labels can be
mixed in one batch. Extensive properties (energies, Cv) are referenced to a
per-element linear fit on the training set, and all targets are standardised.

## Configuration

A configuration has the following sections (see `configs/cross_domain.json`
for a complete example):

| Section | Key | Meaning |
|---|---|---|
| top level | `output_dir`, `seed`, `deterministic` | Output location and reproducibility settings |
| `model` | `hidden_dim`, `vector_dim`, `n_layers`, `cutoff` | Network size and interaction radius (Å) |
| | `readout` | `"sum"` or `"mean"` pooling of atom outputs |
| | `multitask`, `task_dims` | Shared encoder with one head per task: `energy`, `homo_lumo_gap`, `enthalpy`, `free_energy`, `heat_capacity` |
| `data` | `datasets` | List of dataset specs (below) |
| | `sampling_temperature` | Draw dataset k with probability ∝ \|D_k\|^τ (0 = uniform, 1 = by size) |
| | `dataset_weights` | Alternative: explicit sampling probability per dataset |
| | `train_split`, `val_split`, `test_split` | Split fractions, applied per dataset |
| | `max_train_samples`, `max_val_samples`, `max_test_samples` | Caps on split sizes, for all datasets or per dataset spec |
| | `energy_reference` | `"linear_fit"` (per-element reference energies) or `"none"` |
| `training` | `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `gradient_clip` | Optimisation (AdamW) |
| | `gradient_accumulation_steps` | Batches per optimiser step |
| | `scheduler` | `{"type": "warmup_cosine" \| "cosine" \| "reduce_on_plateau", ...}` |
| | `early_stopping_patience`, `save_frequency` | Stopping and checkpointing |
| | `freeze_backbone_epochs` | Train only the output heads for the first N epochs |
| | `use_amp` | Mixed precision (CUDA only) |
| `loss` | `criterion` | `"mae"` or `"mse"` |
| | `task_weighting` | `"fixed"` (use `weights`) or `"gradient"` (gradient-norm balancing) |
| | `weights` | Per-task weights; `forces` > 0 enables force training; a weight of 0 disables a task |
| `transfer` | `pretrained_checkpoint` | Initialise from a checkpoint; tensors are matched by name and shape |

Dataset specs are either a name or an object with options:

```json
"qm9"
"md17:aspirin"
"rmd17:aspirin"
{"name": "qm9", "max_train_samples": 50000}
{"name": "md17:aspirin", "max_train_samples": 1000, "max_test_samples": 1000}
{"name": "qm9", "min_atoms": 20, "max_atoms": 29}
{"name": "ani1x", "max_samples": 50000}
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

> Kushal Raj Roy; Cross-geometry pretraining for equivariant neural networks:
> Improving molecular property prediction under computational constraints.
> *AIP Advances* 1 October 2026; 16 (10): 105006.
> https://doi.org/10.1063/5.0317737

```bibtex
@article{roy2026crossgeometry,
  title   = {Cross-geometry pretraining for equivariant neural networks: Improving molecular property prediction under computational constraints},
  author  = {Roy, Kushal Raj},
  journal = {AIP Advances},
  volume  = {16},
  number  = {10},
  pages   = {105006},
  year    = {2026},
  month   = oct,
  doi     = {10.1063/5.0317737}
}
```

Please also cite the datasets you use: QM9 (Ramakrishnan et al., *Sci. Data*
2014), MD17 (Chmiela et al., *Sci. Adv.* 2017), revised MD17 (Christensen and
von Lilienfeld, *Mach. Learn.: Sci. Technol.* 2020) and ANI-1x (Smith et al.,
*Sci. Data* 2020).

## License

Released under the [MIT License](LICENSE).
