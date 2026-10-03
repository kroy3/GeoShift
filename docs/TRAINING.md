# Training guide

This guide covers practical details not in the [README](../README.md).

## Quick runs

Before a full run, check the setup with a small subset:

```bash
geoshift-train --config configs/cross_domain.json \
    --max-samples 500 --epochs 2 --output-dir experiments/debug
```

`--max-samples` caps the number of molecules loaded from each dataset.

## Resuming

`last.pt` is written after every epoch. To continue an interrupted run:

```bash
geoshift-train --config configs/cross_domain.json --resume experiments/cross_domain/last.pt
```

The optimiser, scheduler, normalisation constants and early-stopping state are
restored from the checkpoint.

## Transfer and few-shot learning

`--pretrained` (or `transfer.pretrained_checkpoint` in the config) copies every
tensor whose name and shape match the new model, and new output heads are
initialised randomly. Loading works between the multitask and single-task
model classes. Use the same `hidden_dim`, `vector_dim`, `n_layers` and
`cutoff` as the pre-trained model.

- `training.freeze_backbone_epochs`: train only the output heads for the first
  N epochs, then fine-tune the whole network.
- `data.max_train_samples` (or `--max-train-samples`): use at most N training
  molecules per dataset. The validation and test splits are unaffected, so
  runs with different N are evaluated on the same molecules.

Sample-efficiency curve on rMD17 (Fig. 1b of the paper), pretrained vs. from scratch:

```bash
for n in 10 20 50 100 200 500 1000; do
  geoshift-train --config configs/transfer.json --max-train-samples $n \
      --output-dir experiments/transfer_pretrained_n$n
  geoshift-train --config configs/transfer.json --max-train-samples $n \
      --pretrained "" --output-dir experiments/transfer_scratch_n$n
done
```

`configs/transfer.json` loads `experiments/cross_domain/best_model.pt` by
default; `--pretrained ""` disables this.

## Mixing datasets

With several datasets, `data.sampling_temperature` (τ) draws dataset k with
probability proportional to |D_k|^τ: τ = 0 samples datasets uniformly, τ = 1
in proportion to their size. The paper uses τ = 0.5. Alternatively,
`data.dataset_weights` sets the probability of each dataset explicitly. With
neither, all training molecules are shuffled together.

## Task balancing and gradient accumulation

With `loss.task_weighting: "gradient"`, each task's weight is recomputed every
step as `(1/|∇L_k|) / Σ_j (1/|∇L_j|)`, which costs one extra backward pass per
task. The validation loss uses the fixed `loss.weights`, so it stays
comparable between epochs. `training.gradient_accumulation_steps` sums
gradients over several batches before each optimiser step; the paper's
effective batch size of 64 is 16 × 4.

## Troubleshooting

| Symptom | Things to try |
|---|---|
| Out of GPU memory | Lower `batch_size`; lower `hidden_dim`; shorten `cutoff` |
| Loss becomes NaN | Lower `learning_rate`; keep `gradient_clip` enabled; disable `use_amp` |
| Slow data loading | Increase `data.num_workers`; the first QM9/MD17 load includes download and processing |
| `FileNotFoundError` for ANI-1x | See "Datasets" in the README; the file must be at `data/ani1x/ani1x-release.h5` |
