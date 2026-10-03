"""Train a model from a JSON configuration.

Examples::

    geoshift-train --config configs/cross_domain.json
    geoshift-train --config configs/single_domain.json --seed 1
    geoshift-train --config configs/transfer.json \
        --pretrained experiments/cross_domain/best_model.pt

Every run writes, into its output directory: the resolved config, software
versions, the exact split indices, per-epoch metrics, checkpoints and the
test-set metrics of the best checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from geoshift.data import Normalizer, build_splits, domain_sampler, spec_label, spec_to_dict
from geoshift.evaluate import equivariance_error, evaluate
from geoshift.model import MultitaskCrossDomainModel, backbone_parameters, build_model
from geoshift.utils import environment_info, load_config, resolve_device, set_seed

CRITERIA = {"mae": F.l1_loss, "mse": F.mse_loss}


def compute_loss(model, batch, normalizer: Normalizer, loss_cfg: Mapping):
    """Per-task losses in normalised units.

    Every task head of the model contributes a term, as do forces when their
    weight is positive. Each term only uses molecules that have reference
    values, so datasets with different labels can be mixed in one batch.

    Returns ``{task: loss tensor}``.
    """
    criterion = CRITERIA[loss_cfg.get("criterion", "mae")]
    weights = loss_cfg.get("weights", {})
    need_forces = weights.get("forces", 0) > 0 and bool(batch.has_force.any())

    out = model.predict(batch, compute_forces=need_forces)
    terms = {}
    for task, pred in out.items():
        if task == "forces" or weights.get(task, 1.0) <= 0 or task not in normalizer.stats:
            continue
        target = normalizer.normalize(task, batch)
        mask = ~torch.isnan(target.view(-1))
        if mask.any():
            terms[task] = criterion(pred[mask], target[mask])

    if need_forces:
        atom_mask = batch.has_force[batch.batch]
        target = normalizer.normalize_force(batch.force[atom_mask])
        terms["forces"] = criterion(out["forces"][atom_mask], target)
    return terms


def gradient_balanced_weights(terms: Mapping[str, torch.Tensor], params) -> Dict[str, torch.Tensor]:
    """Task weights inversely proportional to each task's gradient norm.

    ``lambda_k = (1 / |grad L_k|) / sum_j (1 / |grad L_j|)``, so every task
    contributes a gradient of the same magnitude.
    """
    params = [p for p in params if p.requires_grad]
    inverse = {}
    for name, loss in terms.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        norm = torch.sqrt(sum(g.float().pow(2).sum() for g in grads if g is not None))
        inverse[name] = 1.0 / (norm + 1e-12)
    total = sum(inverse.values())
    return {name: (value / total).detach() for name, value in inverse.items()}


def combine_terms(model, terms, loss_cfg: Mapping, balance: bool) -> torch.Tensor:
    """Weighted sum of loss terms: fixed config weights or gradient balancing."""
    if balance and len(terms) > 1:
        weights = gradient_balanced_weights(terms, model.parameters())
    else:
        cfg_weights = loss_cfg.get("weights", {})
        weights = {name: cfg_weights.get(name, 1.0) for name in terms}
    return sum(weights[name] * value for name, value in terms.items())


def load_pretrained(model, path, device) -> None:
    """Copy every pretrained tensor whose name and shape match the new model.

    Handles transfer between the multitask and single-task model classes by
    adding or stripping the ``encoder.`` prefix.
    """
    state = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    is_multitask = isinstance(model, MultitaskCrossDomainModel)
    remapped = {}
    for key, value in state.items():
        if is_multitask and not key.startswith(("encoder.", "task_heads.")):
            key = "encoder." + key
        elif not is_multitask and key.startswith("encoder."):
            key = key[len("encoder."):]
        remapped[key] = value

    own = model.state_dict()
    loaded = {k: v for k, v in remapped.items() if k in own and own[k].shape == v.shape}
    own.update(loaded)
    model.load_state_dict(own)
    skipped = sorted(set(own) - set(loaded))
    print(f"Loaded {len(loaded)}/{len(own)} tensors from {path}")
    if skipped:
        print(f"  Newly initialised: {', '.join(skipped)}")


def set_backbone_trainable(model, trainable: bool) -> None:
    for param in backbone_parameters(model):
        param.requires_grad_(trainable)


def run_epoch(model, loader, normalizer, loss_cfg, device, optimizer=None, scaler=None, clip=None,
              accumulation_steps: int = 1):
    """One pass over ``loader``; trains when an optimizer is given.

    During training, gradients are accumulated over ``accumulation_steps``
    batches before each optimiser step. With ``loss.task_weighting`` set to
    ``"gradient"``, training uses gradient-balanced task weights; the reported
    (validation) loss always uses the fixed config weights so it is comparable
    between epochs.
    """
    training = optimizer is not None
    model.train(training)
    balance = training and loss_cfg.get("task_weighting", "fixed") == "gradient"
    totals: Dict[str, float] = {}
    n_batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        # Forces are gradients, so autograd is needed even during validation.
        with torch.set_grad_enabled(True):
            with torch.autocast(device.type, enabled=scaler is not None):
                terms = compute_loss(model, batch, normalizer, loss_cfg)
            report = combine_terms(model, terms, loss_cfg, balance=False)
            if training:
                loss = combine_terms(model, terms, loss_cfg, balance) if balance else report
                loss = loss / accumulation_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        totals["loss"] = totals.get("loss", 0.0) + report.item()
        for name, value in terms.items():
            totals[f"loss_{name}"] = totals.get(f"loss_{name}", 0.0) + value.item()
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def build_scheduler(optimizer, cfg: Mapping, epochs: int):
    """Learning-rate schedule, stepped once per epoch."""
    kind = cfg.get("type", "reduce_on_plateau")
    if kind == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.get("factor", 0.5),
            patience=cfg.get("patience", 10), min_lr=cfg.get("min_lr", 1e-6),
        )
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=cfg.get("min_lr", 1e-6)
        )
    if kind == "warmup_cosine":
        warmup = cfg.get("warmup_epochs", 5)
        base_lr = optimizer.param_groups[0]["lr"]
        floor = cfg.get("min_lr", 1e-6) / base_lr

        def factor(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    raise ValueError(f"Unknown scheduler type {kind!r}")


def apply_overrides(config: dict, args) -> dict:
    config = copy.deepcopy(config)
    training = config.setdefault("training", {})
    data = config.setdefault("data", {})
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.lr is not None:
        training["learning_rate"] = args.lr
    if args.data_dir:
        data["data_dir"] = args.data_dir
    if args.num_workers is not None:
        data["num_workers"] = args.num_workers
    if args.max_train_samples is not None:
        data["max_train_samples"] = args.max_train_samples
        specs = [spec_to_dict(s) for s in data["datasets"]]
        for spec in specs:
            spec.pop("max_train_samples", None)  # the command line wins over per-dataset caps
        data["datasets"] = specs
    if args.max_samples is not None:
        specs = [spec_to_dict(s) for s in data["datasets"]]
        for spec in specs:
            spec["max_samples"] = min(args.max_samples, spec.get("max_samples") or args.max_samples)
        data["datasets"] = specs
    if args.pretrained is not None:  # "" disables a checkpoint set in the config
        config.setdefault("transfer", {})["pretrained_checkpoint"] = args.pretrained
    return config


def train(config: dict, device: torch.device, resume=None) -> Path:
    seed = config.get("seed", 42)
    set_seed(seed, config.get("deterministic", False))
    out_dir = Path(config.get("output_dir", f"experiments/{config.get('experiment_name', 'run')}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tcfg, dcfg, loss_cfg = config["training"], config["data"], config.get("loss", {})
    names = [spec_label(s) for s in dcfg["datasets"]]

    splits, split_record = build_splits(dcfg, seed)
    for split, samples in splits.items():
        print(f"{split:>5}: {len(samples)} molecules")
    normalizer = Normalizer.fit(splits["train"], dcfg.get("energy_reference", "linear_fit"))

    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "environment.json").write_text(json.dumps(environment_info(), indent=2))
    (out_dir / "splits.json").write_text(json.dumps(split_record))

    loader_kwargs = dict(num_workers=dcfg.get("num_workers", 0), pin_memory=device.type == "cuda")
    sampler = domain_sampler(splits["train"], dcfg, seed)
    train_loader = DataLoader(
        splits["train"], batch_size=tcfg.get("batch_size", 32), sampler=sampler,
        shuffle=sampler is None, generator=torch.Generator().manual_seed(seed), **loader_kwargs,
    )
    eval_bs = tcfg.get("batch_size", 32) * 2
    val_loader = DataLoader(splits["val"], batch_size=eval_bs, **loader_kwargs)
    test_loader = DataLoader(splits["test"], batch_size=eval_bs, **loader_kwargs)

    model = build_model(config).to(device)
    print(f"Model: {type(model).__name__}, {sum(p.numel() for p in model.parameters()):,} parameters")
    pretrained = config.get("transfer", {}).get("pretrained_checkpoint")
    if pretrained and resume is None:
        load_pretrained(model, pretrained, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.get("learning_rate", 1e-3), weight_decay=tcfg.get("weight_decay", 1e-5)
    )
    epochs = tcfg.get("epochs", 100)
    scheduler = build_scheduler(optimizer, tcfg.get("scheduler", {}), epochs)
    use_amp = tcfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch, best_val, stale_epochs = 0, float("inf"), 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        normalizer = Normalizer.from_state_dict(ckpt["normalizer"])
        start_epoch, best_val, stale_epochs = ckpt["epoch"] + 1, ckpt["best_val_loss"], ckpt["stale_epochs"]
        print(f"Resumed from {resume} at epoch {start_epoch}")

    writer = _tensorboard_writer(config, out_dir)
    metrics_path = out_dir / "metrics.csv"
    freeze_epochs = tcfg.get("freeze_backbone_epochs", 0)
    patience = tcfg.get("early_stopping_patience", 20)
    save_every = tcfg.get("save_frequency", 10)

    def save(path, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "normalizer": normalizer.state_dict(),
                "best_val_loss": best_val,
                "stale_epochs": stale_epochs,
                "config": config,
            },
            path,
        )

    start = time.time()
    for epoch in range(start_epoch, epochs):
        set_backbone_trainable(model, epoch >= freeze_epochs)
        train_m = run_epoch(model, train_loader, normalizer, loss_cfg, device, optimizer, scaler,
                            tcfg.get("gradient_clip"), tcfg.get("gradient_accumulation_steps", 1))
        val_m = run_epoch(model, val_loader, normalizer, loss_cfg, device)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_m["loss"])
        else:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in train_m.items()},
               **{f"val_{k}": v for k, v in val_m.items()}, "elapsed_s": round(time.time() - start, 1)}
        _append_csv(metrics_path, row)
        if writer:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
        print(f"epoch {epoch:4d} | train {train_m['loss']:.4f} | val {val_m['loss']:.4f} | lr {lr:.2e}")

        if val_m["loss"] < best_val:
            best_val, stale_epochs = val_m["loss"], 0
            save(out_dir / "best_model.pt", epoch)
        else:
            stale_epochs += 1
        save(out_dir / "last.pt", epoch)
        if save_every and (epoch + 1) % save_every == 0:
            save(out_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt", epoch)
        if stale_epochs >= patience:
            print(f"Early stopping: no improvement for {patience} epochs")
            break
    if writer:
        writer.close()

    best = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    set_backbone_trainable(model, True)
    results = evaluate(model, test_loader, normalizer, device, domain_names=names)
    results["equivariance"] = equivariance_error(model, test_loader, normalizer, device, seed=seed)
    results["best_epoch"] = best["epoch"]
    results["training_time_hours"] = (time.time() - start) / 3600
    (out_dir / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["overall"], indent=2))
    print(f"Outputs written to {out_dir}")
    return out_dir


def _append_csv(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _tensorboard_writer(config, out_dir: Path):
    if not config.get("logging", {}).get("tensorboard", False):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard is not installed; continuing without it (pip install tensorboard).")
        return None
    return SummaryWriter(out_dir / "tensorboard")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a JSON config (see configs/).")
    parser.add_argument("--output-dir", help="Override the output directory.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--data-dir")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int, help="Cap molecules loaded per dataset (quick runs).")
    parser.add_argument("--max-train-samples", type=int,
                        help="Cap training molecules per dataset; val/test splits are unchanged.")
    parser.add_argument("--pretrained", help="Checkpoint to initialise weights from (transfer learning).")
    parser.add_argument("--resume", help="Resume from a checkpoint in the same output directory.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda' or 'cuda:N'.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    train(config, resolve_device(args.device), resume=args.resume)


if __name__ == "__main__":
    main()
