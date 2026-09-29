"""Evaluate a trained checkpoint.

Examples::

    # Test split of every dataset the model was trained on
    geoshift-evaluate --checkpoint experiments/cross_domain/best_model.pt

    # Transfer to a dataset that was not used in training (all samples)
    geoshift-evaluate --checkpoint experiments/cross_domain/best_model.pt \
        --dataset rmd17:aspirin --split all
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch_geometric.loader import DataLoader

from geoshift.data import Normalizer, build_splits, load_source, spec_label
from geoshift.model import build_model
from geoshift.utils import resolve_device, set_seed


def _regression_metrics(pred: torch.Tensor, true: torch.Tensor, prefix: str) -> Dict[str, float]:
    if true.numel() == 0:
        return {}
    err = pred - true
    ss_tot = ((true - true.mean()) ** 2).sum()
    metrics = {
        f"{prefix}_mae": err.abs().mean().item(),
        f"{prefix}_rmse": err.pow(2).mean().sqrt().item(),
    }
    if true.numel() > 1 and ss_tot > 0:
        metrics[f"{prefix}_r2"] = (1 - err.pow(2).sum() / ss_tot).item()
    return metrics


def _batch_needs_forces(batch) -> bool:
    return bool(batch.has_force.any())


def predict_physical(model, batch, normalizer: Normalizer, compute_forces: bool):
    """Model predictions converted to physical units (eV, eV/Angstrom)."""
    with torch.set_grad_enabled(compute_forces):
        out = model.predict(batch, compute_forces=compute_forces)
    result = {"energy": normalizer.energy_to_ev(out["energy"].detach(), batch)}
    if "homo_lumo_gap" in out:
        result["gap"] = normalizer.gap_to_ev(out["homo_lumo_gap"].detach().double())
    if compute_forces:
        result["forces"] = normalizer.force_to_ev(out["forces"].detach().double())
    return result


def evaluate(
    model,
    loader,
    normalizer: Normalizer,
    device,
    compute_forces: bool = True,
    domain_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Energy, gap and force errors, overall and per source dataset.

    Returns ``{"overall": {...}, "<dataset>": {...}, ...}``. Energies and gaps
    are in eV, forces in eV/Angstrom.
    """
    model.eval()
    records: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for batch in loader:
        batch = batch.to(device)
        want_forces = compute_forces and _batch_needs_forces(batch)
        pred = predict_physical(model, batch, normalizer, want_forces)
        domain = batch.domain.view(-1).cpu()
        records["domain"].append(domain)
        records["energy_pred"].append(pred["energy"].cpu())
        records["energy_true"].append(batch.energy.double().cpu())
        if "gap" in pred:
            records["gap_pred"].append(pred["gap"].cpu())
            records["gap_true"].append(batch.gap.double().cpu())
            records["gap_domain"].append(domain)
        if want_forces:
            atom_mask = batch.has_force[batch.batch].cpu()
            records["force_pred"].append(pred["forces"].cpu()[atom_mask])
            records["force_true"].append(batch.force.double().cpu()[atom_mask])
            records["force_domain"].append(domain[batch.batch.cpu()][atom_mask])

    cat = {k: torch.cat(v) for k, v in records.items()}

    def summarise(mask_fn) -> Dict[str, float]:
        m = mask_fn(cat["domain"])
        out = {"n_molecules": int(m.sum())}
        out.update(_regression_metrics(cat["energy_pred"][m], cat["energy_true"][m], "energy"))
        if "gap_pred" in cat:
            gm = mask_fn(cat["gap_domain"]) & ~torch.isnan(cat["gap_true"])
            out.update(_regression_metrics(cat["gap_pred"][gm], cat["gap_true"][gm], "gap"))
        if "force_pred" in cat:
            fm = mask_fn(cat["force_domain"])
            out.update(
                _regression_metrics(cat["force_pred"][fm].flatten(), cat["force_true"][fm].flatten(), "force")
            )
        return out

    results = {"overall": summarise(lambda d: torch.ones_like(d, dtype=torch.bool))}
    domains = cat["domain"].unique().tolist()
    if len(domains) > 1 or domain_names:
        for d in domains:
            name = domain_names[d] if domain_names and d < len(domain_names) else f"domain_{d}"
            results[name] = summarise(lambda x, d=d: x == d)
    return results


def random_rotation(generator: torch.Generator, dtype=torch.float32) -> torch.Tensor:
    """Uniformly random proper rotation matrix (det = +1)."""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)


def equivariance_error(model, loader, normalizer: Normalizer, device, n_batches: int = 2, seed: int = 0):
    """Maximum change in predictions under a random rotation and translation.

    Energies (and gaps) should be invariant, and forces should rotate with the
    input. Values are reported in eV and eV/Angstrom and should be at the level
    of floating-point round-off.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    energy_err, force_err = 0.0, 0.0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = batch.to(device)
        rot = random_rotation(gen).to(device)
        shift = torch.randn(1, 3, generator=gen).to(device)

        ref = predict_physical(model, batch.clone(), normalizer, compute_forces=True)
        moved = batch.clone()
        moved.pos = batch.pos @ rot.T + shift
        out = predict_physical(model, moved, normalizer, compute_forces=True)

        energy_err = max(energy_err, (out["energy"] - ref["energy"]).abs().max().item())
        expected = ref["forces"] @ rot.T.double()
        force_err = max(force_err, (out["forces"] - expected).abs().max().item())
    return {"energy_max_abs_error": energy_err, "force_max_abs_error": force_err}


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    return model, normalizer, config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint written by geoshift-train.")
    parser.add_argument(
        "--dataset", nargs="+", default=None,
        help="Dataset specs to evaluate on (default: the datasets in the checkpoint's config).",
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="test",
        help="Split to evaluate. Splits use the checkpoint's seed and fractions (default: test).",
    )
    parser.add_argument("--data-dir", default=None, help="Override the data directory.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-forces", action="store_true", help="Skip force evaluation.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None, help="Output JSON (default: next to the checkpoint).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    model, normalizer, config = load_checkpoint(args.checkpoint, device)
    seed = config.get("seed", 42)
    set_seed(seed)

    data_cfg = dict(config["data"])
    if args.data_dir:
        data_cfg["data_dir"] = args.data_dir
    if args.dataset:
        data_cfg["datasets"] = args.dataset
    data_cfg.pop("max_train_samples", None)
    names = [spec_label(s) for s in data_cfg["datasets"]]

    if args.split == "all":
        samples = []
        for domain, spec in enumerate(data_cfg["datasets"]):
            for s in load_source(spec, data_cfg.get("data_dir", "./data"), seed):
                s.domain = torch.tensor([domain])
                samples.append(s)
    else:
        samples = build_splits(data_cfg, seed)[0][args.split]
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=False)

    results = evaluate(model, loader, normalizer, device, not args.no_forces, names)
    results["equivariance"] = equivariance_error(model, loader, normalizer, device, seed=seed)
    results["meta"] = {"checkpoint": str(args.checkpoint), "split": args.split, "datasets": names}

    output = Path(args.output) if args.output else Path(args.checkpoint).with_name(
        f"eval_{args.split}_{'_'.join(n.replace(':', '-') for n in names)}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
