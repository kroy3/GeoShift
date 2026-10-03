import json

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from geoshift import evaluate as evaluate_cli
from geoshift.data import (
    HARTREE_TO_EV,
    Normalizer,
    build_splits,
    domain_probabilities,
    load_source,
    split_indices,
)
from geoshift.train import build_scheduler, gradient_balanced_weights, train
from geoshift.utils import load_config


def test_splits_are_disjoint_and_seeded():
    a = split_indices(100, (0.8, 0.1, 0.1), seed=3)
    b = split_indices(100, (0.8, 0.1, 0.1), seed=3)
    assert a == b
    assert len(a["train"]) == 80 and len(a["val"]) == 10 and len(a["test"]) == 10
    assert set(a["train"]).isdisjoint(a["test"]) and set(a["val"]).isdisjoint(a["test"])


def test_normalizer_round_trip():
    samples = load_source({"name": "synthetic", "n_samples": 64}, "unused")
    norm = Normalizer.fit(samples)
    batch = next(iter(DataLoader(samples, batch_size=16)))
    normalised = norm.normalize_energy(batch)
    torch.testing.assert_close(
        norm.energy_to_ev(normalised, batch), batch.energy.double(), rtol=1e-5, atol=1e-4
    )

    restored = Normalizer.from_state_dict(json.loads(json.dumps(norm.state_dict())))
    torch.testing.assert_close(restored.stats["energy"]["atom_ref"], norm.stats["energy"]["atom_ref"])
    gap = norm.normalize("homo_lumo_gap", batch)
    torch.testing.assert_close(norm.to_physical("homo_lumo_gap", gap, batch), batch.gap.double())


def test_normalizer_skips_missing_labels_and_reads_old_checkpoints():
    samples = load_source({"name": "synthetic", "n_samples": 32}, "unused")
    norm = Normalizer.fit(samples)
    assert "enthalpy" not in norm.stats  # synthetic data has no enthalpy labels
    old = {"atom_ref": [0.0] * 100, "energy_mean": 1.0, "energy_std": 2.0, "gap_mean": 3.0, "gap_std": 4.0}
    restored = Normalizer.from_state_dict(old)
    assert restored.stats["energy"]["std"] == 2.0 and restored.stats["homo_lumo_gap"]["mean"] == 3.0


def test_per_dataset_split_caps():
    cfg = {"datasets": [{"name": "synthetic", "n_samples": 100, "max_train_samples": 7, "max_test_samples": 3}]}
    splits, record = build_splits(cfg, seed=0)
    assert len(splits["train"]) == 7 and len(splits["test"]) == 3 and len(splits["val"]) == 10
    uncapped = split_indices(100, (0.8, 0.1, 0.1), seed=0)
    assert record["synthetic"]["test"] == uncapped["test"][:3]


def test_sampling_temperature():
    counts = torch.tensor([100.0, 400.0])
    torch.testing.assert_close(domain_probabilities(counts, 0.0), torch.tensor([0.5, 0.5], dtype=torch.float64))
    torch.testing.assert_close(domain_probabilities(counts, 1.0), torch.tensor([0.2, 0.8], dtype=torch.float64))
    torch.testing.assert_close(domain_probabilities(counts, 0.5), torch.tensor([1 / 3, 2 / 3], dtype=torch.float64))


def test_gradient_balanced_weights_equalise_gradient_norms():
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    terms = {"a": (w * 10).sum(), "b": (w * 0.1).sum()}
    weights = gradient_balanced_weights(terms, [w])
    torch.testing.assert_close(sum(weights.values()), torch.tensor(1.0))
    norm_a = weights["a"] * 10 * w.numel() ** 0.5
    norm_b = weights["b"] * 0.1 * w.numel() ** 0.5
    torch.testing.assert_close(norm_a, norm_b)


def test_warmup_cosine_schedule():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = build_scheduler(opt, {"type": "warmup_cosine", "warmup_epochs": 5, "min_lr": 1e-6}, epochs=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(2e-4) and lrs[4] == pytest.approx(1e-3)
    assert lrs[5] == pytest.approx(1e-3) and lrs[-1] < 1e-5
    assert all(a >= b for a, b in zip(lrs[5:], lrs[6:]))


def test_ani1x_loader_reads_release_schema(tmp_path):
    h5py = pytest.importorskip("h5py")
    (tmp_path / "ani1x").mkdir()
    with h5py.File(tmp_path / "ani1x" / "ani1x-release.h5", "w") as f:
        g = f.create_group("C1H4")
        g["atomic_numbers"] = np.array([6, 1, 1, 1, 1])
        g["coordinates"] = np.random.default_rng(0).normal(size=(3, 5, 3))
        g["wb97x_dz.energy"] = np.array([-40.5, np.nan, -40.4])
        forces = np.zeros((3, 5, 3))
        forces[2] = np.nan
        g["wb97x_dz.forces"] = forces

    samples = load_source("ani1x", tmp_path)
    assert len(samples) == 2  # NaN-energy conformer dropped
    assert samples[0].energy.item() == pytest.approx(-40.5 * HARTREE_TO_EV)
    assert bool(samples[0].has_force) and not bool(samples[1].has_force)


def test_atom_count_filter():
    samples = load_source({"name": "synthetic", "n_samples": 64, "min_atoms": 5, "max_atoms": 6}, "unused")
    assert samples and all(5 <= s.z.numel() <= 6 for s in samples)


def test_train_evaluate_and_transfer(tmp_path):
    config = load_config("configs/smoke_test.json")
    config["output_dir"] = str(tmp_path / "pretrain")
    out_dir = train(config, torch.device("cpu"))

    for name in ["config.json", "environment.json", "splits.json", "metrics.csv",
                 "best_model.pt", "last.pt", "test_metrics.json"]:
        assert (out_dir / name).exists(), name
    metrics = json.loads((out_dir / "test_metrics.json").read_text())
    assert metrics["overall"]["n_molecules"] == len(build_splits(config["data"], config["seed"])[0]["test"])
    assert metrics["equivariance"]["energy_max_abs_error"] < 1e-3

    eval_out = tmp_path / "eval.json"
    evaluate_cli.main(["--checkpoint", str(out_dir / "best_model.pt"), "--device", "cpu",
                       "--output", str(eval_out)])
    assert "energy_mae" in json.loads(eval_out.read_text())["overall"]

    transfer = dict(config, output_dir=str(tmp_path / "transfer"))
    transfer["model"] = dict(config["model"], multitask=False)
    transfer["transfer"] = {"pretrained_checkpoint": str(out_dir / "best_model.pt")}
    transfer["training"] = dict(config["training"], epochs=2, freeze_backbone_epochs=1)
    train(transfer, torch.device("cpu"))
