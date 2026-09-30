"""Dataset loading, splitting and target normalisation.

All datasets are converted to a common sample layout (PyG ``Data``):

=============  ============  =====================================================
attribute      shape         meaning
=============  ============  =====================================================
``z``          ``[N]``       atomic numbers
``pos``        ``[N, 3]``    Cartesian coordinates (Angstrom)
``energy``     ``[1]``       total energy (eV)
``gap``        ``[1]``       HOMO-LUMO gap (eV); NaN when not available
``force``      ``[N, 3]``    forces (eV/Angstrom); zeros when not available
``has_force``  ``[1]``       whether ``force`` holds reference data
``domain``     ``[1]``       index of the source dataset in the config
=============  ============  =====================================================

Dataset specifications
----------------------
A dataset is referenced by a spec string, optionally wrapped in a dict with
extra options::

    "qm9"
    "md17:aspirin"            # original MD17 trajectory
    "rmd17:aspirin"           # revised MD17
    "ani1x"                   # requires data/ani1x/ani1x-release.h5
    "synthetic"               # small random pair-potential systems (tests only)
    {"name": "qm9", "max_samples": 50000, "min_atoms": 10, "max_atoms": 29}
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np
import torch
from torch_geometric.data import Data

HARTREE_TO_EV = 27.211386245988
KCAL_PER_MOL_TO_EV = 0.0433641153087705

# Column indices of torch_geometric.datasets.QM9 targets (already in eV).
QM9_GAP_INDEX = 4
QM9_U0_INDEX = 7

MD17_MOLECULES = (
    "aspirin", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "salicylic acid", "toluene", "uracil",
)

ANI1X_FILENAME = "ani1x-release.h5"
ANI1X_URL = "https://springernature.figshare.com/articles/dataset/ANI-1x_Dataset_Release/10047041"

DatasetSpec = Union[str, Mapping]


def spec_to_dict(spec: DatasetSpec) -> dict:
    return {"name": spec} if isinstance(spec, str) else dict(spec)


def spec_label(spec: DatasetSpec) -> str:
    return spec_to_dict(spec)["name"]


def _sample(z, pos, energy, gap=None, force=None) -> Data:
    n_atoms = z.numel()
    return Data(
        z=z.long(),
        pos=pos.float(),
        energy=torch.tensor([float(energy)]),
        gap=torch.tensor([float("nan") if gap is None else float(gap)]),
        force=torch.zeros(n_atoms, 3) if force is None else force.float(),
        has_force=torch.tensor([force is not None]),
    )


def _load_qm9(root: Path) -> List[Data]:
    from torch_geometric.datasets import QM9

    dataset = QM9(str(root / "qm9"))
    return [
        _sample(d.z, d.pos, d.y[0, QM9_U0_INDEX], gap=d.y[0, QM9_GAP_INDEX])
        for d in dataset
    ]


def _load_md17(root: Path, molecule: str, revised: bool) -> List[Data]:
    from torch_geometric.datasets import MD17

    molecule = molecule.replace("_", " ")
    if molecule not in MD17_MOLECULES:
        raise ValueError(f"Unknown MD17 molecule {molecule!r}; choose from {MD17_MOLECULES}")
    name = f"revised {molecule}" if revised else molecule
    dataset = MD17(str(root / ("rmd17" if revised else "md17")), name=name)
    return [
        _sample(
            d.z, d.pos, d.energy.view(-1)[0] * KCAL_PER_MOL_TO_EV,
            force=d.force * KCAL_PER_MOL_TO_EV,
        )
        for d in dataset
    ]


def _load_ani1x(root: Path, max_samples=None, seed: int = 0) -> List[Data]:
    import h5py

    path = root / "ani1x" / ANI1X_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"ANI-1x not found at {path}. Download '{ANI1X_FILENAME}' from {ANI1X_URL} "
            "and place it there (see README, 'Datasets')."
        )
    with h5py.File(path, "r") as f:
        index = []
        for key in f.keys():
            energies = f[key]["wb97x_dz.energy"][()]
            index.extend((key, i) for i in np.flatnonzero(~np.isnan(energies)))
        if max_samples is not None and max_samples < len(index):
            rng = np.random.default_rng(seed)
            index = [index[i] for i in sorted(rng.choice(len(index), max_samples, replace=False))]

        samples, cache_key, group = [], None, None
        for key, i in index:
            if key != cache_key:
                g = f[key]
                group = {
                    "z": torch.from_numpy(g["atomic_numbers"][()]),
                    "pos": torch.from_numpy(g["coordinates"][()]),
                    "energy": g["wb97x_dz.energy"][()],
                    "force": torch.from_numpy(g["wb97x_dz.forces"][()]),
                }
                cache_key = key
            force = group["force"][i] * HARTREE_TO_EV
            samples.append(
                _sample(
                    group["z"], group["pos"][i], group["energy"][i] * HARTREE_TO_EV,
                    force=None if torch.isnan(force).any() else force,
                )
            )
    return samples


def _load_synthetic(n_samples: int = 256, seed: int = 0) -> List[Data]:
    """Random clusters with a Lennard-Jones-like energy; used for tests only."""
    gen = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(n_samples):
        n = int(torch.randint(3, 9, (1,), generator=gen))
        z = torch.randint(1, 9, (n,), generator=gen)
        pos = (torch.randn(n, 3, generator=gen) * 1.2).requires_grad_(True)
        d = torch.pdist(pos) + 0.5
        energy = (4 * ((1.5 / d) ** 12 - (1.5 / d) ** 6)).clamp(max=50).sum() - 0.3 * z.sum()
        (grad,) = torch.autograd.grad(energy, pos)
        samples.append(_sample(z, pos.detach(), energy.item(), gap=z.float().mean(), force=-grad))
    return samples


def load_source(spec: DatasetSpec, data_dir: Union[str, Path], seed: int = 0) -> List[Data]:
    """Load every sample of a dataset spec (after atom-count filtering and subsampling)."""
    opts = spec_to_dict(spec)
    name = opts["name"]
    root = Path(data_dir)
    family, _, variant = name.partition(":")

    if family == "qm9":
        samples = _load_qm9(root)
    elif family in ("md17", "rmd17"):
        if not variant:
            raise ValueError(f"Specify a molecule, e.g. '{family}:aspirin'")
        samples = _load_md17(root, variant, revised=family == "rmd17")
    elif family == "ani1x":
        samples = _load_ani1x(root, opts.get("max_samples"), seed)
    elif family == "synthetic":
        samples = _load_synthetic(opts.get("n_samples", 256), seed)
    else:
        raise ValueError(f"Unknown dataset {name!r}")

    min_atoms, max_atoms = opts.get("min_atoms"), opts.get("max_atoms")
    if min_atoms is not None or max_atoms is not None:
        lo, hi = min_atoms or 0, max_atoms or math.inf
        samples = [s for s in samples if lo <= s.z.numel() <= hi]

    max_samples = opts.get("max_samples")
    if max_samples is not None and max_samples < len(samples):
        keep = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed))
        samples = [samples[i] for i in sorted(keep[:max_samples].tolist())]
    return samples


def split_indices(n: int, fractions: Sequence[float], seed: int) -> Dict[str, List[int]]:
    """Seeded random train/val/test split."""
    train_frac, val_frac = fractions[0], fractions[1]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_train, n_val = int(train_frac * n), int(val_frac * n)
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def build_splits(data_cfg: Mapping, seed: int):
    """Load all configured datasets and split each of them.

    Returns:
        splits: ``{"train"|"val"|"test": list of Data}``, concatenated over datasets.
        split_record: indices of each sample in its source, per dataset, for provenance.
    """
    fractions = (
        data_cfg.get("train_split", 0.8),
        data_cfg.get("val_split", 0.1),
        data_cfg.get("test_split", 0.1),
    )
    n_train_max = data_cfg.get("max_train_samples")
    splits = {"train": [], "val": [], "test": []}
    record = {}
    for domain, spec in enumerate(data_cfg["datasets"]):
        samples = load_source(spec, data_cfg.get("data_dir", "./data"), seed)
        idx = split_indices(len(samples), fractions, seed)
        if n_train_max is not None:
            idx["train"] = idx["train"][:n_train_max]
        record[spec_label(spec)] = idx
        for split, indices in idx.items():
            for i in indices:
                sample = samples[i]
                sample.domain = torch.tensor([domain])
                splits[split].append(sample)
    return splits, record


def domain_sampler(train: Sequence[Data], data_cfg: Mapping, seed: int):
    """Weighted sampler that draws each dataset with the configured probability.

    Returns ``None`` (plain shuffling) when no dataset weights are configured.
    """
    weights_cfg = data_cfg.get("dataset_weights")
    if not weights_cfg:
        return None
    labels = [spec_label(s) for s in data_cfg["datasets"]]
    missing = set(labels) - set(weights_cfg)
    if missing:
        raise ValueError(f"dataset_weights is missing entries for {sorted(missing)}")
    domains = torch.cat([s.domain for s in train])
    counts = torch.bincount(domains, minlength=len(labels)).clamp(min=1).float()
    per_domain = torch.tensor([float(weights_cfg[label]) for label in labels]) / counts
    return torch.utils.data.WeightedRandomSampler(
        per_domain[domains].double(),
        num_samples=len(train),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


class Normalizer:
    """Energy and gap normalisation fitted on the training set.

    Energies are referenced to a per-element linear model (least-squares fit of
    ``E ~ sum_i e[z_i]``), then standardised. Predictions are made in the
    normalised space; ``energy_to_ev`` and ``force_to_ev`` map them back.
    """

    def __init__(self, atom_ref, energy_mean, energy_std, gap_mean, gap_std):
        self.atom_ref = torch.as_tensor(atom_ref, dtype=torch.float64)
        self.energy_mean = float(energy_mean)
        self.energy_std = float(energy_std)
        self.gap_mean = float(gap_mean)
        self.gap_std = float(gap_std)

    @classmethod
    def fit(cls, train: Sequence[Data], reference: str = "linear_fit", max_z: int = 100):
        energies = torch.cat([s.energy for s in train]).double()
        atom_ref = torch.zeros(max_z, dtype=torch.float64)
        if reference == "linear_fit":
            counts = torch.stack(
                [torch.bincount(s.z, minlength=max_z) for s in train]
            ).double()
            present = counts.sum(0) > 0
            solution = torch.linalg.lstsq(counts[:, present], energies.unsqueeze(1)).solution
            atom_ref[present] = solution.squeeze(1)
        elif reference != "none":
            raise ValueError(f"energy_reference must be 'linear_fit' or 'none', got {reference!r}")
        residual = energies - torch.stack([atom_ref[s.z].sum() for s in train])
        gaps = torch.cat([s.gap for s in train])
        gaps = gaps[~torch.isnan(gaps)]
        return cls(
            atom_ref,
            residual.mean(),
            _safe_std(residual),
            gaps.mean() if gaps.numel() else 0.0,
            _safe_std(gaps) if gaps.numel() > 1 else 1.0,
        )

    def reference_energy(self, batch) -> torch.Tensor:
        ref = self.atom_ref.to(batch.pos.device)[batch.z]
        n_graphs = batch.energy.size(0)
        return torch.zeros(n_graphs, dtype=ref.dtype, device=ref.device).index_add_(
            0, batch.batch, ref
        )

    def normalize_energy(self, batch) -> torch.Tensor:
        residual = batch.energy.double() - self.reference_energy(batch)
        return ((residual - self.energy_mean) / self.energy_std).float().unsqueeze(1)

    def energy_to_ev(self, pred: torch.Tensor, batch) -> torch.Tensor:
        residual = pred.view(-1).double() * self.energy_std + self.energy_mean
        return residual + self.reference_energy(batch)

    def normalize_force(self, force: torch.Tensor) -> torch.Tensor:
        return force / self.energy_std

    def force_to_ev(self, pred: torch.Tensor) -> torch.Tensor:
        return pred * self.energy_std

    def normalize_gap(self, gap: torch.Tensor) -> torch.Tensor:
        return ((gap - self.gap_mean) / self.gap_std).unsqueeze(1)

    def gap_to_ev(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.view(-1) * self.gap_std + self.gap_mean

    def state_dict(self) -> dict:
        return {
            "atom_ref": self.atom_ref.tolist(),
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "gap_mean": self.gap_mean,
            "gap_std": self.gap_std,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "Normalizer":
        return cls(**state)


def _safe_std(x: torch.Tensor) -> float:
    std = float(x.std()) if x.numel() > 1 else 1.0
    return std if std > 1e-12 else 1.0
