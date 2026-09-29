"""Hybrid equivariant network for molecular property prediction.

Each interaction block has two parts:

* ``EquivariantMessagePassingLayer``: EGNN-style messages built from invariant
  features and interatomic distances. They update the scalar features and
  accumulate directional (vector) features along bond directions.
* ``ScalarVectorMixing``: PaiNN-style exchange between the scalar features and
  the norms of the vector features.

Scalar outputs are invariant, and vector features rotate with the input, under
E(3) transformations of the atomic positions. Forces are obtained as the
negative gradient of the predicted energy, so they are equivariant by
construction.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import scatter

DEFAULT_TASK_DIMS = {"energy": 1, "homo_lumo_gap": 1}


class MLP(nn.Module):
    """Fully connected network with SiLU activations between layers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 2):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@torch.no_grad()
def radius_graph(pos: torch.Tensor, batch: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Directed edges between distinct atoms of the same molecule within ``cutoff``.

    Returns ``edge_index`` of shape ``[2, E]`` with row 0 the source atom and
    row 1 the receiving atom.
    """
    same_molecule = batch.unsqueeze(0) == batch.unsqueeze(1)
    within_cutoff = torch.cdist(pos, pos) < cutoff
    adjacency = same_molecule & within_cutoff
    adjacency.fill_diagonal_(False)
    src, dst = adjacency.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)


class EquivariantMessagePassingLayer(nn.Module):
    """Message passing that updates scalar features ``h`` and vector features ``V``."""

    def __init__(self, hidden_dim: int, vector_dim: int, cutoff: float = 5.0):
        super().__init__()
        if vector_dim > hidden_dim:
            raise ValueError("vector_dim must not exceed hidden_dim")
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        self.cutoff = cutoff

        # Input: receiver features, sender features, distance.
        self.edge_mlp = MLP(2 * hidden_dim + 1, hidden_dim, hidden_dim)
        self.node_mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.vector_gate = MLP(hidden_dim, hidden_dim, vector_dim)

    def cutoff_function(self, distance: torch.Tensor) -> torch.Tensor:
        """Smooth polynomial envelope that decays to zero at ``cutoff``."""
        x = torch.clamp(distance / self.cutoff, 0.0, 1.0)
        return 1 - 6 * x**5 + 15 * x**4 - 10 * x**3

    def forward(self, h, pos, V, edge_index):
        """
        Args:
            h: Scalar features ``[N, hidden_dim]``.
            pos: Atomic positions ``[N, 3]``.
            V: Vector features ``[N, 3, vector_dim]``.
            edge_index: Edges ``[2, E]`` (source, receiver).
        """
        src, dst = edge_index
        rel_pos = pos[src] - pos[dst]
        distance = torch.norm(rel_pos, dim=-1, keepdim=True)

        edge_feat = torch.cat([h[dst], h[src], distance], dim=-1)
        message = self.edge_mlp(edge_feat) * self.cutoff_function(distance)
        direction = rel_pos / (distance + 1e-8)

        n_atoms = h.size(0)
        h_message = scatter(message, dst, dim=0, dim_size=n_atoms, reduce="sum")
        vector_message = direction.unsqueeze(-1) * message[:, None, : self.vector_dim]
        V_message = scatter(vector_message, dst, dim=0, dim_size=n_atoms, reduce="sum")

        h = h + self.node_mlp(h_message)
        gate = self.vector_gate(h_message).unsqueeze(1)
        V = V + V_message * torch.sigmoid(gate)
        return h, V


class ScalarVectorMixing(nn.Module):
    """Exchange information between scalar features and vector-feature norms."""

    def __init__(self, hidden_dim: int, vector_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        self.norm_mlp = MLP(hidden_dim + vector_dim, hidden_dim, hidden_dim)
        self.gate_mlp = MLP(hidden_dim, hidden_dim, vector_dim)

    def forward(self, h, V):
        V_norm = _safe_norm(V, dim=1)
        h = h + self.norm_mlp(torch.cat([h, V_norm], dim=-1))
        V = V * torch.sigmoid(self.gate_mlp(h).unsqueeze(1))
        return h, V


class CrossDomainEquivariantNet(nn.Module):
    """Equivariant encoder with a molecule-level output.

    ``forward`` returns ``(molecule_output, atom_output)``. When the model is
    used on its own, the first output channel is the (normalised) energy.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        vector_dim: int = 64,
        n_layers: int = 5,
        cutoff: float = 5.0,
        n_outputs: int = 1,
        max_atomic_num: int = 100,
        readout: str = "mean",
    ):
        super().__init__()
        if readout not in ("mean", "sum"):
            raise ValueError(f"readout must be 'mean' or 'sum', got {readout!r}")
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        self.n_layers = n_layers
        self.cutoff = cutoff
        self.readout = readout

        self.embedding = nn.Embedding(max_atomic_num, hidden_dim)
        self.mp_layers = nn.ModuleList(
            [EquivariantMessagePassingLayer(hidden_dim, vector_dim, cutoff) for _ in range(n_layers)]
        )
        self.mixing_layers = nn.ModuleList(
            [ScalarVectorMixing(hidden_dim, vector_dim) for _ in range(n_layers)]
        )
        self.output_mlp = MLP(hidden_dim, hidden_dim, n_outputs)

    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch
        if batch is None:
            batch = torch.zeros_like(z)
        edge_index = radius_graph(pos, batch, self.cutoff)

        h = self.embedding(z)
        V = torch.zeros(h.size(0), 3, self.vector_dim, device=h.device, dtype=h.dtype)
        for mp_layer, mix_layer in zip(self.mp_layers, self.mixing_layers):
            h, V = mp_layer(h, pos, V, edge_index)
            h, V = mix_layer(h, V)

        atom_out = self.output_mlp(h)
        pool = global_mean_pool if self.readout == "mean" else global_add_pool
        return pool(atom_out, batch), atom_out

    def predict(self, data, compute_forces: bool = False) -> Dict[str, torch.Tensor]:
        """Return ``{"energy": [B, 1]}`` and, optionally, ``{"forces": [N, 3]}``."""
        if compute_forces:
            data.pos.requires_grad_(True)
        mol_out, _ = self(data)
        out = {"energy": mol_out[:, :1]}
        if compute_forces:
            out["forces"] = _forces(out["energy"], data.pos, create_graph=self.training)
        return out


class MultitaskCrossDomainModel(nn.Module):
    """Shared equivariant encoder with one output head per task."""

    def __init__(
        self,
        hidden_dim: int = 128,
        vector_dim: int = 64,
        n_layers: int = 5,
        cutoff: float = 5.0,
        task_dims: Optional[Mapping[str, int]] = None,
        readout: str = "mean",
    ):
        super().__init__()
        task_dims = dict(task_dims or DEFAULT_TASK_DIMS)
        if "energy" not in task_dims:
            raise ValueError("task_dims must include 'energy'")
        self.task_dims = task_dims
        self.encoder = CrossDomainEquivariantNet(
            hidden_dim=hidden_dim,
            vector_dim=vector_dim,
            n_layers=n_layers,
            cutoff=cutoff,
            n_outputs=hidden_dim,
            readout=readout,
        )
        self.task_heads = nn.ModuleDict(
            {task: MLP(hidden_dim, hidden_dim // 2, dim) for task, dim in task_dims.items()}
        )

    def forward(self, data, tasks=None) -> Dict[str, torch.Tensor]:
        features, _ = self.encoder(data)
        tasks = list(self.task_dims) if tasks is None else tasks
        return {task: self.task_heads[task](features) for task in tasks}

    def predict(self, data, compute_forces: bool = False) -> Dict[str, torch.Tensor]:
        """Predict every task; forces are ``-dE/dpos`` of the energy head."""
        if compute_forces:
            data.pos.requires_grad_(True)
        out = self(data)
        if compute_forces:
            out["forces"] = _forces(out["energy"], data.pos, create_graph=self.training)
        return out


def _safe_norm(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Euclidean norm with finite first and second derivatives at zero.

    Identical to ``torch.norm`` for non-zero vectors. Zero vectors occur for
    atoms without neighbours inside the cutoff, where ``torch.norm`` would
    produce NaN gradients during force training.
    """
    sq = x.pow(2).sum(dim)
    nonzero = sq > 0
    return torch.where(nonzero, torch.where(nonzero, sq, torch.ones_like(sq)).sqrt(), torch.zeros_like(sq))


def _forces(energy: torch.Tensor, pos: torch.Tensor, create_graph: bool) -> torch.Tensor:
    (grad,) = torch.autograd.grad(energy.sum(), pos, create_graph=create_graph)
    return -grad


def build_model(config: Mapping) -> nn.Module:
    """Build a model from a full experiment config or its ``"model"`` section."""
    cfg = config.get("model", config)
    common = dict(
        hidden_dim=cfg.get("hidden_dim", 128),
        vector_dim=cfg.get("vector_dim", 64),
        n_layers=cfg.get("n_layers", 5),
        cutoff=cfg.get("cutoff", 5.0),
        readout=cfg.get("readout", "mean"),
    )
    if cfg.get("multitask", False):
        return MultitaskCrossDomainModel(task_dims=cfg.get("task_dims"), **common)
    return CrossDomainEquivariantNet(n_outputs=cfg.get("n_outputs", 1), **common)


def backbone_parameters(model: nn.Module):
    """Parameters of the shared encoder (everything except output heads)."""
    encoder = model.encoder if isinstance(model, MultitaskCrossDomainModel) else model
    for name, param in encoder.named_parameters():
        if not name.startswith("output_mlp"):
            yield param
