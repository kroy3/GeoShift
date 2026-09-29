import pytest
import torch
from torch_geometric.data import Batch, Data

from geoshift.evaluate import random_rotation
from geoshift.model import build_model, radius_graph


def make_batch(sizes=(5, 8), seed=0):
    gen = torch.Generator().manual_seed(seed)
    mols = [
        Data(
            z=torch.randint(1, 9, (n,), generator=gen),
            pos=torch.randn(n, 3, generator=gen, dtype=torch.float64) * 1.5,
        )
        for n in sizes
    ]
    return Batch.from_data_list(mols)


SMALL = {"hidden_dim": 32, "vector_dim": 16, "n_layers": 3, "cutoff": 5.0}


@pytest.fixture(params=[False, True], ids=["single", "multitask"])
def model(request):
    torch.manual_seed(0)
    return build_model({"model": {**SMALL, "multitask": request.param}}).double().eval()


def test_build_model_accepts_full_or_model_config():
    a = build_model({"model": SMALL})
    b = build_model(SMALL)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_radius_graph_stays_within_molecules_and_cutoff():
    batch = make_batch()
    edges = radius_graph(batch.pos, batch.batch, cutoff=2.0)
    src, dst = edges
    assert (batch.batch[src] == batch.batch[dst]).all()
    assert (src != dst).all()
    assert ((batch.pos[src] - batch.pos[dst]).norm(dim=-1) < 2.0).all()


def test_output_shapes(model):
    batch = make_batch()
    out = model.predict(batch, compute_forces=True)
    assert out["energy"].shape == (2, 1)
    assert out["forces"].shape == (batch.num_nodes, 3)


def test_rotation_translation_equivariance(model):
    batch = make_batch()
    rot = random_rotation(torch.Generator().manual_seed(1), dtype=torch.float64)
    ref = model.predict(batch.clone(), compute_forces=True)

    moved = batch.clone()
    moved.pos = batch.pos @ rot.T + torch.tensor([[0.3, -1.0, 2.0]], dtype=torch.float64)
    out = model.predict(moved, compute_forces=True)

    torch.testing.assert_close(out["energy"], ref["energy"])
    torch.testing.assert_close(out["forces"], ref["forces"] @ rot.T)


def test_forces_are_negative_energy_gradient(model):
    batch = make_batch(sizes=(4,))
    forces = model.predict(batch.clone(), compute_forces=True)["forces"]

    eps, atom, axis = 1e-5, 1, 2
    plus, minus = batch.clone(), batch.clone()
    plus.pos = batch.pos.clone()
    minus.pos = batch.pos.clone()
    plus.pos[atom, axis] += eps
    minus.pos[atom, axis] -= eps
    with torch.no_grad():
        e_plus = model.predict(plus)["energy"].sum()
        e_minus = model.predict(minus)["energy"].sum()
    numeric = -(e_plus - e_minus) / (2 * eps)
    torch.testing.assert_close(forces[atom, axis], numeric, rtol=1e-4, atol=1e-6)


def test_isolated_atom_gives_finite_force_gradients(model):
    model.train()
    mol = Data(z=torch.tensor([6, 1, 8]), pos=torch.tensor([[0.0, 0, 0], [1.1, 0, 0], [20.0, 0, 0]]))
    batch = Batch.from_data_list([mol])
    batch.pos = batch.pos.double()
    forces = model.predict(batch, compute_forces=True)["forces"]
    forces.pow(2).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_gradients_are_finite(model):
    model.train()
    batch = make_batch()
    out = model.predict(batch, compute_forces=True)
    (out["energy"].sum() + out["forces"].pow(2).sum()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
