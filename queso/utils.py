import itertools
import torch
from jax import numpy as jnp
import numpy as np


def shots_to_counts(
        shots,
        phis
):
    if isinstance(shots, torch.Tensor):
        shots = shots.numpy().astype('int8')

    assert isinstance(shots, np.ndarray)
    bin_str = ["".join(p) for p in itertools.product("01", repeat=shots.shape[2])]
    counts = []
    for i, phi in enumerate(phis):
        basis, count = jnp.unique(shots[i, :, :], return_counts=True, axis=0)
        c = {
            "".join([str(j) for j in basis[i]]): count[i].item()
            for i in range(len(count))
        }
        cts = [c.get(b, 0) for b in bin_str]
        counts.append(cts)
    counts = jnp.array(
        counts
    )  # / shots.shape[1]  # normalize data to get relative frequency
    return counts
