import itertools
import torch
from jax import numpy as jnp
import numpy as np
from prettytable import PrettyTable


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


def count_parameters(model, verbose=True):  # todo: move to utils
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if verbose:
        print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
