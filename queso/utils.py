import itertools
import torch
from jax import numpy as jnp
import numpy as np
from prettytable import PrettyTable
import psutil
import platform
from datetime import datetime
import GPUtil


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


def bit_to_integer(a, endian='le'):
    if endian == 'le':
        k = 1 << jnp.arange(a.shape[-1] - 1, -1, -1)  # little-endian
    elif endian == 'be':
        k = 1 << jnp.arange(a.shape[-1] - 1, -1, -1)
    else:
        raise NotImplementedError
    s = jnp.einsum('ijk,k->ij', a, k)
    return jnp.expand_dims(s, 2)


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_machine_info():
    # https://www.thepythoncode.com/article/get-hardware-system-information-python

    uname = platform.uname()
    svmem = psutil.virtual_memory()
    gpus = GPUtil.getGPUs()

    info = dict(
        system=uname.system,
        name=uname.node,
        processor=uname.processor,
        version=uname.version,
        #
        cpus_physical=psutil.cpu_count(logical=False),
        cpus_logical=psutil.cpu_count(logical=True),
        #
        mem=get_size(svmem.total),
    )

    if gpus:
        gpu = gpus[0]
        info.update(
            dict(
                gpu_id=gpu.id,
                gpu_name=gpu.name,
                gpu_total_memory=gpu.memoryTotal,
            )
        )

    return info
