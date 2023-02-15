"""
Batch runs for tensor circuit simulations of quantum sensors
Parameters to vary:
    n: number of qubits
    k: number of layers
    contractor: tensor contraction algorithm

Data to save:
    machine metadata
    learning curve
    run parameters (n, k)

"""

import itertools
import tensorcircuit as tc
import jax.numpy as jnp
import argparse
import jax
from jax import random
import tqdm
import seaborn as sns
import numpy as np
import optax
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys

sys.path.append(".")

from queso.utils.io import IO

#%%
backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("greedy")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”


def optimize_run(n, k, n_steps=200, seed=0, lr=0.25, repeat=5, progress=True):
    def qfi(_params, phi):
        psi = sensor(_params, phi).state()[:, None]
        f_dpsi_phi = backend.jacrev(lambda phi: sensor(params=_params, phi=phi).state())
        d_psi = f_dpsi_phi(phi)
        fi = 4 * backend.real((backend.conj(d_psi.T) @ d_psi) + (backend.conj(d_psi.T) @ psi) ** 2)
        return fi[0, 0]

    def neg_qfi(_params, _phi):
        return -qfi(_params, _phi)

    def sensor(params, phi):
        dmc = tc.Circuit(n)

        for i in range(k):
            for j in range(n):
                dmc.r(j, theta=params[3 * j, i], alpha=params[3 * j + 1, i], phi=params[3 * j + 2, i])

            for j in range(1, n, 2):
                dmc.cnot(j-1, j)

            for j in range(2, n, 2):
                dmc.cnot(j-1, j)

        # interaction
        for j in range(n):
            dmc.rz(j, theta=phi[0])

        return dmc

    phi = np.array([0.0])
    gamma = np.array([0.0])
    key = random.PRNGKey(seed)
    params = random.uniform(key, ([3 * n, k]))
    dmc = sensor(params, phi)

    # %%
    # cfi_val_grad_jit = backend.jit(backend.value_and_grad(neg_cfi, argnums=0))
    cfi_val_grad_jit = jax.jit(jax.value_and_grad(neg_qfi, argnums=0))
    val, grad = cfi_val_grad_jit(params, phi)

    def _optimize(n_steps=250, lr=0.25, progress=True, subkey=None):
        opt = tc.backend.optimizer(optax.adagrad(learning_rate=lr))
        params = random.uniform(subkey, ([3 * n, k]))

        loss = []
        t0 = time.time()
        for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
            val, grad = cfi_val_grad_jit(params, phi)
            params = opt.update(grad, params)
            loss.append(val)
            if progress:
                pbar.set_description(f"Cost: {-val:.10f}")
        t = time.time() - t0
        return -val, -np.array(loss), t

    # %%
    df = []
    print(f"\nOptimizing circuit: n={n}, k={k}")
    plt.pause(0.01)
    _loss = []
    for j in range(repeat):
        key, subkey = random.split(key)
        val, loss, t = _optimize(n_steps=n_steps, lr=lr, progress=progress, subkey=subkey)

        df.append(dict(
            n=n,
            k=k,
            gamma=gamma,
            cfi=val,
            loss=loss,
            time=t,
        ))

    return pd.DataFrame(df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parser for starting multiple runs on Graham')
    # Optional argument
    parser.add_argument('--folder', type=str, default="qfi-tensor-pure", help='Default save diretoty ')
    parser.add_argument('--n', type=int, default=2, help='Number of qubits')
    parser.add_argument('--k', type=int, default=2, help='Number of layers')
    parser.add_argument('--seed', type=int, default=None, help='An optional integer argument: seed for RNG')
    args = parser.parse_args()

    folder = args.folder
    n = args.n
    k = args.k
    seed = args.seed if args.seed is not None else time.time_ns()

    io = IO(folder=folder, include_date=False, include_id=False)

    lr = 0.20
    repeat = 11
    progress = True
    n_steps = 250

    df = optimize_run(n, k, n_steps=n_steps, lr=lr, repeat=repeat, progress=progress, seed=seed)
    io.save_dataframe(df, filename=f"n={n}_k={k}")
