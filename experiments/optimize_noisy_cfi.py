"""
Batch runs for tensor circuit simulations of quantum sensors
Parameters to vary:
    n: number of qubits
    k: number of layers
    gamma: dephasing/deploaring coeefficient
    contractor: tensor contraction algorithm

Data to save:
    machine metadata
    learning curve
    run parameters (n, k, gamma)

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


def optimize_run(n, k, gammas, n_steps=200, contractor="greedy", seed=0, lr=0.25, repeat=5, progress=True):

    def cfi(_params, _phi, _gamma):
        def probs(_params, _phi, _gamma):
            return backend.abs(backend.diagonal(sensor(_params, _phi, _gamma).densitymatrix()))
        pr = probs(_params, _phi, _gamma)
        dpr_phi = backend.jacrev(lambda _phi: probs(_params=_params, _phi=_phi, _gamma=_gamma))
        d_pr = dpr_phi(phi).squeeze()
        fim = backend.sum(d_pr * d_pr / pr)
        return fim

    def neg_cfi(_params, _phi, _gamma):
        return -cfi(_params, _phi, _gamma)

    def sensor(params, phi, gamma):
        dmc = tc.DMCircuit(n)

        for i in range(k):
            for j in range(n):
                dmc.r(j, theta=params[3 * j, i], alpha=params[3 * j + 1, i], phi=params[3 * j + 2, i])

            for j in range(1, n, 2):
                dmc.cnot(j-1, j)

            for j in range(2, n, 2):
                dmc.cnot(j-1, j)

            for j in range(n):
                dmc.phasedamping(j, gamma=gamma[0])
                # dmc.depolarizing(j, px=gamma[0], py=gamma[0], pz=gamma[0])

        # interaction
        for j in range(n):
            dmc.rz(j, theta=phi[0])

        # measurement
        for j in range(n):
            dmc.u(j, theta=params[3 * j, -1], phi=params[3 * j + 1, -1])

        return dmc

    phi = np.array([0.0])
    gamma = np.array([0.0])
    key = random.PRNGKey(seed)
    params = random.uniform(key, ([3 * n, k + 1]))
    dmc = sensor(params, phi, gamma)

    # %%
    # cfi_val_grad_jit = backend.jit(backend.value_and_grad(neg_cfi, argnums=0))
    t0 = time.time()
    cfi_val_grad_jit = jax.jit(jax.value_and_grad(neg_cfi, argnums=0))
    val, grad = cfi_val_grad_jit(params, phi, gamma)
    print(f"Time to compile {time.time() - t0}")

    # print(dmc.draw(output="text"))
    # print(val, grad)

    def _optimize(gamma, n_steps=250, lr=0.25, progress=True, subkey=None):
        opt = tc.backend.optimizer(optax.adagrad(learning_rate=lr))
        # params = backend.implicit_randn([3 * n, k + 1])
        params = random.uniform(subkey, ([3 * n, k + 1]))

        loss = []
        t0 = time.time()
        for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
            val, grad = cfi_val_grad_jit(params, phi, gamma)
            params = opt.update(grad, params)
            loss.append(val)
            if progress:
                pbar.set_description(f"Cost: {-val:.10f}")
        t = time.time() - t0
        return -val, -np.array(loss), t

    # %%
    df = []
    for gamma in gammas:
        print(f"\nOptimizing circuit: n={n}, k={k}, gamma={gamma}")
        plt.pause(0.01)
        _loss = []
        for j in range(repeat):
            val, loss, t = _optimize(np.array([gamma]), n_steps=n_steps, lr=lr, progress=progress, subkey=subkey)

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
    parser.add_argument('--folder', type=str, default="cfi-tensor-noisy", help='Default save diretoty ')
    parser.add_argument('--n', type=int, default=2, help='Number of qubits')
    parser.add_argument('--k', type=int, default=2, help='Number of layers')
    parser.add_argument('--seed', type=int, default=100, help='An optional integer argument: seed for RNG')
    args = parser.parse_args()

    folder = args.folder
    n = args.n
    k = args.k
    seed = args.seed
    print(f"Beginning optimization for n={n}, k={k}. Save folder: {folder}")

    io = IO(folder=args.folder, include_date=True, include_id=False)

    lr = 0.25
    repeat = 7
    progress = True
    n_steps = 250
    gammas = np.hstack([np.array([0.0]), np.exp(np.linspace(-5, -1, 11))])

    df = optimize_run(n, k, gammas, n_steps=n_steps, contractor="greedy", lr=lr, repeat=repeat, progress=progress, seed=seed)
    io.save_dataframe(df, filename=f"n={n}_k={k}")
