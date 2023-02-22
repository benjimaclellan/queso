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

import tensorcircuit as tc
import argparse
import jax
from jax import random
import tqdm
import numpy as np
import pandas as pd
import time
import sys

sys.path.append(".")

from queso.io import IO

# %%
backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("auto")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”


def optimize_run(n, k, n_samples=200, seed=0, progress=True):
    def qfi(_params, phi):
        psi = sensor(_params, phi).state()[:, None]
        f_dpsi_phi = backend.jacrev(lambda phi: sensor(params=_params, phi=phi).state())
        d_psi = f_dpsi_phi(phi)
        fi = 4 * backend.real(
            (backend.conj(d_psi.T) @ d_psi) + (backend.conj(d_psi.T) @ psi) ** 2
        )
        return fi[0, 0]

    def neg_qfi(_params, _phi):
        return -qfi(_params, _phi)

    def sensor(params, phi):
        mps = tc.Circuit(n)
        # mps.set_split_rules({"max_singular_values": 2})

        for i in range(k):
            for j in range(n):
                mps.r(
                    j,
                    theta=params[3 * j, i],
                    alpha=params[3 * j + 1, i],
                    phi=params[3 * j + 2, i],
                )

            for j in range(1, n, 2):
                mps.cnot(j - 1, j)

            for j in range(2, n, 2):
                mps.cnot(j - 1, j)

        # interaction
        for j in range(n):
            mps.rz(j, theta=phi[0])

        return mps

    phi = np.array([0.0])
    gamma = np.array([0.0])
    key = random.PRNGKey(seed)
    params = random.uniform(key, ([3 * n, k]))

    # %%
    cfi_val_grad_jit = jax.jit(jax.value_and_grad(neg_qfi, argnums=0))
    _ = cfi_val_grad_jit(params, phi)

    def _sample_circuit(n_samples=250, progress=True, key=None):
        t0 = time.time()
        vals, grads = [], []

        for step in (pbar := tqdm.tqdm(range(n_samples), disable=(not progress))):
            key, subkey = random.split(key)
            params = random.uniform(subkey, ([3 * n, k]))

            val, grad = cfi_val_grad_jit(params, phi)
            vals.append(val)
            grads.append(grad)

            if progress:
                pbar.set_description(f"Cost: {-val:.10f}")
        t = time.time() - t0
        return -np.array(vals), -np.array(grads), t

    # %%
    df = []
    print(f"\nSampling from circuit: n={n}, k={k}")

    key, subkey = random.split(key)
    vals, grads, t = _sample_circuit(n_samples=n_samples, progress=progress, key=key)

    df.append(
        dict(
            n=n,
            k=k,
            gamma=gamma,
            fi_type=qfi.__name__,
            fi_vals=vals,
            fi_grad_vals=grads,
            time=t,
        )
    )

    return pd.DataFrame(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser for starting multiple runs on Graham"
    )
    # Optional argument
    parser.add_argument(
        "--folder", type=str, default="qfi-tensor-pure", help="Default save diretoty "
    )
    parser.add_argument("--n", type=int, default=2, help="Number of qubits")
    parser.add_argument("--k", type=int, default=2, help="Number of layers")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="An optional integer argument: seed for RNG",
    )
    args = parser.parse_args()

    folder = args.folder
    n = args.n
    k = args.k
    seed = args.seed if args.seed is not None else time.time_ns()

    io = IO(folder=folder, include_date=False, include_id=False)

    progress = True
    n_samples = 500

    df = optimize_run(n, k, n_samples=n_samples, progress=progress, seed=seed)
    io.save_dataframe(df, filename=f"n={n}_k={k}")
