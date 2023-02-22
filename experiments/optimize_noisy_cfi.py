"""
Batch runs for tensor circuit simulations of quantum sensors
Parameters to vary:
    n: number of qubits
    k: number of layers

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
import optax
import pandas as pd
import matplotlib.pyplot as plt
import time

from queso.io import IO
from queso import sensors
from queso.quantities import classical_fisher_information

backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("auto")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Parser for starting multiple runs on Graham"
    # )
    # # Optional argument
    # parser.add_argument(
    #     "--folder", type=str, default="qfi-tensor-pure", help="Default save directory "
    # )
    # parser.add_argument("--n", type=int, default=2, help="Number of qubits")
    # parser.add_argument("--k", type=int, default=2, help="Number of layers")
    # parser.add_argument("--ansatz", type=str, default=None, help="Number of layers")
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=None,
    #     help="An optional integer argument: seed for RNG",
    # )
    # args = parser.parse_args()
    #
    # if args.ansatz is None:
    #     raise ValueError("Ansatz is required")
    #
    # folder = args.folder
    # n = args.n
    # k = args.k
    # ansatz = args.ansatz
    # seed = args.seed if args.seed is not None else time.time_ns()
    n = 4
    k = 4
    ansatz = "cnot_2local_dephased_ansatz"
    seed = 0
    folder = "noisy_state"
    io = IO(folder=folder, include_date=False, include_id=False).subpath(ansatz)

    lr = 0.20
    repeat = 3
    progress = True
    n_steps = 400
    n_samples = 1000
    fi_name = "cfi"

    #%%
    circ, shape = sensors.build(ansatz, n, k)

    phi = 0.0
    key = random.PRNGKey(seed)
    theta = random.uniform(key, shape, minval=0, maxval=2*np.pi)
    gammas = np.logspace(-5, -0.25, 15)

    fi_val_grad_jit = backend.jit(
        backend.value_and_grad(
            lambda _theta, _gamma: classical_fisher_information(circ=circ, theta=_theta, phi=phi, gamma=_gamma, n=n, k=k),
            argnums=0,
        )
    )
    val, grad = fi_val_grad_jit(theta, gammas[0])
    print(-val, -grad)

    #%% optimize the sensor circuit `repeat` times
    def _optimize(gamma, n_steps=250, lr=0.25, progress=True, subkey=None):
        opt = tc.backend.optimizer(optax.adagrad(learning_rate=lr))
        theta = random.uniform(subkey, shape, minval=0, maxval=2*np.pi)
        loss = []
        t0 = time.time()
        for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
            val, grad = fi_val_grad_jit(theta, gamma)
            theta = opt.update(grad, theta)
            loss.append(val)
            if progress:
                pbar.set_description(f"Cost: {-val:.10f}")
        t = time.time() - t0
        return -val, -np.array(loss), t

    df = []
    for j in range(repeat):
        key, subkey = random.split(key)
        for gamma in gammas:
            print(f"\nOptimizing circuit: n={n}, k={k}, gamma={gamma}, repeat={j}")
            val, loss, t = _optimize(
                gamma=gamma, n_steps=n_steps, lr=lr, progress=progress, subkey=subkey
            )

            df.append(
                dict(
                    n=n,
                    k=k,
                    gamma=gamma,
                    fi=val,
                    loss=loss,
                    time=t,
                    lr=lr,
                    n_steps=n_steps,
                    fi_name=fi_name,
                )
            )

            # save after each repeat, in case of runtime error
            io.save_dataframe(pd.DataFrame(df), filename=f"optimization/n={n}_k={k}")
            plt.pause(0.01)

    #%% sample FI and gradient vectors
    def _sample(n_samples=250, progress=True, key=None):
        df = []
        for i, gamma in enumerate(gammas):
            vals, grads = [], []
            _key = key
            t0 = time.time()
            for sample in (pbar := tqdm.tqdm(range(n_samples), disable=(not progress))):
                _key, subkey = random.split(_key)
                theta = random.uniform(subkey, shape, minval=0, maxval=2*np.pi)
                val, grad = fi_val_grad_jit(theta, gamma)

                vals.append(val)
                grads.append(grad)

                if progress:
                    pbar.set_description(f"Cost: {-val:.10f} | Gamma = 10e{np.log10(gamma):.5f}")
            t = time.time() - t0

            df.append(
                dict(
                    n=n,
                    k=k,
                    gamma=gamma,
                    vals=-np.array(vals),
                    grads=-np.array(grads),
                    fi_name=fi_name,
                    t=t,
                )
            )

        return pd.DataFrame(df)

    print("\nSampling FI and gradients.")
    df = _sample(n_samples=n_samples, progress=True, key=key)
    io.save_dataframe(df, filename=f"samples/n={n}_k={k}")
