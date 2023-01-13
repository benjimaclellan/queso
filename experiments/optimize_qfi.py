import optax
import tqdm
from functools import partial
import matplotlib.pyplot as plt
import jax.numpy as np

from qsense.qfi import qfim
from qsense.io import IO
from benchmarks.circuits import *

device = jax.devices('cpu')[0]


def optimize_qfi(n, d, n_layers=1, lr=0.2, n_steps=100, progress=True):
    ket_i = nketz0(n=n, d=d)

    circuit = local_entangling_circuit(n, d, n_layers=n_layers)
    # circuit = nonlocal_entangling_circuit(n, d)

    circuit.append([Phase("phase", d=d) for _ in range(n)])

    params = initialize(circuit)
    params["phase"] = np.array([0.0])
    keys = ["phase"]
    #
    # # note the negative for minimizing
    # qfi = lambda params, circuit, ket_i, keys: -qfim(params, circuit, ket_i, keys)[0, 0]
    # qfi = jax.jit(partial(qfi, circuit=circuit, ket_i=ket_i, keys=keys), device=device)
    #
    # optimizer = optax.adagrad(learning_rate=lr)
    # opt_state = optimizer.init(params)
    #
    # losses = []
    # grad = jax.jit(jax.grad(qfi), device=device)
    # gradient = grad(params)
    #
    # print(qfi(params).device())
    #
    # for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
    #     ell = qfi(params)
    #     gradient = grad(params)
    #     updates, opt_state = optimizer.update(gradient, opt_state)
    #     params = optax.apply_updates(params, updates)
    #     losses.append(ell)
    #
    #     if progress:
    #         pbar.set_description(f"Cost: {-ell:.10f}")
    #     # else:
    #     #     print(step, ell, params)
    #
    # losses = -np.array(losses)
    # return circuit, params, losses
    return circuit, params, []


if __name__ == "__main__":
    io = IO(folder="qfi-optimization", include_date=True)

    n = 7  # number of particles
    d = 2
    n_layers = 4
    lr = 0.1
    n_steps = 200
    circuit, params, losses = optimize_qfi(n, d, n_layers=n_layers, lr=lr, n_steps=n_steps)

    #%%
    fig, axs = plt.subplots(1, 1)
    axs.axhline(n**d, **dict(color="teal", ls="--"))
    axs.plot(losses, **dict(color="salmon", ls="-"))
    axs.set(
        xlabel="Optimization step",
        ylabel=r"Quantum Fischer Information: $\mathcal{F}_\phi$",
    )

    plt.show()

    # io.save_figure(fig, filename=f"qfi_n={n}")
