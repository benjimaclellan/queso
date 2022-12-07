import jax
import optax
import uuid
import tqdm
from functools import partial
import matplotlib.pyplot as plt

from qsense.functions import *
from qsense.functions import initialize, compile
from qsense.qfi import qfim
from qsense.io import IO
from examples.circuits import *


if __name__ == "__main__":
    io = IO(folder="qfi-optimization", include_date=True)

    n = 4  # number of particles

    ket_i = nketz0(n)
    circuit = local_entangling_circuit(n, n_layers=8)
    # circuit = nonlocal_entangling_circuit(n)
    circuit.append([Phase("phase") for i in range(n)])

    params = initialize(circuit)
    params['phase'] = np.array([0.0])

    compile = jax.jit(partial(compile, circuit=circuit))
    keys = ["phase"]

    # note the negative for minimizing
    qfi = lambda params, circuit, ket_i, keys: -qfim(params, circuit, ket_i, keys)[0, 0]
    qfi = jax.jit(partial(qfi, circuit=circuit, ket_i=ket_i, keys=keys))

    learning_rate = 0.2
    n_step = 100
    progress = True
    optimizer = optax.adagrad(learning_rate)
    opt_state = optimizer.init(params)

    losses = []
    grad = jax.jit(jax.grad(qfi))
    print(grad(params))

    for step in (
            pbar := tqdm.tqdm(range(n_step), disable=(not progress))
    ):
        ell = qfi(params)
        gradient = grad(params)
        updates, opt_state = optimizer.update(gradient, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(ell)

        if progress:
            pbar.set_description(f"Cost: {-ell:.10f}")
        else:
            print(step, ell, params)

    losses = np.array(losses)
    print(losses)

    #%%
    fig, axs = plt.subplots(1, 1)
    axs.axhline(n**2, **dict(color="teal", ls="--"))
    axs.plot(-losses, **dict(color="salmon", ls='-'))
    axs.set(xlabel="Optimization step", ylabel=r"Quantum Fischer Information: $\mathcal{F}_\phi$")

    plt.show()

    # io.save_figure(fig, filename=f"qfi_n={n}")
