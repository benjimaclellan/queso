import pandas as pd
import tqdm
import qutip as qt
import jax.numpy as np
import numpy as onp
import jax
import optax
from functools import partial

from qsense.io import IO
from benchmarks.circuits import nketz0, local_unitary_circuit, nonlocal_entangling_circuit, \
    local_phase_interaction_circuit
from qsense.entanglement import genuine_multipartite_entanglement
from optimize_fischer_information import optimize_qfi, initialize
from qsense.gates import Phase
from qsense.functions import dagger, compile
from qsense.cfi import cfim


if __name__ == "__main__":

    n, d = 6, 2

    ket_i = nketz0(n=n, d=d)

    probe_circ = nonlocal_entangling_circuit(n, d)
    probe_params = initialize(probe_circ)

    interaction_circ = [[Phase("phase", d=d) for _ in range(n)]]
    interaction_params = initialize(interaction_circ)
    interaction_params["phase"] = np.array([0.0])

    measure_circ = local_unitary_circuit(n, d)
    measure_params = initialize(measure_circ)
    proj_mu = nketz0(n=n, d=d)

    keys = ["phase"]

    #%%
    lr = 0.2
    n_steps = 300

    params = probe_params | interaction_params | measure_params

    def cfi(params, probe_circ, interaction_circ, measure_circ, ket_i, proj_mu, keys):
        return -cfim(params, probe_circ, interaction_circ, measure_circ, ket_i, proj_mu, keys)[0, 0]
    cfi = jax.jit(partial(cfi, probe_circ=probe_circ, interaction_circ=interaction_circ, measure_circ=measure_circ, ket_i=ket_i, proj_mu=proj_mu, keys=keys))

    optimizer = optax.adagrad(learning_rate=lr)
    opt_state = optimizer.init(params)

    losses = []
    grad = jax.jit(jax.grad(cfi))
    progress = True
    for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
        ell = cfi(params)
        gradient = grad(params)
        updates, opt_state = optimizer.update(gradient, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(ell)

        if progress:
            pbar.set_description(f"Cost: {-ell:.10f}")

    losses = -np.array(losses)
