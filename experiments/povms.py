import pandas as pd
import tqdm
import qutip as qt
import jax.numpy as np
import numpy as onp

from qsense.io import IO
from benchmarks.circuits import nketz0, local_unitary_circuit, nonlocal_entangling_circuit
from qsense.entanglement import genuine_multipartite_entanglement
from optimize_fischer_information import optimize_qfi, initialize
from qsense.gates import Phase
from qsense.functions import dagger


if __name__ == "__main__":

    n, d = 4, 2
    lr = 0.2
    n_steps = 1

    ket_i = nketz0(n=n, d=d)
    circuit = nonlocal_entangling_circuit(n, d)
    circuit.append([Phase("phase", d=d) for _ in range(n)])

    povm = local_unitary_circuit(n, d)
    proj_l = np.zeros([1, d ** n])
    proj_l.at[0, 0].set(1.0)

    p_l = np.abs(dagger(proj_l)) ** 2

    circ_params = initialize(circuit)
    params["phase"] = np.array([0.0])
    keys = ["phase"]

    # final state
    ket_i = nketz0(n=n, d=d)
    u = compile(params, circuit)
    ket_f = np.array(u @ ket_i)

    # rho = qt.ket2dm(qt.Qobj(onp.asarray(ket_f), dims=[[d] * n, [1] * n]))
    # genuine_multipartite_entanglement(rho)
