import tqdm
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

from qsense.qfi import qfim
from qsense.io import IO
from benchmarks.circuits import *


if __name__ == "__main__":
    io = IO(folder="qfi-sample-parameters", include_date=True, include_id=False)

    n = 6  # number of particles
    d = 2
    n_samples = 10000

    ket_i = nketz0(n=n, d=d)

    # circuit = local_entangling_circuit(n, d, n_layers=1)
    circuit = nonlocal_entangling_circuit(n, d)
    circuit.append([Phase("phase", d=d) for _ in range(n)])

    params = initialize(circuit)
    params["phase"] = np.array([0.0])
    keys = ["phase"]

    qfi = lambda params, circuit, ket_i, keys: qfim(params, circuit, ket_i, keys)[0, 0]
    qfi = jax.jit(partial(qfi, circuit=circuit, ket_i=ket_i, keys=keys))

    qfis = []
    for m in (pbar := tqdm.tqdm(range(n_samples))):
        params = initialize(circuit)
        qfis.append(qfi(params))
        pbar.set_description(f"Sample: {m:d}")

    qfis = np.array(qfis)

    fig, ax = plt.subplots(1, 1)
    sns.histplot(qfis, ax=ax)
    ax.set(
        xlabel=r"QFI, $\mathcal{F}(\varphi)$",
        ylabel=r"Count",
    )
    fig.suptitle(f"n={n}, d={d}")
    plt.show()

    io.save_figure(fig, filename=f"qfi_n_samples={n_samples}")
