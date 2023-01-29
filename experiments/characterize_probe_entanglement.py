import tqdm
import qutip as qt
import numpy as np

from queso.utils.io import IO
from benchmarks.circuits import nketz0
from queso.quantities.entanglement import genuine_multipartite_entanglement
from optimize_qfi import optimize_qfi, initialize


if __name__ == "__main__":
    io = IO(
        folder="qfi_entanglement_characteristics", include_date=True, include_id=True
    )

    df = []
    n_runs = 1
    r = [
        # (2, 2),
        # (4, 2),
        (6, 2),
    ]
    for (n, d) in r:
        # for run in range(n_runs):
        for run in (pbar := tqdm.tqdm(range(n_runs))):
            # pbar.set_description(f"n = {n}, d = {2} | run = {run}")

            lr = 0.2
            n_steps = 1
            circuit, params, losses = optimize_qfi(
                n, d, n_layers=1, lr=lr, n_steps=n_steps, progress=True
            )
            params = initialize(circuit)
            # final state
            ket_i = nketz0(n=n, d=d)
            u = compile(params, circuit)
            ket_f = np.array(u @ ket_i)

            rho = qt.ket2dm(qt.Qobj(np.asarray(ket_f), dims=[[d] * n, [1] * n]))

            genuine_multipartite_entanglement(rho)

    #         df.append(dict(
    #             n=n,
    #             d=2,
    #             n_layers=1,
    #             run=run,
    #             lr=lr,
    #             n_steps=n_steps,
    #             losses=losses,
    #             qfi=losses[-1],
    #         ))
    #     io.save_dataframe(pd.DataFrame(df), filename=f"qfi-batch")
    #
    # df = pd.DataFrame(df)
    # io.save_dataframe(df, filename=f"qfi-batch")
