import jax
import pandas as pd
from qsense.utils.io import IO

from optimize_qfi import optimize_qfi


if __name__ == "__main__":
    io = IO(folder="qfi-batch-optimization", include_date=True, include_id=True)

    df = []
    n_runs = 3
    n_layers = 1
    r = [
        (2, 2),
        # (2, 3),
        # (2, 4),
        # (4, 2),
        # (4, 3),
        # (4, 4),
        # (6, 2),
        # (6, 2),
        # (8, 2),
    ]
    for (n, d) in r:
        for run in range(n_runs):
            print(f"n = {n}, d = {d} | run = {run}")
            lr = 0.2
            n_steps = 50
            losses = optimize_qfi(n, d, n_layers=n_layers, lr=lr, n_steps=n_steps)

            df.append(dict(
                n=n,
                d=d,
                n_layers=1,
                run=run,
                lr=lr,
                n_steps=n_steps,
                losses=losses,
                qfi=losses[-1],
            ))
        io.save_dataframe(pd.DataFrame(df), filename=f"qfi-batch")

    df = pd.DataFrame(df)
    io.save_dataframe(df, filename=f"qfi-batch")
