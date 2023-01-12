import pandas as pd
from qsense.io import IO

from optimize_cfi import optimize_cfi


if __name__ == "__main__":
    io = IO(folder="cfi-batch-optimization", include_date=True, include_id=True)

    df = []
    n_runs = 3
    r = [
        (2, 2),
        (4, 2),
        (6, 2),
        (8, 2),
    ]
    for (n, d) in r:
        print(f"n = {n}, d = {d}")
        lr = 0.15
        n_steps = 300
        _losses = optimize_cfi(n, d, lr=lr, n_steps=n_steps, n_runs=n_runs, progress=True)

        for run, losses in enumerate(_losses):

            df.append(dict(
                n=n,
                d=d,
                run=run,
                lr=lr,
                n_steps=n_steps,
                losses=losses,
                cfi=losses[-1],
            ))
        io.save_dataframe(pd.DataFrame(df), filename=f"cfi-batch")

    df = pd.DataFrame(df)
    io.save_dataframe(df, filename=f"cfi-batch")
