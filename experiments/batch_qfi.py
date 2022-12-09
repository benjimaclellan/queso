import pandas as pd
from qsense.io import IO

from optimize_fischer_information import optimize_qfi


if __name__ == "__main__":
    io = IO(folder="qfi-batch-optimization", include_date=True, include_id=True)

    df = []
    r = [
        (2, 2),
        # (2, 3),
        # (2, 4),
        (4, 2),
        # (4, 3),
        # (4, 4),
        (6, 2),
        (8, 2),
    ]
    for (n, d) in r:
        for n_layers in (1,):
            print(n, d, n_layers)
            lr = 0.2
            n_steps = 50
            circuit, params, losses = optimize_qfi(n, d, n_layers=1, lr=lr, n_steps=n_steps)

            df.append(dict(
                n=n,
                d=d,
                n_layers=n_layers,
                lr=lr,
                n_steps=n_steps,
                losses=losses,
                qfi=losses[-1],
            ))
    df = pd.DataFrame(df)
    io.save_dataframe(df, filename=f"qfi-batch")
