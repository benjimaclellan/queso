import pandas as pd
from queso.utils.io import IO

from queso.quantities.information import neg_qfi, neg_cfi
from optimize_fi import optimize_fi


if __name__ == "__main__":
    io = IO(folder="cfi-qfi-batch", include_date=True, include_id=True)

    df = []
    n_runs = 3
    r = [
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
        (6, 2),
        (7, 2),
        (8, 2),
        (9, 2),
    ]
    lr = 0.1
    n_steps = 400
    n_layers = 8
    for (n, d) in r:
        print(f"n = {n}, d = {d}")
        for name, fi in (("cfi", neg_cfi), ("qfi", neg_qfi)):
            losses, _ = optimize_fi(
                n=n,
                fi=neg_cfi,
                n_layers=n_layers,
                n_runs=n_runs,
                n_steps=n_steps,
                lr=lr,
                progress=True,
            )

            for run, loss in enumerate(losses):
                df.append(
                    dict(
                        n=n,
                        d=d,
                        run=run,
                        n_layers=n_layers,
                        lr=lr,
                        n_steps=n_steps,
                        loss=loss,
                        max=loss[-1],
                        device=loss.device().device_kind,
                        fi=name,
                    )
                )
            io.save_dataframe(pd.DataFrame(df), filename=f"batch")

    df = pd.DataFrame(df)
    io.save_dataframe(df, filename=f"batch")
