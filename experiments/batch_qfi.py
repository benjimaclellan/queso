import jax
import optax
import uuid
import tqdm
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd

from qsense.functions import *
from qsense.functions import initialize, compile
from qsense.qfi import qfim
from qsense.io import IO
from experiments.circuits import *

from optimize_fischer_information import optimize_qfi


if __name__ == "__main__":
    io = IO(folder="qfi-batch-optimization", include_date=True, include_id=True)

    df = []
    for n in (2, 4):
        for d in (2, 3):
            for n_layers in (1,):
                print(n, d, n_layers)
                lr = 0.2
                n_steps = 100
                circuit, params, losses = optimize_qfi(n, d, n_layers=1, lr=lr, n_steps=n_steps)

                df.append(dict(
                    n=n,
                    d=d,
                    n_layers=n_layers,
                    lr=lr,
                    n_steps=n_steps,
                    losses=losses,
                ))
    df = pd.DataFrame(df)
    io.save_dataframe(df, filename=f"qfi-batch")
