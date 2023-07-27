from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, asdict
import yaml
import pathlib
import jax.numpy as jnp

from queso.io import IO


# %%
@dataclass
class Configuration:
    folder: str = None
    seed: int = None

    train_circuit: bool = True
    sample_circuit: bool = True
    train_nn: bool = True
    benchmark_estimator: bool = True

    # circuit args
    n: int = 2
    k: int = 2

    preparation: str = 'brick_wall_cr'
    interaction: str = "local_rx"
    detection: str = 'local_r'
    backend: str = 'ket'

    # training circuit args
    n_phis: int = 100
    n_steps: int = 20000
    lr_circ: float = 1e-3
    phi_range: list[float] = field(default_factory=lambda: [-1.157, 1.157])

    # sample circuit args
    n_shots: int = 5000
    n_shots_test: int = 1000

    # train estimator args
    n_epochs: int = 1000
    batch_size: int = 50
    n_grid: int = 100  # todo: make more general - not requiring matching training phis and grid
    nn_dims: list[int] = field(default_factory=lambda: [32, 32])
    lr_nn: float = 1e-3
    from_checkpoint: bool = False

    # benchmark estimator args
    n_trials: int = 100
    phis_inds: list[int] | jnp.ndarray = field(default_factory=lambda: [50])
    n_sequences: list[int] | jnp.ndarray = field(default_factory=lambda: [1, 10, 100, 1000])

    @classmethod
    def from_yaml(cls, file):
        with open(file, "r") as fid:
            data = yaml.safe_load(fid)
        return cls(**data)

    def __post_init__(self):
        if self.folder is None:
            self.folder = IO(folder="data", include_date=True, include_id=True).path.name
        if self.seed is None:
            self.seed = time.time_ns()

        if self.n_grid != self.n_phis:
            raise Warning("should be the same")

        # convert all lists to jax.numpy arrays
        # for field in fields(self.__class__):
        #     val = getattr(self, field.name)
        #     if isinstance(val, list):
        #         val = jnp.array(val)
        #         setattr(self, field.name, val)

    def to_yaml(self, file):
        data = asdict(self)
        for key, val in data.items():
            if isinstance(val, jnp.ndarray):
                data[key] = val.tolist()
        with open(file, "w") as fid:
            yaml.dump(data, fid)


#%%
if __name__ == "__main__":
    #%%
    d = Configuration()
    print(d)
    file = pathlib.Path(__file__).parent.joinpath("default.yaml")
    d = Configuration.from_yaml(file)
    print(d)

    #%%
    file = pathlib.Path(__file__).parent.joinpath("run1.yaml")
    d.to_yaml(file)

    #%%
    file = pathlib.Path(__file__).parent.joinpath("run1.yaml")

    d = Configuration.from_yaml(file)
    print(d)
    #%%