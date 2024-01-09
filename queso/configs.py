from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, asdict
import networkx as nx
import yaml
import pathlib


# %%
@dataclass
class Configuration:
    folder: str = "tmp"
    seed: int = None

    train_circuit: bool = True
    sample_circuit: bool = True
    sample_circuit_training_data: bool = True
    sample_circuit_testing_data: bool = True
    train_nn: bool = True
    benchmark_estimator: bool = True

    # circuit args
    n: int = 2
    k: int = 2

    preparation: str = "brick_wall_cr"
    interaction: str = "local_rx"
    detection: str = "local_r"
    loss_fi: str = "loss_cfi"
    backend: str = "ket"

    # optional circuit args
    gamma_dephasing: float = 0.0
    n_ancilla: int = 0

    # training circuit args
    n_phis: int = 100
    n_steps: int = 20000
    lr_circ: float = 1e-3

    phi_offset: float = 0.0
    phi_range: list[float] = field(default_factory=lambda: [-1.157, 1.157])

    # sample circuit args
    n_shots: int = 5000
    n_shots_test: int = 1000
    phis_test: list = field(default_factory=lambda: [0, 1.157])

    # train estimator args
    n_epochs: int = 1000
    batch_size: int = 50
    n_grid: int = (
        100  # todo: make more general - not requiring matching training phis and grid
    )
    nn_dims: list[int] = field(default_factory=lambda: [32, 32])
    lr_nn: float = 1e-3
    l2_regularization: float = 0.0  # L2 regularization for NN estimator
    from_checkpoint: bool = False

    # benchmark estimator args
    n_trials: int = 100
    phis_inds: list[int] = field(default_factory=lambda: [50])
    n_sequences: list[int] = field(default_factory=lambda: [1, 10, 100, 1000])

    @classmethod
    def from_yaml(cls, file):
        with open(file, "r") as fid:
            data = yaml.safe_load(fid)
        return cls(**data)

    def __post_init__(self):
        if self.seed is None:
            self.seed = time.time_ns()

        # if self.n_grid != self.n_phis:
        # raise Warning("should be the same")

        # convert all lists to jax.numpy arrays
        # for field in fields(self.__class__):
        #     val = getattr(self, field.name)
        #     if isinstance(val, list):
        #         val = jnp.array(val)
        #         setattr(self, field.name, val)

    def to_yaml(self, file):
        data = asdict(self)
        # for key, val in data.items():
        #     if isinstance(val, jnp.ndarray):
        #         data[key] = val.tolist()
        with open(file, "w") as fid:
            yaml.dump(data, fid)
