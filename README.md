<div align="center">

# ![Queso](docs/imgs/logo.png)

<h2 align="center">
    Variational quantum sensing protocols
</h2>

[![Documentation Status](https://readthedocs.org/projects/queso/badge/?version=latest)](https://queso.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2403.02394-red)](https://arxiv.org/abs/2403.02394)

</div>





[//]: # (<p align="center" style="font-size:20px">)

[//]: # (    The design and optimization of quantum sensing protocols using variational methods.)

[//]: # (</p>)


## What does it do:
Explore, optimize, and benchmark circuits and estimators for quantum sensing protocols.
The quantum probe is represented as parameterized quantum circuits, and the estimators as classical neural networks.


## Basic usage:
```py
import jax
import jax.numpy as jnp
from queso.sensors import Sensor
from queso.estimators import BayesianDNNEstimator

sensor = Sensor(n=4, k=4)

theta, phi, mu = sensor.theta, sensor.phi, sensor.mu
sensor.qfi(theta, phi)
sensor.cfi(theta, phi, mu)
sensor.state(theta, phi, mu)

data = sensor.sample(theta, phi, mu, n_shots=10)

estimator = BayesianDNNEstimator()
posterior = estimator(data)
```



## Install
```bash
pip install git+https://github.com/benjimaclellan/queso.git
```
Quantum circuit simulations are done with [`tensorcircuit`](https://github.com/tencent-quantum-lab/tensorcircuit) 
with [JAX](https://github.com/google/jax) as the differentiable programming backend.
Neural networks are also built on top of JAX using the [`flax`](https://github.com/google/flax) library.

## Citing
```
@article{maclellan2024endtoend,
      title={End-to-end variational quantum sensing}, 
      author={Benjamin MacLellan and Piotr Roztocki and Stefanie Czischek and Roger G. Melko},
      year={2024},
      eprint={2403.02394},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```


## Acknowledgements
This project is supported by the Perimeter Institute Quantum Intelligence Lab and the 
Institute for Quantum Computing.
