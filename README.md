<div align="center">

# ![Queso](docs/imgs/logo.png)

<h2 align="center">
    Variational quantum sensing protocols
</h2>

<div align="center">

[![Documentation Status](https://readthedocs.org/projects/queso/badge/?version=latest)](https://queso.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
</div>

[//]: # ([![arXiv Paper]&#40;https://img.shields.io/badge/arXiv-2402.09285-red&#41;]&#40;https://arxiv.org/abs/2402.09285&#41;)




<p align="center" style="font-size:20px">
    The design and optimization of quantum sensing protocols using variational methods.
</p>


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

## Installation
`pip install queso`
Quantum circuit simulations are done with [`tensorcircuit`](https://github.com/tencent-quantum-lab/tensorcircuit) 
with [JAX](https://github.com/google/jax) as the differentiable programming backend.
Neural networks are also built on top of JAX using the [`flax`](https://github.com/google/flax) library.

## Citing
Preprint to be submitted soon. 


## Acknowledgements
This project is supported by the Perimeter Institute Quantum Intelligence Lab and the 
Institute for Quantum Computing.