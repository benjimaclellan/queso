#

<p align="center">
  <img src=imgs/logo.png alt="Queso" width="500" />
</p>


## Variational methods for programmable quantum sensors
[![GitHub Workflow Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/ki3-qbt/graph-compiler/actions)
[![docs.rs](https://img.shields.io/badge/docs-passing-brightgreen)](https://github.com/ki3-qbt/graph-compiler/tree/gh-pages)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)

!!! note
    Welcome to Queso - a design toolbox for variational quantum sensing protocols. 
    This documentation is still under development, so expect major changes.
    © Benjamin MacLellan, 2023



## What does it do?
Queso is a toolbox for designing, simulating, and benchmarking variational quantum sensing protocols.
<p align="center">
  <img src=imgs/fig1.png alt="GraphiQ" width="400" />
</p>


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
!!! note
    Official `pip` wheel is in the works.
