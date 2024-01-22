import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from math import pi
from queso.io import IO

from queso.train.vqs import vqs
from queso.configs import Configuration
from queso.sensors.tc.sensor import Sensor


#%%
if __name__ == "__main__":
    #%%
    gammas = jnp.logspace(-5, -0.5, 10)
    # gammas = [0.01]

    for gamma in gammas:
        sensor = Sensor(
            n=4,
            k=4,
            **dict(
                preparation='ghz_dephasing',
                interaction='local_rz',
                detection='hadamard_bases',
                gamma_dephasing=gamma,
                backend='ket',
            )
        )

        theta = sensor.theta
        phi = jnp.array(0.001)
        mu = sensor.mu

        # print(sensor.probs(theta, phi, mu))
        print(f"gamma={gamma} | qfi = {sensor.qfi(theta, phi):0.5f} | cfi = {sensor.cfi(theta, phi, mu):0.5f}")
        # dpr = jax.jacrev(sensor.probs, argnums=1, holomorphic=False)(theta, phi, mu)
        # print(dpr)

        #%%