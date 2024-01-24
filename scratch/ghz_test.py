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
    # gammas = jnp.logspace(-5, -0.5, 10)
    gammas = [0.000001]
    n = 4
    for gamma in gammas:
        sensor = Sensor(
            n=n,
            k=4,
            **dict(
                preparation='ghz_dephasing',
                # interaction='local_rx',
                interaction='local_rz',
                # detection='computational_bases',
                detection='hadamard_bases',
                gamma_dephasing=gamma,
                backend='ket',
            )
        )

        #%%
        theta = sensor.theta
        # phi = jnp.array(jnp.pi/2/n)
        phi = jnp.array(0.1)
        mu = sensor.mu

        probs = sensor.probs(theta, phi, mu)
        print(probs)
        plt.figure()
        plt.bar([i for i in range(len(probs))], probs)
        plt.show()
        print(f"gamma={gamma} | qfi = {sensor.qfi(theta, phi):0.5f} | cfi = {sensor.cfi(theta, phi, mu):0.5f}")
        # dpr = jax.jacrev(sensor.probs, argnums=1, holomorphic=False)(theta, phi, mu)
        # print(dpr)

        shots = sensor.sample(theta, phi, mu, n_shots=10)
        print(shots)
        print(shots.sum(axis=-1) % 2)

        phis = jnp.linspace(0.0, jnp.pi/2/n, 100)

        #%%
        def estimation(shots, phis):
            parity = shots.sum(axis=-1) % 2  # compute the parity of the bitstring
            posteriors = jnp.log(parity[:, None] * jnp.sin(n * phis[None, :])**2 + jnp.logical_not(parity[:, None]) * jnp.cos(n * phis[None, :])**2)
            posterior = jnp.sum(posteriors, axis=0)
            return posterior


        # estimator =
        posterior = estimation(shots, phis)
        plt.figure()
        plt.plot(phis, jnp.exp(posterior))
        plt.xlabel(r"$\phi$")
        plt.show()

        """
        todo: 
            - compute the bias and variance for bootstrapped shots
            - 
        """
        #%%