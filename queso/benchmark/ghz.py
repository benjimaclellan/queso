import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from math import pi
from queso.io import IO
import tensorcircuit as tc

from queso.train.vqs import vqs
from queso.configs import Configuration
from queso.sensors.tc.sensor import Sensor


#%%
def ghz_comparison(
    io: IO,
    config: Configuration,
):
    #%%
    sensor = Sensor(
        n=config.n,
        k=None,
        **dict(
            preparation=config.preparation,
            interaction=config.interaction,
            detection=config.detection,
            gamma_dephasing=config.gamma_dephasing,
            backend=config.backend,
        )
    )
    n = config.n
    theta = sensor.theta
    # phi = jnp.array(0.0)
    phi = config.phi_center
    mu = sensor.mu
    gamma = config.gamma_dephasing

    fig = sensor.circuit(theta, phi, mu).draw(**dict(output='mpl'))
    # fig = sensor.circuit(theta, phi, mu).draw(**dict(output='iqp'))

    cfi = sensor.cfi(theta, phi, mu)
    # qfi = sensor.qfi(theta, phi)

    # compute fidelity with pure GHZ state
    c = sensor._circ(sensor.n)
    c = sensor.preparation(c, theta, sensor.n, sensor.k)
    state = c.state()
    if len(state.shape) == 1:  # ket
        print("ket")
        fid = 0.5 * jnp.abs(state[0] + state[-1]) ** 2
    elif len(state.shape) == 2:  # density matrix
        print("dm")
        fid = 0.5 * (state[0, 0] + state[-1, 0] + state[0, -1] + state[-1, -1])
    else:
        raise RuntimeError("State should always have 1 or 2 dims.")
    print(f"gamma={gamma} | cfi = {cfi:0.5f} | fidelity to GHZ = {fid}")

    metrics = dict(
        cfi=sensor.cfi(theta, phi, mu),
        fidelity=fid,
        entropy_vn=sensor.entanglement(theta, phi)
    )
    print(metrics)

    fig.show()
    io.save_figure(fig, filename="circuit.png")

    hf = io.save_h5('circ.h5')
    hf.create_dataset("gamma", data=gamma)
    for metric, arr in metrics.items():
        hf.create_dataset(metric, data=arr)
    hf.close()
