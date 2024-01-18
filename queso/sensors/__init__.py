from jax.config import config

from queso.sensors.tc.sensor import Sensor


config.update("jax_enable_x64", True)
