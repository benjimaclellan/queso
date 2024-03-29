import uuid

from queso.sensors.pennylane.sensor import brick_wall_probe
from queso.sensors.pennylane.sensor import Interaction, Measurement, Sensor
from queso.sensors.pennylane.sensor import Phase, U3


def test_calculate_state():
    n_layers = 2
    n = 4

    probe = brick_wall_probe(n=n, d=2, n_layers=n_layers)

    interaction = Interaction(n=n)
    interaction.add([Phase("phi") for _ in range(n)])

    measurement = Measurement(n=n)
    measurement.add([U3(str(uuid.uuid4())) for _ in range(n)])

    sensor = Sensor(probe, interaction, measurement)
    params = sensor.initialize()
