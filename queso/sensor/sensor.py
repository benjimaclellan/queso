from queso.sensor.blocks import Probe, Interaction, Measurement
from queso.sensor.functions import nketz0


class Sensor:
    def __init__(
        self,
        probe: Probe,
        interaction: Interaction = None,
        measurement: Measurement = None,
    ):
        self.probe = probe
        self.interaction = interaction
        self.measurement = measurement

        self.state_i = nketz0(n=self.probe.n, d=self.probe.d)

    def initialize(self):
        params = {}
        for block in [self.probe, self.interaction, self.measurement]:
            params.update(block.initialize())
        return params

    def __call__(self, params):
        return (
            self.measurement(params)
            @ self.interaction(params)
            @ self.probe(params)
            @ self.state_i
        )
