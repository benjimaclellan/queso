from queso.sensor.functions import *


class Gate:
    def __init__(self, key=None, d=2):
        self.key = key
        self.d = d


class StaticGate(Gate):
    def __init__(self, key=None, d=2):
        super().__init__(key, d)
        self.m, self.bounds, self.initial = None, None, None


class ParameterizedGate(Gate):
    def __init__(self, key, d):
        super().__init__(key, d)
        return


class Identity(StaticGate):
    def __call__(self):
        return eye(self.d)


class X(StaticGate):
    def __call__(self):
        return x(self.d)


class Y(StaticGate):
    def __call__(self):
        return y(self.d)


class Z(StaticGate):
    def __call__(self):
        return z(self.d)


class H(StaticGate):
    def __call__(self):
        return h(self.d)


class CNOT(StaticGate):
    def __init__(self, key=None, d=2, n=2, control=0, target=1):
        super().__init__(key, d)
        self.n = n
        self.control = control
        self.target = target

    def __call__(self):
        return cnot(d=self.d, n=self.n, control=self.control, target=self.target)


class Phase(ParameterizedGate):
    def __init__(self, key=None, d=2):
        super().__init__(key, d)
        self.m, self.bounds, self.initial = 1, (-np.pi, np.pi), [0.0]

    def __call__(self, phi):
        return phase(phi, d=self.d)


class RDX(ParameterizedGate):
    def __init__(self, key=None, d=2):
        super().__init__(key, d)
        self.m = len(list(itertools.combinations(range(d), 2)))
        self.bounds = (-np.pi, np.pi)
        self.initial = self.m * [0.0]

    def __call__(self, *args):
        return rdx(*args, d=self.d)


class RX(ParameterizedGate):
    def __init__(self, key=None, d=2):
        super().__init__(key, d)
        self.m, self.bounds, self.initial = 1, (-np.pi, np.pi), [0.0]

    def __call__(self, theta):
        return u3(theta, 0.0, 0.0, d=self.d)


class RZ(ParameterizedGate):
    def __init__(self, key=None, d=2):
        super().__init__(key, d)
        self.m, self.bounds, self.initial = 1, (-np.pi, np.pi), [0.0]

    def __call__(self, phi):
        return rz(phi, d=self.d)


class U2(ParameterizedGate):
    def __init__(self, key=None, d=2):
        super().__init__(key, d)
        self.m, self.bounds, self.initial = 2, (-np.pi, np.pi), [0.0, 0.0]

    def __call__(self, theta, phi):
        return rx(theta, d=self.d) @ rz(phi, d=self.d)


class U3(ParameterizedGate):
    def __init__(self, key=None, d=2):
        super().__init__(key, d)
        self.m, self.bounds, self.initial = 3, (-np.pi, np.pi), [0.0, 0.0, 0.0]

    def __call__(self, theta, phi, lam):
        return u3(theta, phi, lam, d=self.d)
