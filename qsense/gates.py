from qsense.functions import *


class Gate:
    def __init__(self, key):
        self.key = key


class StaticGate(Gate):
    def __init__(self, key):
        super().__init__(key)
        self.m, self.bounds, self.initial = None, None, None


class ParameterizedGate(Gate):
    def __init__(self, key):
        super().__init__(key)
        return


class Identity(StaticGate):
    def __call__(self):
        return eye()


class X(StaticGate):
    def __call__(self):
        return x()


class Y(StaticGate):
    def __call__(self):
        return y()


class Z(StaticGate):
    def __call__(self):
        return z()


class H(StaticGate):
    def __call__(self):
        return h()


class CNOT(StaticGate):
    def __init__(self, key, n=2, control=0, target=1):
        super().__init__(key)
        self.n = n
        self.control = control
        self.target = target

    def __call__(self):
        d0 = {self.control: ketz0() @ ketz0().T, self.target: eye()}
        d1 = {self.control: ketz1() @ ketz1().T, self.target: x()}
        return tensor([d0.get(reg, eye()) for reg in range(self.n)]) + tensor([d1.get(reg, eye()) for reg in range(self.n)])


class Phase(ParameterizedGate):
    def __init__(self, key):
        super().__init__(key)
        self.m, self.bounds, self.initial = 1, (-np.pi, np.pi), [0.0]

    def __call__(self, phi):
        return phase(phi)


class RX(ParameterizedGate):
    def __init__(self, key):
        super().__init__(key)
        self.m, self.bounds, self.initial = 1, (-np.pi, np.pi), [0.0]

    def __call__(self, theta):
        return u3(theta, 0.0, 0.0)


class RZ(ParameterizedGate):
    def __init__(self, key):
        super().__init__(key)
        self.m, self.bounds, self.initial = 1, (-np.pi, np.pi), [0.0]

    def __call__(self, phi):
        return rz(phi)


class U2(ParameterizedGate):
    def __init__(self, key):
        super().__init__(key)
        self.m, self.bounds, self.initial = 2, (-np.pi, np.pi), [0.0, 0.0]

    def __call__(self, theta, phi):
        return rx(theta) @ rz(phi)


class U3(ParameterizedGate):
    def __init__(self, key):
        super().__init__(key)
        self.m, self.bounds, self.initial = 3, (-np.pi, np.pi), [0.0, 0.0, 0.0]

    def __call__(self, theta, phi, lam):
        return u3(theta, phi, lam)
