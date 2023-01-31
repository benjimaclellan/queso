class Channel:
    def __init__(self, key):
        self.key = key


class StaticChannel(Channel):
    def __init__(self, key):
        super().__init__(key)
        self.m, self.bounds, self.initial = None, None, None


class DephasingChannel(StaticChannel):
    def __init__(
        self,
        key,
        lam=0.0,
    ):
        super().__init__(key)
        raise NotImplementedError()

    def __call__(self):
        raise NotImplementedError()
