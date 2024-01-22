def local_rx(c, phi, n):
    for i in range(n):
        c.rx(i, theta=phi)
    c.barrier_instruction()
    return c


def local_rz(c, phi, n):
    for i in range(n):
        c.rz(i, theta=phi)
    c.barrier_instruction()
    return c


def fourier_rx(c, phi, n):
    for i in range(0, n, 2):
        c.rx(i, theta=phi)
    for i in range(1, n, 2):
        c.rz(i, theta=-phi)
    c.barrier_instruction()
    return c


def local_depolarizing(c, phi, n):
    for i in range(n):
        c.depolarizing(i, phi)
    c.barrier_instruction()
    return c


def single_rx(c, phi, n):
    c.rx(0, theta=phi)
    c.barrier_instruction()
    return c
