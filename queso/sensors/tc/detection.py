def local_r(c, mu, n, k):
    for i in range(n):
        c.r(
            i,
            theta=mu[i, 0],
            alpha=mu[i, 1],
            phi=mu[i, 2],
        )
    c.barrier_instruction()
    return c


def computational_bases(c, mu, n, k):
    return c


def hadamard_bases(c, mu, n, k):
    for i in range(n):
        c.h(i)
    return c


def local_rx_ry_ry(c, mu, n, k):
    for i in range(n):
        c.rx(i, theta=mu[i, 0])
        c.ry(
            i,
            theta=mu[i, 1],
        )
        c.ry(
            i,
            theta=mu[i, 2],
        )
    c.barrier_instruction()
    return c
