from queso.utils import *
import qutip as qt

# n = 6
# inds = list(kbits(n))
#
# lst = list(range(n))
# for s in inds:
#     a = [lst[i] for i in range(n) if s[i] == 1]
#     b = [lst[i] for i in range(n) if s[i] == 0]
#     print(a, b)

# %%
n = 6
# rho = qt.ket2dm(qt.rand_ket(dims=[[2] * n, [1] * n]))
# rho = qt.ket2dm(qt.tensor(n * [qt.basis(2, 0)]) + qt.tensor(n * [qt.basis(2, 1)])).unit()   # GHZ state
# rho = qt.ket2dm(qt.tensor(n * [qt.basis(2, 0) + qt.basis(2, 1)])).unit()   # n-plus state
# rho = qt.ket2dm(qt.tensor(n * [qt.basis(2, 0)])).unit()   # n-plus state

ghz = qt.ket2dm(
    qt.tensor((n // 2) * [qt.basis(2, 0)]) + qt.tensor((n // 2) * [qt.basis(2, 1)])
).unit()  # GHZ state
rho = qt.tensor(2 * [ghz]).unit()

partitions = partition(list(range(n)))

for inds_a, inds_b in partitions:
    for inds in (inds_a, inds_b):
        prho = qt.ptrace(rho, inds)
        entropy = qt.entropy_vn(prho)
        print(inds, entropy)
