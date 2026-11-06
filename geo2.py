import numpy as np
import scipy
import matplotlib.pyplot as plt

M = 1e9
rs = 2 * scipy.constants.G * M / np.square(scipy.constants.c)

f = lambda r, rs: 1 - rs / r
f_1 = lambda r, rs: rs / np.square(r)
G_ttr = lambda r, rs: f_1(r, rs) / (2 * f(r, rs))
G_rtt = lambda r, rs: 0.5 * np.square(scipy.constants.c) * f(r, rs) * f_1(r, rs)
G_rrr = lambda r, rs: -f_1(r, rs) / (2 * f(r, rs))
G_raa = lambda r, rs: -r * f(r, rs)
G_rbb = lambda r, rs, a: -r * f(r, rs) * np.square(np.sin(a))
G_ara = lambda r: 1 / r
G_abb = lambda a: -np.sin(a) * np.cos(a)
G_brb = lambda r: 1 / r
G_bab = lambda a: 1 / np.tan(a)

t = 0
r = 10 * rs
a = 0
b = 0

r1 = 0
a1 = np.pi / 4
b1 = 0
t1 = (
    np.sqrt(
        (np.square(r1) / f(r, rs) + np.square(r * a1) + np.square(r * np.sin(a) * b1))
        / f(r, rs)
    )
    / scipy.constants.c
)

t_arr = []
r_arr = []
a_arr = []
b_arr = []

nsteps = 1000000
dt = 1e-5

for i in range(nsteps):
    t_arr.append(t)
    r_arr.append(r)
    a_arr.append(a)
    b_arr.append(b)
    t2 = -2 * G_ttr(r, rs) * t1 * r1
    r2 = (
        -G_rtt(r, rs) * np.square(t1)
        - G_rrr(r, rs) * np.square(r1)
        - G_rbb(r, rs, a) * np.square(b)
    )
    b2 = -2 * G_brb(r) * r1 * b1
    t1 += t2 * dt
    r1 += r2 * dt
    b1 += b2 * dt
    t += (0.5 * t2 * dt + t1) * dt
    r += (0.5 * r2 * dt + r1) * dt
    a += a1 * dt
    b += (0.5 * b2 * dt + b1) * dt

plt.scatter(r_arr * np.cos(a_arr), r_arr * np.sin(a_arr))
plt.scatter(
    rs * np.cos(np.linspace(0, 2 * np.pi, 100)),
    rs * np.sin(np.linspace(0, 2 * np.pi, 100)),
    color="red",
)
plt.axis("equal")
plt.show()
