import numpy as np
import matplotlib.pyplot as plt

rs = 1

r0 = 10 * rs
phi0 = 0

r10 = -1
phi10 = 1e3

u0 = 1 / r0
u10 = -r10 / (r0 * r0)

dphi = 1e-5
nsteps = int(0.5 * np.pi / dphi)

x_arr = []
y_arr = []


def getx(u, phi) -> float:
    return np.cos(phi) / u


def gety(u, phi) -> float:
    return np.sin(phi) / u


u = u0
phi = phi0
u1 = u10
phi1 = phi10

for i in range(nsteps):
    x_arr.append(getx(u, phi))
    y_arr.append(gety(u, phi))
    u2 = (3 * rs / 2 * u - 1) * u
    u1 += u2 * dphi
    u += (0.5 * u2 * dphi + u1) * dphi
    phi += dphi


plt.scatter(x_arr, y_arr)
plt.scatter([0], [0], color="red")
plt.show()
