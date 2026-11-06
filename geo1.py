import numpy as np
import matplotlib.pyplot as plt

m_const = 1
c_const = 1
G_const = 1
E_const = 100

x = np.array([100, 100, 0])
t = 0
p = np.array([0, -10, 0])
pt = -E_const

r = np.linalg.norm(x)


def A(x, m) -> float:
    ratio = m / np.linalg.norm(x)
    return np.square((1 - ratio) / (1 + ratio))


def B(x, m) -> float:
    rho = np.linalg.norm(x)
    return np.power(1 + m / rho, 4)


def F(x, p, m, c) -> float:
    a = A(x, m)
    b = B(x, m)
    rho = np.linalg.norm(x)
    return (pt * 4 * m * (1 - m / rho)) / (
        2 * np.square(c * a * rho) * np.power(1 + m / rho, 3)
    ) + ((np.square(pt) - np.dot(p, p)) * 4 * m * np.power(1 + m / rho, 3)) / (
        2 * np.square(b * rho)
    )


nsteps = 100000
dt = 0.001

x_arr = []
y_arr = []
z_arr = []

for i in range(nsteps):
    x_arr.append(x[0])
    y_arr.append(x[1])
    z_arr.append(x[2])
    rho = np.linalg.norm(x)
    a = A(x, m_const)
    b = B(x, m_const)
    f = F(x, p, m_const, c_const)
    t += -(dt * pt) / (a * np.square(c_const))
    x = x + dt * p / b
    p = p - dt * f * x / rho

plt.scatter(x_arr, y_arr)
plt.scatter([0], [0], color="red")
plt.show()
