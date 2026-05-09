import numpy as np

def f(t, y):
    y1, y2 = y
    dy1 = y2
    dy2 = np.exp(-t)*(np.sin(3*t)+np.cos(2*t)) - 0.5*y2 - np.sin(y1)
    return np.array([dy1, dy2])

def rk2(f, y0, t0, tf, h):
    t = np.arange(t0, tf+h, h)
    y = np.zeros((len(t), 2))
    y[0] = y0

    for i in range(len(t)-1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h, y[i] + k1)
        y[i+1] = y[i] + 0.5 * (k1 + k2)

    return t, y

# condiciones iniciales
y0 = [0, 1]

t, y = rk2(f, y0, 0, 10, 0.1)