import numpy as np
import matplotlib.pyplot as plt

def sistema_ej2(t, u):
    # u[0]=y, u[1]=y_punto, u[2]=y_dos_puntos
    y, dy, d2y = u
    
    du0 = dy
    du1 = d2y
    # Ecuación despejada para y_tres_puntos (du2):
    # y''' = exp(-t)*sin(3t) - (y'')^2 + 3y' - (cos(y))^2
    du2 = np.exp(-t) * np.sin(3 * t) - (d2y**2) + (3 * dy) - (np.cos(y)**2)
    
    return np.array([du0, du1, du2])

def rk4_step(f, t, u, h):
    k1 = f(t, u)
    k2 = f(t + 0.5*h, u + 0.5*h*k1)
    k3 = f(t + 0.5*h, u + 0.5*h*k2)
    k4 = f(t + h, u + h*k3)
    return u + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rk4_integrador(f, t_span, u0, h):
    t = np.arange(t_span[0], t_span[1] + h, h)
    u = np.zeros((len(t), len(u0)))
    u[0] = u0
    for i in range(len(t) - 1):
        u[i+1] = rk4_step(f, t[i], u[i], h)
    return t, u

# Parámetros para el Ejercicio 2
t_intervalo = (1.0, 1.65)
u0_ej2 = [1.0, 2.0, 1.0] # y(1)=1, y'(1)=2, y''(1)=1

# Ejecución para comparar precisión
t_h1, sol_h1 = rk4_integrador(sistema_ej2, t_intervalo, u0_ej2, h=0.001)
t_h2, sol_h2 = rk4_integrador(sistema_ej2, t_intervalo, u0_ej2, h=0.0001)

print(f"Resultado y(1.65) con h=0.001: {sol_h1[-1, 0]:.6f}")
print(f"Resultado y(1.65) con h=0.0001: {sol_h2[-1, 0]:.6f}")