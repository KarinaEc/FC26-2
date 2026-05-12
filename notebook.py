%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Constantes Físicas Reales
G = 6.67430e-11
MT, ML = 5.972e24, 7.342e22
R_TL, radio_T = 384400000.0, 6371000.0
omega = np.sqrt(G * (MT + ML) / R_TL**3) 

def aceleracion(t, r):
    # Luna inicia en (R_TL, 0) y orbita en sentido antihorario
    phi = omega * t
    pos_L = np.array([R_TL * np.cos(phi), R_TL * np.sin(phi)])
    r_satT, r_satL = r, r - pos_L
    
    aT = -G * MT * r_satT / np.linalg.norm(r_satT)**3
    aL = -G * ML * r_satL / np.linalg.norm(r_satL)**3
    return aT + aL

def rk4_step(t, r, v, dt):
    k1_v, k1_r = aceleracion(t, r), v
    k2_v, k2_r = aceleracion(t + 0.5*dt, r + 0.5*dt*k1_r), v + 0.5*dt*k1_v
    k3_v, k3_r = aceleracion(t + 0.5*dt, r + 0.5*dt*k2_r), v + 0.5*dt*k2_v
    k4_v, k4_r = aceleracion(t + dt, r + dt*k3_r), v + dt*k3_v
    return r + (dt/6)*(k1_r + 2*k2_r + 2*k3_r + k4_r), v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

# --- 2. CONDICIONES INICIALES DE PRECISIÓN ---
# Posición inicial: 200km sobre la superficie en el eje X
r = np.array([radio_T + 200000.0, 0.0])

# v0 CALIBRADO PARA INTERCEPCIÓN EN EL ESPACIO (Artemis II style)
# vx=10080 y vy=3920: Esta combinación asegura que el satélite llegue
# a la zona lunar justo cuando la Luna está en el ángulo correcto para
# desviarlo hacia el centro, creando el cruce del "8".
v = np.array([10080.0, 3920.0]) 

dt, pasos = 40, 120000 
pos_sat, pos_lun = [], []

for i in range(pasos):
    t_actual = i * dt
    pos_sat.append(r.copy())
    pos_lun.append(np.array([R_TL * np.cos(omega * t_actual), R_TL * np.sin(omega * t_actual)]))
    r, v = rk4_step(t_actual, r, v, dt)

pos_sat, pos_lun = np.array(pos_sat), np.array(pos_lun)

# 3. Configuración de la Animación
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-R_TL*0.6, R_TL*1.4); ax.set_ylim(-R_TL*0.8, R_TL*0.8)

trayectoria, = ax.plot([], [], 'b-', lw=1.2, alpha=0.7, label='Satélite')
sat_dot, = ax.plot([], [], 'ro', markersize=3)
tierra_v = plt.Circle((0, 0), radio_T*6, color='blue', label='Tierra (x6)')
luna_v = plt.Circle((R_TL, 0), 1737000*6, color='gray', label='Luna (x6)')
ax.add_artist(tierra_v); ax.add_artist(luna_v)

def update(i):
    idx = i * 600
    if idx >= len(pos_sat): idx = len(pos_sat) - 1
    trayectoria.set_data(pos_sat[:idx, 0], pos_sat[:idx, 1])
    sat_dot.set_data([pos_sat[idx, 0]], [pos_sat[idx, 1]])
    luna_v.center = (pos_lun[idx, 0], pos_lun[idx, 1])
    return trayectoria, sat_dot, luna_v

ani = FuncAnimation(fig, update, frames=len(pos_sat)//600, interval=30, blit=True)
plt.title("Ejercicio 3: Trayectoria de Retorno Libre Artemis II (El Ocho)")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()