import numpy as np
import time

def F_newton_raphson(e, N, guess=None, tol=1e-15, max_iter=50):
    t0 = time.time()
    F = N if guess == None else guess # Chute inicial
    for _ in range(max_iter):
        f = e*np.sinh(F) - F - N
        f_prime = e * np.cosh(F) - 1
        delta = -f/f_prime
        F+=delta
        if np.abs(delta) < tol:
            break
    print(f"NEWTON-RAPHSON TIMING: {time.time()-t0}")
    return F

hc = 350 #km
rt = 6378 #km
grav_param = 398600 #km3/s2
v_inf = 3 #km/s
t = 6 * 60 * 60 #s

# Altitude no perigeu
rp = rt + hc
print(f"Altitude no perigeu {rp} km")

# semi-eixo maior
a = -grav_param/(v_inf**2)
print(f"Semi-eixo maior {a:.4f} km")

# Anomalia Media Hiperbolica
print(f"Tempo em segundos {t} s")

N = np.sqrt(grav_param/((-a)**3))*t
print(f"Anomalia media Hiperbolica {N} rads")

# excentricidade
e = 1 + (rp*v_inf**2)/grav_param
print(f"Excentricidade {e}")

# Anomalia Excentrica Hiperbolica
F = F_newton_raphson(e, N, guess=N)
print(f" Anomalia excentrica Hiperbolica {F}")

# Distancia ao centro da terra
r=a*(1-e*np.cosh(F))
print(np.cosh(F))
print(f"Distancia ao centro da terra {r}")

