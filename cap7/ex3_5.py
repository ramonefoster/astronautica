import numpy as np

MU = 398600

# INPUT
r0 = 12000
THETA_DEG = 75
THETA = np.radians(THETA_DEG)

# case 1
# 1 impulso
if 0 <= THETA_DEG <= 38.94:
    delta_v = 2*np.sqrt(MU/r0)*np.sin(THETA/2)
    print(f"deltaV 1 impulso {delta_v:.4f} km/s")

elif 38.94 <= THETA_DEG <= 60:
    print("3 impulsos")
    # Case 2
    # 3 impulsos
    sigma = np.sin(THETA/2)/(1 - 2*np.sin(THETA/2))
    print(f"sigma: {sigma:.4f}")
    e1 = (sigma - 1)/(sigma + 1)
    print(f"Excentricidade manobra intermediaria: {e1:.4f}")
    dv0 = np.sqrt(MU/r0)*(np.sqrt(1+e1) - 1)
    print(f"deltaV manobra inicial = final: {dv0:.4f} km/s")
    dvf = dv0

    r = r0*sigma
    print(f"raio manobra intermediaria: {r:.4f} km")
    dvi = 2*np.sqrt(MU/r)*np.sqrt(1-e1)*np.sin(THETA/2)
    print(f"deltaV manobra inermediaria: {dvi:.4f} km/s")
    Cf = abs(dv0) + abs(dvf) + abs(dvi)
    print(f"deltaV Total 3 impulsos {Cf:.4f} km/s")

elif 60 <= THETA_DEG <=180:
    print("3 impulsos Bi-Parabolica")
    # case 3
    dv0 = np.sqrt(MU/r0)*(-1 + np.sqrt(2))
    print(f"Primeiro incremento: {dv0:.4f} km/s")

    #segundo incremento
    dvf = np.sqrt(MU/(r0))*(np.sqrt(2) -1)
    print(f"Segundo Incremento: {dvf:.4f} km/s")

    Cf = dv0 + dvf
    print(f"Delta V total: {Cf:.4f} km/s")

    print("Impulso intermediario a uma distancia infinita, de valor infinitezimal")

