import numpy as np

MU = 398600

# INPUT
r0 = 7000
THETA = 45
THETA = np.radians(THETA)

# case 1
# 1 impulso
delta_v = 2*np.sqrt(MU/r0)*np.sin(THETA/2)
print(f"deltaV 1 impulso {delta_v:.4f} km/s")

# Case 2
# 3 impulsos
sigma = np.sin(THETA/2)/(1 - 2*np.sin(THETA/2))
e1 = (sigma - 1)/(sigma + 1)
print(f"Excentricidade manobra intermediaria: {e1:.4f}")
dv0 = np.sqrt(MU/r0)*(np.sqrt(1+e1) - 1)
print(f"deltaV manobra inicial = final: {dv0:.4f} km/s")
dvf = dv0

r = r0*sigma
print(f"raio manobra intermediaria: {r:.4f} km")
dvi = 2*np.sqrt(MU/r)*np.sqrt(1-e1)*np.sin(THETA/2)
print(f"deltaV manobra inermediaria: {dvi:.4f} km/s")
Cf = dv0 + dvf + dvi
print(f"deltaV Total 3 impulsos {Cf:.4f} km/s")


