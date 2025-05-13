import numpy as np

rt = 6378
grav_param = 398600
P = 120 * 60
rp = 1000 + rt
r = 1050 + rt
print(f"Raio Perigeu: {rp} km")
print(f"Raio a 1050 km: {r} km")

a = ((grav_param * P**2) / (4 * np.pi**2))**(1/3) # km 
print(f"Semi-eixo maior: {a} km")

e = 1 - rp/a 
print(f"Excentricidade: {e}")

print("Anomalias Excentricas nos dois pontos onde h = 1050km")
E = np.arccos(1/e - r/(a*e))
E2 = 2 * np.pi - E # rad
print(f"Anomalia excentrica: {np.degrees(E)} graus, ou {E} rad")
print(f"Anomalia excentrica_2: {np.degrees(E2)} graus, ou {E2} rad")

# print("Anomalias Verdadeiras nos dois pontos onde h = 1050km")
# f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2)) # rad
# f2 = 2*np.pi - f # rad
# print(f"Anomalia verdadeira: {np.degrees(f)} graus")
# print(f"Anomalia verdadeira_2: {np.degrees(f2)} graus")

print("Anomalias medias nos dois pontos onde h = 1050km")
M = E - e * np.sin(E) # rad
M2 = E2 - e * np.sin(E2) # rad
print(f"Anomalia media: {np.degrees(M)} graus, ou {M} rad")
print(f"Anomalia media_2: {np.degrees(M2)} graus, ou {M2} rad")

t = P * M / (2 * np.pi) # s
t2 = P * M2 / (2 * np.pi) # s
print(f"Tempo de voo (ate M): {t} s")
print(f"Tempo de voo_2 (ate M2): {t2} s")
dt = t2 - t
print(dt)

print(f"Tempo de voo hh:mm:ss: {int(dt/3600)}:{int((dt%3600)/60)}:{int(dt%60)}")