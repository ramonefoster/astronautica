import numpy as np
from rv_to_elements import calc_elements
from utils import plot_comparison
from universais import via_anomalia_universal
import math

# Set numpy print precision to 15 decimal places
def via_series_fg(r_vec, v_vec, dt, grav_parameter=398600):
    """    
    Parâmetros:
    r_vec (np.array): vetor de posição inicial (km)
    v_vec (np.array): vetor de velocidade inicial (km/s)
    dt (list): intervalo de tempo (horas)
    grav_parameter (float): parâmetro gravitacional (km³/s²)
    
    Retorna:
    r_vec_t (np.array): vetor de posição no instante t (km)
    v_vec_t (np.array): vetor de velocidade no instante t (km/s)
    """
    answer = {}
    # Determinar os elementos orbitais iniciais

    # r0, v0 e r0v0
    r0 = np.linalg.norm(r_vec)
    v0 = np.linalg.norm(v_vec)
    r0v0 = np.dot(r_vec, v_vec)

    u = grav_parameter/r0**3
    p = r0v0/r0**2
    q = -u + v0**2/r0**2

    F0 = 1
    F1 = 0
    F2 = -u
    F3 = 3*u*p
    F4 = -15*u*p**2 + 3*u*q + u**2
    F5 = 105*u*p**3 - (45*u*q + 15*u**2)*p
    F6 = -945*u*p**4 + (210*u**2 + 630*u*q)*p**2 - 24*u**2*q -u**3 -45*u*q**2
    F7 = 10395*u*p**5 - (9450*u*q + 3150*u**2)*p**3 + (1575*u*q**2 + 882*u**2*q + 63*u**3)*p
    
    G0 = 0
    G1 = 1
    G2 = 0
    G3 = -u
    G4 = 6*u*p
    G5 = -45*u*p**2 + 9*u*q + u**2
    G6 = 420*u*p**3 - (180*u*q + 30*u**2)*p
    G7 = -4725*u*p**4 + (3150*u*q + 630*u**2)*p**2 - 225*u*q**2 -54*q*u**2 -u**3

    Fn = [F0, F1, F2, F3, F4, F5, F6, F7]
    Gn = [G0, G1, G2, G3, G4, G5, G6, G7]
    
    for t in dt:
        # Convertendo horas para segundos
        T = t * 3600

        Fr = sum(Fn[n] * (T**n) / math.factorial(n) for n in range(7))
        Gr = sum(Gn[n] * (T**n) / math.factorial(n) for n in range(7))
        
        Fv = sum(Fn[n+1] * (T**n) / math.factorial(n) for n in range(7))
        Gv = sum(Gn[n+1] * (T**n) / math.factorial(n) for n in range(7))

        r_vec_t = Fr * r_vec + Gr * v_vec
        v_vec_t = Fv * r_vec + Gv * v_vec

        answer[f"{t} horas"] = {
                "r_vec": r_vec_t,
                "v_vec": v_vec_t
            }
        
    return answer

# Dados do exercicio
r_vec = np.array([1.0, 1.0, 0]) * 1e4  # km (vetor posição)
v_vec = np.array([2.0, 4.0, 4.0])  # km/s (vetor velocidade)

dt = [1/60, 2/60, 5/60, 10/60] # Horas

# Dados Exemplo
# r_vec = np.array([8195.31, 528.38, 3945.04])  # km (vetor posição)
# v_vec = np.array([0.6852, 6.6822, 1.5863])  # km/s (vetor velocidade)
# dt = [10]

# Adaptado da lista 3
orbit_elem = calc_elements(r_vec, v_vec)
# print(f"Orbital Elements: {orbit_elem}")

# Constants
GRAV_PARAM = 398600 # km³/s² (Terra)

answer = via_series_fg(r_vec, v_vec, dt, grav_parameter=GRAV_PARAM)
answer_exact = via_anomalia_universal(r_vec, v_vec, dt, orbit_elem, grav_parameter=GRAV_PARAM)

r_exact = [(answer_exact.get(f'{t} horas')['r_vec']) for t in dt]
r_serie = [(answer.get(f'{t} horas')['r_vec']) for t in dt]
v_exact = [(answer_exact.get(f'{t} horas')['v_vec']) for t in dt]
v_serie = [(answer.get(f'{t} horas')['v_vec']) for t in dt]

rv_exact = np.array([np.concatenate((r, v)) for r, v in zip(r_exact, v_exact)])
rv_serie = np.array([np.concatenate((r, v)) for r, v in zip(r_serie, v_serie)])

# for i, t in enumerate(dt):
#     err_percent = 100*(np.mean(np.abs((rv_serie[i] - rv_exact[i]) / rv_exact[i])))
#     print(f"t = {t} horas: Erro relativo {err_percent}")

dt = [1*60, 2*60, 5*60, 10*60]
plot_comparison(dt, r_exact, v_exact, r_serie, v_serie)



