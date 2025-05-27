import numpy as np
from rv_to_elements import calc_elements
from utils import newton_raphson

def via_anomalia_excentrica(r_vec, v_vec, dt, orbit_elem, grav_parameter=398600):
    """
    Calcula a anomalia excentrica e os vetores de posição e velocidade em um instante de tempo t
    a partir da anomalia excentrica inicial e do vetor de posição e velocidade iniciais.
    
    Parâmetros:
    r_vec (np.array): vetor de posição inicial (km)
    v_vec (np.array): vetor de velocidade inicial (km/s)
    dt (list): intervalo de tempo (horas)
    orbit_elem (dict): elementos orbitais iniciais
    grav_parameter (float): parâmetro gravitacional (km³/s²)
    
    Retorna:
    r_vec_t (np.array): vetor de posição no instante t (km)
    v_vec_t (np.array): vetor de velocidade no instante t (km/s)
    """
    answer = {}
    # Determinar os elementos orbitais iniciais
    t0_orbit_elem = orbit_elem
    a = t0_orbit_elem['a']
    e = t0_orbit_elem['e']

    # r0
    r0 = np.linalg.norm(r_vec)

    # Anomalia Excentrica E0
    E0 = 2*np.arctan(np.sqrt((1 - e)/(1 + e)) * np.tan(np.radians(t0_orbit_elem['f'])/2))

    # Anomalia Media M
    M0 = E0 - e * np.sin(E0)
    n = np.sqrt(grav_parameter/a**3)

    for t in dt:
        # Convertendo horas para segundos
        t = t * 3600
        M = M0 + n*t

        # Anomalia Excentria no tempo T, utilizando New-Raphson
        E = newton_raphson(E0, M, e)

        deltaE = E - E0

        # Determinacao de r
        r = a*(1-e*np.cos(E))

        # Determinacao dos coeficientes de Lagrange f, g, f_dot e g_dot	
        F = 1 - (a/r0)*(1-np.cos(deltaE))
        G = t - np.sqrt(a**3/grav_parameter)*(deltaE-np.sin(deltaE))
        F_dot = -((np.sqrt(grav_parameter*a)/(r*r0)) * np.sin(deltaE))
        G_dot = 1 - (a/r)*(1-np.cos(deltaE))

        # Determinacao dos vetores r e v
        r_vec_t = F * r_vec.T + G * v_vec.T
        v_vec_t = F_dot * r_vec.T + G_dot * v_vec.T

        answer[f"{t/3600:.2f} horas"] = {
                "r_vec": r_vec_t,
                "v_vec": v_vec_t
            }
        
    return answer

# Dados do exercicio
r_vec = np.array([1.0, 1.0, 0]) * 1e4  # km (vetor posição)
v_vec = np.array([2.0, 4.0, 4.0])  # km/s (vetor velocidade)

dt = [0.5, 1, 2]

# Dados Exemplo
# r_vec = np.array([1.0, 0, np.sqrt(3.0)]) * 1e4  # km (vetor posição)
# v_vec = np.array([2.0, 4.0, 4.0])  # km/s (vetor velocidade)

# dt = [10, 4, 3] # Horas

# Adaptado da lista 3
orbit_elem = calc_elements(r_vec, v_vec)

# Constants
GRAV_PARAM = 398600 # km³/s² (Terra)

print(via_anomalia_excentrica(r_vec, v_vec, dt, orbit_elem, grav_parameter=GRAV_PARAM))


