import numpy as np
from rv_to_elements import calc_elements
from utils import newton_raphson, plot_orbit

def via_anomalia_verdadeira(r_vec, v_vec, dt, orbit_elem, grav_parameter=398600):
    """    
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

    # r0
    r0 = np.linalg.norm(r_vec)

    # semi lactus p
    p = t0_orbit_elem['a'] * (1 - t0_orbit_elem['e']**2)

    # Anomalia Excentrica E0
    E0 = 2*np.arctan(np.sqrt((1 - t0_orbit_elem['e'])/(1 + t0_orbit_elem['e'])) * np.tan(np.radians(t0_orbit_elem['f'])/2))

    # Anomalia Media M
    M0 = E0 - t0_orbit_elem['e'] * np.sin(E0)
    n = np.sqrt(grav_parameter/t0_orbit_elem['a']**3)

    for t in dt:
        # Convertendo horas para segundos
        deltaT = t * 3600
        M = M0 + n*deltaT

        # Anomalia Excentria no tempo T, utilizando New-Raphson
        E = newton_raphson(E0, M, t0_orbit_elem['e'])

        # anomalia verdadeira f
        f = 2 * np.arctan(np.sqrt((1 + t0_orbit_elem['e'])/(1 - t0_orbit_elem['e'])) * np.tan(E/2))

        # delta f
        delta_f = f - np.radians(t0_orbit_elem['f'])

        # Determinacao de r
        r = p / (1 + t0_orbit_elem['e'] * np.cos(f))

        # Determinacao dos coeficientes de Lagrange f, g, f_dot e g_dot	
        F = 1 - (r/p)*(1-np.cos(delta_f))
        G = r*r0*np.sin(delta_f)/np.sqrt(grav_parameter*p)
        F_dot = np.sqrt(grav_parameter/p) * np.tan(delta_f/2)*( (1 - np.cos(delta_f))/p - (1/r) - (1/r0))
        G_dot = 1 - r0*(1-np.cos(delta_f))/p

        # Determinacao dos vetores r e v
        r_vec_t = F * r_vec.T + G * v_vec.T
        v_vec_t = F_dot * r_vec.T + G_dot * v_vec.T

        answer[f"{t} horas"] = {
                "r_vec": r_vec_t,
                "v_vec": v_vec_t
            }
        
    return answer

# Dados do exercicio
r_vec = np.array([1.0, 1.0, 0]) * 1e4  # km (vetor posição)
v_vec = np.array([2.0, 4.0, 4.0])  # km/s (vetor velocidade)

dt = [0.5, 1, 2] # Horas
# Comparando com Ex2
# dt = [1/60, 2/60, 5/60, 10/60] # Horas

# Dados Exemplo
# r_vec = np.array([1.0, 0, np.sqrt(3.0)]) * 1e4  # km (vetor posição)
# v_vec = np.array([2.0, 4.0, 4.0])  # km/s (vetor velocidade)
# dt = [10]

# Adaptado da lista 3
orbit_elem = calc_elements(r_vec, v_vec)
# print(f"Orbital Elements: {orbit_elem}")

# Constants
GRAV_PARAM = 398600 # km³/s² (Terra)

answer = via_anomalia_verdadeira(r_vec, v_vec, dt, orbit_elem, grav_parameter=GRAV_PARAM)
print(answer)
for t in dt:    
    fig = plot_orbit(**calc_elements(answer.get(f'{t} horas')['r_vec'], answer.get(f'{t} horas')['v_vec']))
    fig.savefig(f"images/{t}_horas.png")


