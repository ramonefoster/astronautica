import numpy as np
from rv_to_elements import calc_elements
from utils import newton_raphson, newton_raphson_hyperbolic, plot_orbit

def via_anomalia_excentrica(r_vec, v_vec, dt, orbit_elem, grav_parameter=398600):
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
    e = t0_orbit_elem['e']

    # r0
    r0 = np.linalg.norm(r_vec)

    if e >= 0 and e < 1:
        answer = calc_ecliptic(r_vec, v_vec, r0, orbit_elem, grav_parameter)
    elif round(e, 3) == 1:
        answer = calc_parabolic(r_vec, v_vec, r0, orbit_elem, grav_parameter)
    elif e > 1:
        answer = calc_hyperbolic(r0, orbit_elem, grav_parameter)
    else:
        raise ValueError("Excentricidade inválida. Deve ser >= 0.")
    
    return answer

def calc_ecliptic(r_vec, v_vec, r0=None, orbit_elem=None, grav_parameter=398600):
    answer = {}
    # Anomalia Excentrica E0
    E0 = 2*np.arctan(np.sqrt((1 - orbit_elem['e'])/(1 + orbit_elem['e'])) * np.tan(np.radians(orbit_elem['f'])/2))

    # Anomalia Media M
    M0 = E0 - orbit_elem['e'] * np.sin(E0)
    n = np.sqrt(grav_parameter/orbit_elem['a']**3)

    for t in dt:
        # Convertendo horas para segundos
        deltaT = t * 3600
        M = M0 + n*deltaT

        # Anomalia Excentria no tempo T, utilizando New-Raphson
        E = newton_raphson(E0, M, orbit_elem['e'])

        deltaE = E - E0

        # Determinacao de r
        r = orbit_elem['a']*(1-orbit_elem['e']*np.cos(E))

        # Determinacao dos coeficientes de Lagrange f, g, f_dot e g_dot	
        F = 1 - (orbit_elem['a']/r0)*(1-np.cos(deltaE))
        G = deltaT - np.sqrt(orbit_elem['a']**3/grav_parameter)*(deltaE-np.sin(deltaE))
        F_dot = -((np.sqrt(grav_parameter*orbit_elem['a'])/(r*r0)) * np.sin(deltaE))
        G_dot = 1 - (orbit_elem['a']/r)*(1-np.cos(deltaE))

        # Determinacao dos vetores r e v
        r_vec_t = F * r_vec.T + G * v_vec.T
        v_vec_t = F_dot * r_vec.T + G_dot * v_vec.T

        answer[f"{t} horas"] = {
                "r_vec": r_vec_t,
                "v_vec": v_vec_t
            }
        
    return answer

def calc_parabolic(r_vec, v_vec, r0=None, orbit_elem=None, grav_parameter=398600):
    """
    Calcula os vetores de posição e velocidade em um instante de tempo t
    a partir da anomalia excentrica inicial e do vetor de posição e velocidade iniciais.
    
    Parâmetros:
    r0 (float): raio da órbita (km)
    orbit_elem (dict): elementos orbitais iniciais
    grav_parameter (float): parâmetro gravitacional (km³/s²)
    
    Retorna:
    answer (dict): dicionário com os vetores de posição e velocidade
    """
    answer = {}
    
    # Semi-latus rectum p
    p = np.linalg.norm(np.cross(r_vec, v_vec))**2 / grav_parameter
    Mp = 0
    D0 = np.sqrt(p) * ((3*Mp + np.sqrt(1 + 9*Mp**2))**(1/3) - (3*Mp + np.sqrt(1 + 9*Mp**2))**(-1/3))

    for t in dt:
        deltaT = t * 3600

        # Anomalia Parabolica Media Mp
        Mp = np.sqrt(grav_parameter/p**3) * deltaT  # rad

        # Equação de Barker - Anomalia Parabolica Inicial
        D = np.sqrt(p) * ((3*Mp + np.sqrt(1 + 9*Mp**2))**(1/3) - (3*Mp + np.sqrt(1 + 9*Mp**2))**(-1/3))

        r = 0.5 * (p + D**2)  # Distância do centro da Terra
    
        # Convertendo horas para segundos
        deltaD = D-D0
        
        F = 1 - (0.5/r0)*(deltaD)**2
        G = ((deltaD)/np.sqrt(grav_parameter))*(r0+0.5*D0*(deltaD))
        F_dot = -(np.sqrt(grav_parameter)/(r*r0)) * (deltaD)
        G_dot = 1 - (0.5/r)*(deltaD)**2


        # Determinacao dos vetores r e v
        r_vec_t = F * r_vec.T + G * v_vec.T
        v_vec_t = F_dot * r_vec.T + G_dot * v_vec.T

        answer[f"{t} horas"] = {
                "r_vec": r_vec_t,
                "v_vec": v_vec_t
            }
        
    return answer

def calc_hyperbolic(r0=None, orbit_elem=None, grav_parameter=398600):
    """
    Calcula os vetores de posição e velocidade em um instante de tempo t
    a partir da anomalia excentrica inicial e do vetor de posição e velocidade iniciais.
    
    Parâmetros:
    r0 (float): raio da órbita (km)
    orbit_elem (dict): elementos orbitais iniciais
    grav_parameter (float): parâmetro gravitacional (km³/s²)
    
    Retorna:
    answer (dict): dicionário com os vetores de posição e velocidade
    """
    answer = {}
    
    # Anomalia Hiperbolica Media Mh
    F0 = 2 * np.arctanh(np.sqrt((orbit_elem['e'] - 1)/(orbit_elem['e'] + 1)) * np.tan(np.radians(orbit_elem['f'])/2))
    N0 = orbit_elem['e'] * np.sinh(F0) - F0
    n = np.sqrt(grav_parameter/-orbit_elem['a']**3)

    for t in dt:
        deltaT = t * 3600
        N = N0 + n*deltaT

        # Anomalia Excentria no tempo T, utilizando New-Raphson
        F_anom = newton_raphson_hyperbolic(F0, N, orbit_elem['e'])
        
        deltaF = F_anom - F0

        # Determinacao de r
        r = orbit_elem['a']*(1-orbit_elem['e']*np.cosh(F_anom))
        
        F = 1 - (orbit_elem['a']/r0)*(1-np.cosh(deltaF))
        G = deltaT + np.sqrt(-orbit_elem['a']**3/grav_parameter) * (deltaF - np.sinh(deltaF))
        F_dot = -((np.sqrt(-grav_parameter*orbit_elem['a'])/(r*r0)) * np.sinh(deltaF))
        G_dot = 1 - (orbit_elem['a']/r)*(1-np.cosh(deltaF))

        # Determinacao dos vetores r e v# Determinacao dos vetores r e v
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

dt = [0.5, 1, 2]

# Dados Exemplo
# r_vec = np.array([1.0, 0, np.sqrt(3.0)]) * 1e4  # km (vetor posição)
# v_vec = np.array([2.0, 4.0, 4.0])  # km/s (vetor velocidade)

# dt = [10] # Horas

# # TESTE PARAB.
# r_vec=np.array([10000.000000,0.000000,0.000000])
# v_vec = np.array([0.000000,6.313481,6.313481])

# # TESTE HIPERB.
# r_vec = np.array([15000.000000,0.000000,15000.000000])
# v_vec = np.array([4.000000,4.000000,4.000000])

orbit_elem = calc_elements(r_vec, v_vec)

# Constants
GRAV_PARAM = 398600 # km³/s² (Terra)

answer = via_anomalia_excentrica(r_vec, v_vec, dt, orbit_elem, grav_parameter=GRAV_PARAM)
print(answer)
for t in dt:    
    fig = plot_orbit(**calc_elements(answer.get(f'{t} horas')['r_vec'], answer.get(f'{t} horas')['v_vec']))
    fig.savefig(f"images/{t}_horas.png")

