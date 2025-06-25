import numpy as np
from rv_to_elements import calc_elements
from utils import newton_raphson_universal, plot_orbit

def via_anomalia_universal(r_vec, v_vec, dt, orbit_elem, grav_parameter=398600):
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
    a = t0_orbit_elem['a']

    for t in dt:
        deltaT = t * 3600  # Convertendo horas para segundos
        if t0_orbit_elem['e'] < 1:
            guess = np.sqrt(grav_parameter)*deltaT/a
        elif round(t0_orbit_elem['e'], 3) == 1:
            p = np.linalg.norm(np.cross(r_vec, v_vec))**2 / grav_parameter
            Mp = np.sqrt(grav_parameter/p**3) * deltaT
            D = np.sqrt(p) * ((3*Mp + np.sqrt(1 + 9*Mp**2))**(1/3) - (3*Mp + np.sqrt(1 + 9*Mp**2))**(-1/3))
            guess = Mp-D
        elif t0_orbit_elem['e'] > 1:
            guess = deltaT*np.sqrt(-a)*np.log((-2*grav_parameter*deltaT)/(a*(np.dot(r_vec, v_vec) + deltaT + np.sqrt(-grav_parameter*a)*(1-r0/a)))) 

        X, z = newton_raphson_universal(guess, deltaT, a, r_vec, v_vec, grav_parameter)
        r = a + a*( (np.dot(r_vec, v_vec)*np.sin(X/np.sqrt(a)))/(np.sqrt(grav_parameter*a)) + 
                (r0/a - 1)*np.cos(X/np.sqrt(a)) )
                
        if z > 0:
            C = (1 - np.cos(np.sqrt(z))) / z
            S = (np.sqrt(z) - np.sin(np.sqrt(z))) / np.sqrt(z**3)
        elif round(z, 10) == 0:
            C = 0.5
            S = 1/6
        else:
            C = (1 - np.cosh(np.sqrt(-z))) / z
            S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / np.sqrt(-z**3)
            
        F = 1 - (X**2/r0)*C
        G = deltaT - (X**3/np.sqrt(grav_parameter))*S
        F_dot = np.sqrt(grav_parameter)*X*(z*S -1)/(r*r0)
        G_dot = 1 - (X**2/r)*C

        # Determinacao dos vetores r e v
        r_vec_t = F * r_vec.T + G * v_vec.T
        v_vec_t = F_dot * r_vec.T + G_dot * v_vec.T

        answer[f"{t} horas"] = {
                "r_vec": r_vec_t,
                "v_vec": v_vec_t
            }
        
    return answer

# # Dados do exercicio
# r_vec = np.array([1.0, 1.0, 0]) * 1e4  # km (vetor posição)
# v_vec = np.array([2.0, 4.0, 4.0])  # km/s (vetor velocidade)

# dt = [0.5, 1, 2] # Horas

# # Adaptado da lista 3
# orbit_elem = calc_elements(r_vec, v_vec)

# # Constants
# GRAV_PARAM = 398600 # km³/s² (Terra)

# answer = via_anomalia_universal(r_vec, v_vec, dt, orbit_elem, grav_parameter=GRAV_PARAM)
# print(answer)
# # for t in dt:    
#     fig = plot_orbit(**calc_elements(answer.get(f'{t} horas')['r_vec'], answer.get(f'{t} horas')['v_vec']))
#     fig.savefig(f"images/{t}_horas.png")


