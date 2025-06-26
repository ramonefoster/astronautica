import numpy as np
import pandas as pd
import math

from utils import calc_elements, plot_orbit

def solve_gauss(r1_vec, r2_vec, delta_t, p0, grav_param=398600, solver="p"):
    if solver == 'fg':
        v1_vec = solve_fg(r1_vec, r2_vec, delta_t)
    if solver == 'p':
        v1_vec = via_p(r1_vec, r2_vec, delta_t, p0)
    if solver == 'u':
        v1_vec = universais(r1_vec, r2_vec, delta_t, grav_param)
    if solver == 'ub':
        v1_vec = universais_bi(r1_vec, r2_vec, delta_t, grav_param)
    if solver == 'pb':
        v1_vec = via_p_bi(r1_vec, r2_vec, delta_t, grav_param)

    print(f"v1_vec = {v1_vec}")

    orb_ele = calc_elements(r1_vec, v1_vec, grav_parameter=grav_param)
    # plot_orbit(**orb_ele)
    return orb_ele
    

def via_p(r1_vec, r2_vec, delta_t, p0, grav_param=398600):
    table = pd.DataFrame()
    # constantes
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    print(f"Magnitude for r1 = {r1} and r2 = {r2}")

    r_r = np.dot(r1_vec, r2_vec)
    cos_delta_f = r_r / (r1 * r2)
    delta_f = np.arccos(cos_delta_f)
    print(f"Delta_f = {delta_f}")

    # Constantes
    k = r1 * r2 * (1 - np.cos(delta_f))
    print(f"k: {k}")
    l = r1 + r2
    print(f"l: {l}")
    m = r1 * r2 * (1 + np.cos(delta_f))
    print(f"m: {m}")

    # Valores criticos de p
    pi = k / (l + np.sqrt(2*m))
    pii = k / (l - np.sqrt(2*m))
    print(f"critical values for p: pi = {pi} and pii = {pii}")

    print(f"Initial p values for iteration {(pi+pii)/2}")

    # Iterative solution for p
    if p0 == None:
        p0 = (pi+pii)/2
        
    p = p0

    tolerance = 1e-7
    max_iter = 100
    stop = False
    for i in range(max_iter):
        a = (m*k*p)/((2*m-l**2)*p**2 + 2*k*l*p - k**2)
        g = r1*r2*np.sin(delta_f)/(np.sqrt(grav_param*p))
        if a > 0:
            cosDeltaE = 1 - (k/(p*a))
            senDeltaE = -((k - p*l)/(p*np.sqrt(p*a)))*np.tan(delta_f/2)
            deltaE = np.arctan2(senDeltaE,cosDeltaE)
            if deltaE<0:
                deltaE = 2*np.pi + deltaE
            delta_tn = g + np.sqrt(a**3/grav_param)*(deltaE - senDeltaE)
            dDtDp = -(g/(2*p)) - (3/2)*a*(delta_tn - g)*((k**2 + (2*m - l**2)*p**2)/(m*k*p**2)) + np.sqrt(a**3/grav_param)*(2*k*senDeltaE)/(p*(k-l*p))
        else:
            coshDeltaF = 1 - (k/(p*a))
            deltaF = np.arccosh(coshDeltaF)
            delta_tn = g + np.sqrt(((-a)**3)/grav_param)*(np.sinh(deltaF) - deltaF)
            dDtDp = -(g/(2*p)) - (3/2)*a*(delta_tn - g)*((k**2 + (2*m - l**2)*p**2)/(m*k*p**2)) - np.sqrt((-a)**3/grav_param)*(2*k*np.sinh(deltaF))/(p*(k-l*p))

        # Update p        
        p_new = p + (delta_t - delta_tn)/dDtDp

        if abs(delta_t - delta_tn) < tolerance:
            print(f"Converged delta_t: {delta_tn} after {i+1} iterations")
            print(f"Iteration: {i+1}")
            stop = True

        p = p_new

        row = {
            'p (km)': p,
            'a (km)': a,
            'g (s)': g,
            'DeltaE (rad)': deltaE if a > 0 else deltaF,
            'Deltat (s)': delta_tn
        }
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        if stop:
            break
    else:
        print("Did not converge for p")

    f = 1 - (a/r1)*(1-cosDeltaE)
    g_dot = 1 - (r1/p)*(1-np.cos(delta_f))
    print(f"f: {f}, g_dot: {g_dot}")

    # Velocity
    v1_vec = (r2_vec - f*r1_vec)/g
    print(table)
    return v1_vec

def via_series_fg(r_vec, v_vec, dt, grav_parameter=398600):
    """    
    Parâmetros:
    r_vec (np.array): vetor de posição inicial (km)
    v_vec (np.array): vetor de velocidade inicial (km/s)
    dt (float): intervalo de tempo (segundos)
    grav_parameter (float): parâmetro gravitacional (km³/s²)
    
    Retorna:
    F, G
    """

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
    
    F = sum(Fn[n] * (dt**n) / math.factorial(n) for n in range(7))
    G = sum(Gn[n] * (dt**n) / math.factorial(n) for n in range(7))
    
    return F, G

def solve_fg(r1_vec, r2_vec, dt):
    table = pd.DataFrame()

    v1_vec = (r2_vec - r1_vec)/dt

    tolerance = 1e-4
    max_iter = 100
    stop = False

    for i in range(max_iter):
        f, g = via_series_fg(r1_vec, v1_vec, dt)
        v1_new = (r2_vec - f*r1_vec)/g

        if abs(np.linalg.norm(v1_new) - np.linalg.norm(v1_vec)) < tolerance:
            print(f"Iteration: {i+1}")
            stop = True
        
        v1_vec = v1_new
        row = {
            'f': f,
            'g': g,
            'v': np.linalg.norm(v1_new),
        }
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        
        if stop:
            break

    print(table)
    return v1_vec

def universais(r1_vec, r2_vec, delta_t, grav_param=398600):
    table = pd.DataFrame()
    # constantes
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    print(f"Magnitude for r1 = {r1} and r2 = {r2}")

    r_r = np.dot(r1_vec, r2_vec)
    cos_delta_f = r_r / (r1 * r2)
    delta_f = np.arccos(cos_delta_f)
    print(f"Delta_f = {delta_f}")

    A = np.sqrt((r1*r2)/(1 - np.cos(delta_f)))*np.sin(delta_f)

    z = 2.5
    
    tolerance = 1e-7
    max_iter = 100
    stop = False
    for i in range(max_iter):
        if z > 0:
            C = (1 - np.cos(np.sqrt(z))) / z
            S = (np.sqrt(z) - np.sin(np.sqrt(z))) / np.sqrt(z**3)
        elif round(z, 10) == 0:
            C = 0.5
            S = 1/6
        else:
            C = (1 - np.cosh(np.sqrt(-z))) / z
            S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / np.sqrt(-z**3)

        y = r1 + r2 - A*(1-z*S)/np.sqrt(C)
        X = np.sqrt(y/C)

        delta_tn = (X**3*S + A*np.sqrt(y))/np.sqrt(grav_param)

        # get new Z
        #Newton-Raphson
        func = delta_tn - delta_t
        dsdz = (1/(2*z)*(C-3*S))
        dcdz = (1/(2*z))*(1-z*S-2*C)
        fprime = (X**3*( dsdz - (3*S)/(2*C)*dcdz ) + (A/8)*(3*S*np.sqrt(y)/C + A/X))/np.sqrt(grav_param)
        z_new = z - func/fprime

        if abs(delta_t - delta_tn) < tolerance:
            print(f"Converged z: {z} after {i+1} iterations")
            print(f"Iteration: {i+1}")
            stop = True

        row = {
            'z': z_new,
            'y': y,
            'X': X,
            'Deltat (s)': delta_tn
        }
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        if stop:
            break
        
        z = z_new
    else:
        print("Did not converge for z")

    f = 1 - y/r1
    g = A*np.sqrt(y/grav_param)
    g_dot = 1 - y/r2
    print(f"f: {f}, g_dot: {g_dot}")

    # Velocity
    v1_vec = (r2_vec - f*r1_vec)/g
    print(table)
    return v1_vec

def bissecao(f, a, b, TOL, N):  
    i = 1  
    fa = f(a)  
    while (i <= N):  
        # iteracao da bissecao  
        p = a + (b-a)/2  
        fp = f(p)  
        # condicao de parada  
        if ((fp == 0) or ((b-a)/2 < TOL)):  
            return p  
        # bissecta o intervalo  
        i = i+1  
        if (fa * fp > 0):  
            a = p  
            fa = fp  
        else:  
            b = p  

    raise NameError('Num. max. de iter. excedido!')

def universais_bi(r1_vec, r2_vec, delta_t, grav_param=398600):
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    print(f"Magnitude for r1 = {r1} and r2 = {r2}")

    r_r = np.dot(r1_vec, r2_vec)
    cos_delta_f = r_r / (r1 * r2)
    delta_f = np.arccos(cos_delta_f)
    print(f"Delta_f = {delta_f}")

    A = np.sqrt((r1 * r2) / (1 - np.cos(delta_f))) * np.sin(delta_f)

    a = 0
    b = 10  # Adjust bounds based on the problem
    tolerance = 1e-7
    max_iter = 100

    # Function for bisection method to find the correct z
    def find_z(z):
        if z > 0:
            C = (1 - np.cos(np.sqrt(z))) / z
            S = (np.sqrt(z) - np.sin(np.sqrt(z))) / np.sqrt(z**3)
        elif round(z, 10) == 0:
            C = 0.5
            S = 1/6
        else:
            C = (1 - np.cosh(np.sqrt(-z))) / z
            S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / np.sqrt(-z**3)

        y = r1 + r2 - A * (1 - z * S) / np.sqrt(C)
        X = np.sqrt(y / C)
        delta_tn = (X**3 * S + A * np.sqrt(y)) / np.sqrt(grav_param)
        row = {
            'z': z,
            'y': y,
            'X': X,
            'Deltat (s)': delta_tn
        }
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        return delta_tn - delta_t

    try:
        z_solution = bissecao(find_z, a, b, tolerance, max_iter)
        print(f"Converged z: {z_solution}")
    except NameError as e:
        print(e)
    
    z = z_solution
    if z > 0:
        C = (1 - np.cos(np.sqrt(z))) / z
        S = (np.sqrt(z) - np.sin(np.sqrt(z))) / np.sqrt(z**3)
    elif abs(z) < 1e-8:
        C = 0.5
        S = 1 / 6
    else:
        C = (1 - np.cosh(np.sqrt(-z))) / z
        S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / np.sqrt((-z)**3)

    y = r1 + r2 - A * (1 - z * S) / np.sqrt(C)
    X = np.sqrt(y / C)

    f = 1 - y / r1
    g = A * np.sqrt(y / grav_param)
    g_dot = 1 - y / r2

    print(f"f = {f}, g = {g}, g_dot = {g_dot}")

    # Compute initial velocity vector
    v1_vec = (r2_vec - f * r1_vec) / g

    return v1_vec

def via_p_bi(r1_vec, r2_vec, delta_t, p0=None, grav_param=398600.0):
    table = pd.DataFrame()

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    print(f"Magnitude for r1 = {r1} and r2 = {r2}")

    r_r = np.dot(r1_vec, r2_vec)
    cos_delta_f = r_r / (r1 * r2)
    delta_f = np.arccos(cos_delta_f)
    print(f"Delta_f = {delta_f}")

    # Constantes
    k = r1 * r2 * (1 - np.cos(delta_f))
    l = r1 + r2
    m = r1 * r2 * (1 + np.cos(delta_f))

    # Valores críticos de p
    pi = k / (l + np.sqrt(2 * m))
    pii = k / (l - np.sqrt(2 * m))
    print(f"Valores críticos de p: pi = {pi}, pii = {pii}")

    if p0 is None:
        p0 = (pi + pii) / 2
    
    # Define função a ser zerada
    def funcao_delta_t(p):
        try:            
            a = (m*k*p)/((2*m-l**2)*p**2 + 2*k*l*p - k**2)
            g = r1*r2*np.sin(delta_f)/(np.sqrt(grav_param*p))
            if a > 0:
                cosDeltaE = 1 - (k/(p*a))
                senDeltaE = -((k - p*l)/(p*np.sqrt(p*a)))*np.tan(delta_f/2)
                deltaE = np.arctan2(senDeltaE,cosDeltaE)
                if deltaE<0:
                    deltaE = 2*np.pi + deltaE
                delta_tn = g + np.sqrt(a**3/grav_param)*(deltaE - senDeltaE)
                dDtDp = -(g/(2*p)) - (3/2)*a*(delta_tn - g)*((k**2 + (2*m - l**2)*p**2)/(m*k*p**2)) + np.sqrt(a**3/grav_param)*(2*k*senDeltaE)/(p*(k-l*p))
            else:
                coshDeltaF = 1 - (k/(p*a))
                deltaF = np.arccosh(coshDeltaF)
                delta_tn = g + np.sqrt(((-a)**3)/grav_param)*(np.sinh(deltaF) - deltaF)
                dDtDp = -(g/(2*p)) - (3/2)*a*(delta_tn - g)*((k**2 + (2*m - l**2)*p**2)/(m*k*p**2)) - np.sqrt((-a)**3/grav_param)*(2*k*np.sinh(deltaF))/(p*(k-l*p))

            row = {
                'p': p,
                'a': a,
                'g': g,
                'Deltat (s)': delta_tn
            }
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
            return delta_tn - delta_t
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return 1e6  # valor fora do domínio ou divisão por zero

    # Intervalo para bisseção — pode ser ajustado conforme o problema
    a_p = max(1e-3, pi * 0.1)
    b_p = pii * 1.5
    tolerance = 1e-4
    max_iter = 100

    try:
        p_final = bissecao(funcao_delta_t, a_p, b_p, tolerance, max_iter)
        print(f"Converged p: {p_final}")
    except NameError as e:
        print(e)
        return None

    # Recalcula as variáveis com p_final
    p = p_final
    a = (m * k * p) / ((2 * m - l**2) * p**2 + 2 * k * l * p - k**2)
    g = r1 * r2 * np.sin(delta_f) / np.sqrt(grav_param * p)

    if a > 0:
        cosDeltaE = 1 - (k / (p * a))
        senDeltaE = -((k - p * l) / (p * np.sqrt(p * a))) * np.tan(delta_f / 2)
        deltaE = np.arctan2(senDeltaE, cosDeltaE)
        if deltaE < 0:
            deltaE += 2 * np.pi
        delta_tn = g + np.sqrt(a**3 / grav_param) * (deltaE - senDeltaE)
        delta_val = deltaE
    else:
        coshDeltaF = 1 - (k / (p * a))
        deltaF = np.arccosh(coshDeltaF)
        delta_tn = g + np.sqrt((-a)**3 / grav_param) * (np.sinh(deltaF) - deltaF)
        delta_val = deltaF

    f = 1 - (a/r1)*(1-cosDeltaE)
    g_dot = 1 - (r1/p)*(1-np.cos(delta_f))
    print(f"f: {f}, g_dot: {g_dot}")

    # Velocity
    v1_vec = (r2_vec - f*r1_vec)/g

    row = {
        'p (km)': p,
        'a (km)': a,
        'g (s)': g,
        'Delta (rad)': delta_val,
        'Deltat (s)': delta_tn
    }
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    print(table)

    return v1_vec

if __name__ == "__main__":
    grav_param = 398600
    
    # EXAMPLE
    # r1_vec = np.array([5, 0, 5]) * 1e3
    # r2_vec = np.array([4, 4, 5]) * 1e3
    # dt = 45 * 60
    # p0 = 1000
    
    #Example2
    # r1_vec = np.array([5, 10, 2.1]) * 1e3
    # r2_vec = np.array([-14.6, 2.5, 7]) * 1e3
    # dt = 60 * 60
    # p0 = 10000

    #Sample
    # r1_vec = np.array([6, 0, 4])*1e3 # u.d.
    # r2_vec = np.array([5, 0, 5])*1e3 # u.d.
    # dt = 3 * 60
    # p0 = None

    # # Exercicio 5
    r1_vec = np.array([.5, .6, .7]) * 6378 # u.d.
    r2_vec = np.array([0, -1, 0]) * 6378 # u.d.
    dt = 25 * 60
    p0 = None

    orb_u = solve_gauss(r1_vec, r2_vec, dt, p0, grav_param=grav_param, solver="pb")
    orb = solve_gauss(r1_vec, r2_vec, dt, p0, grav_param=grav_param, solver="p")
    # orb_u = solve_gauss(r1_vec, r2_vec, dt, p0, grav_param=grav_param, solver="u")
    print("Difference between orb and orb_u:")
    for key in orb:
        if key in orb_u:
            print(f"{key}: {orb[key]} - {orb_u[key]} = {orb[key] - orb_u[key]}")
    # print(orb)
    

