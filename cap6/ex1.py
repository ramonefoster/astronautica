import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import calc_elements, p_NR

def solve_gauss(r1_vec, r2_vec, delta_t, p0, grav_param=398600):
    # Misc
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

    tolerance = 1e-4
    max_iter = 100
    stop = False
    for i in range(max_iter):
        a = (m*k*p)/((2*m-l**2)*p**2 + 2*k*l*p - k**2)
        g = r1*r2*np.sin(delta_f)/(np.sqrt(grav_param*p))
        if a > 0:
            cosDeltaE = 1 - (k/(p*a))
            senDeltaE = -((k - p*l)/(p*np.sqrt(p*a)))*np.tan(delta_f/2)
            deltaE = np.arctan2(senDeltaE,cosDeltaE)
            delta_tn = g + np.sqrt(a**3/grav_param)*(deltaE - senDeltaE)
            dDtDp = -(g/(2*p)) - (3/2)*a*(delta_tn - g)*((k**2 + (2*m - l**2)*p**2)/(m*k*p**2)) + np.sqrt(a**3/grav_param)*(2*k*senDeltaE)/(p*(k-l*p))
        else:
            coshDeltaF = 1 - (k/(p*a))
            deltaF = np.arccosh(coshDeltaF)
            delta_tn = g + np.sqrt(((-a)**3)/grav_param)*(np.sinh(deltaF) - deltaF)
            dDtDp = -g/(2*p) - (3*a*(delta_tn - g)/2)*((k**2 + (2*m - l**2)*p**2)/(m*k*p**2)) - np.sqrt((-a)**3/grav_param)*(2*k*np.sinh(deltaE))/(p*(k-l*p))

        # Update p        
        p_new = p + (delta_t - delta_tn)/dDtDp

        if abs(delta_t - delta_tn) < tolerance:
            print(f"Converged delta_t: {delta_tn} after {i+1} iterations")
            break

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

    print(f"g: {g}")
    p_NR(r1, r2, delta_f, k, l, m, delta_t, tolerance, p0, 100)
    # Coeficientes de Lagrange
    f = 1 - (a/r1)*(1-cosDeltaE)
    g_dot = 1 - (r1/p)*(1-np.cos(delta_f))
    print(f"f: {f}, g_dot: {g_dot}")

    # Velocity
    v1_vec = (r2_vec - f*r1_vec)/g
    print(f"v1_vec = {v1_vec}")
    v2_vec = (g_dot*r2_vec - r1_vec)/g
    print(f"v2_vec = {v2_vec}")

    # excentricidade
    e = np.sqrt(1 - (p/a))
    print(f"Excentricidade e: {e}")

    # raio pericentro
    rp = a*(1-e)
    print(f"Raio do pericentro rp: {rp}")

    # semi-eixo maior correspondente à elipse de mínima energia
    am = m/(2*l - 2*np.sqrt(l**2 - 2*m))
    print(f"am de minima energia: {am}")

    if rp < 6378:
        print("Voo suborbital")

    # Constnate A
    A = np.sqrt((r1*r2)/(1 - np.cos(delta_f))) * np.sin(delta_f)
    print(f"Constant A: {A}")

    print(table)
    
    p_values = np.linspace(0.1, 5, 300)
    t_values = []
    a_values = []
    # delta_f = 360-delta_f
    for p in p_values:
        # if p <= pi:
        #     p_values = np.delete(p_values, np.where(p_values == p))
        #     continue
        a = p / (1 - e**2)
        g = r1*r2*np.sin(delta_f)/(np.sqrt(grav_param*p))
        if a > 0:
            cosDeltaE = 1 - (k/(p*a))
            senDeltaE = -(k - p*l)*np.tan(delta_f/2)/(p*np.sqrt(p*a))
            deltaE = np.arctan2(senDeltaE,cosDeltaE)
            delta_tn = g + np.sqrt(a**3/grav_param)*(deltaE - senDeltaE)
            
        else:
            # if a == 0:
            #     p_values = np.delete(p_values, np.where(p_values == p))
            #     continue
            # if p == pii:
            #     p_values = np.delete(p_values, np.where(p_values == p))
            #     continue
            
            coshDeltaF = 1 - (k/(p*a))
            deltaF = np.arccosh(coshDeltaF)
            delta_tn = g + np.sqrt(((-a)**3)/grav_param)*(np.sinh(deltaF) - deltaF)

        a_values.append(a)
        t_values.append(delta_tn)

    plot_a_vs_p(p_values, a_values, pi, pii)
    # plot_t_vs_p(p_values, t_values)
    print(calc_elements(r1_vec, v1_vec, grav_parameter=grav_param))

def plot_a_vs_p(p, a, pi, pii):
    plt.figure(figsize=(8, 5))
    plt.plot(p, a)
    plt.axvline(pi, color='r', linestyle='--', label='pi (assíntota)')
    plt.axvline(pii, color='r', linestyle='--', label='pii (assíntota)')
    plt.xlabel('Semi-latus rectum p (km)')
    plt.ylabel('Semi-major axis a (km)')
    plt.title('Semi-major axis a vs Semi-latus rectum p')
    plt.grid(True)
    plt.show()

def plot_t_vs_p(p, t):
    plt.figure(figsize=(8, 5))
    plt.plot(p, t)
    plt.xlabel('Semi-latus rectum p (km)')
    plt.ylabel('Tempo de voo - t (s)')
    plt.title('Semi-major axis a vs Semi-latus rectum p')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # r1_vec = np.array([1, 1, 0]) # u.d.
    # r2_vec = np.array([-1, 1, 0]) # u.d.
    # dt = 500
    # p0 = None
    # grav_param = 1

    # EXAMPLE
    # r1_vec = np.array([5, 0, 5]) * 1e3
    # r2_vec = np.array([4, 4, 5]) * 1e3
    # dt = 45 * 60
    # p0 = 1000
    # grav_param = 389600

    #Example2
    # r1_vec = np.array([5, 10, 2.1]) * 1e3
    # r2_vec = np.array([-14.6, 2.5, 7]) * 1e3
    # dt = 60 * 60
    # p0 = 10000
    grav_param = 398600

    # Exercicio 5
    r1_vec = np.array([.5*6378, .6*6378, .7*6378]) # u.d.
    r2_vec = np.array([0, -1*6378, 0]) # u.d.
    dt = 25 * 60
    p0 = None
    # grav_param = 1
    solve_gauss(r1_vec, r2_vec, dt, p0, grav_param=grav_param)
    

