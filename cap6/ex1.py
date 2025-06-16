import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import calc_elements, p_NR, plot_orbit

def solve_gauss(r1_vec, r2_vec, grav_param=1):
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

    p_values = np.linspace(0.1, 5, 300)
    t_values = []
    a_values = []

    delta_f = 2*np.pi-delta_f
    for p in p_values:
        if p <= pi or p >= pii:
            p_values = np.delete(p_values, np.where(p_values == p))
            continue

        a = (m*k*p)/((2*m-l**2)*p**2 + 2*k*l*p - k**2)
        g = r1*r2*np.sin(delta_f)/(np.sqrt(grav_param*p))
        if a > 0:
            cosDeltaE = 1 - (k/(p*a))
            senDeltaE = -(k - p*l)*np.tan(delta_f/2)/(p*np.sqrt(p*a))
            deltaE = np.arctan2(senDeltaE,cosDeltaE)
            if deltaE<0:
                deltaE = 2*np.pi + deltaE
            delta_tn = g + np.sqrt(a**3/grav_param)*(deltaE - senDeltaE)
            
        else:
            # p_values = np.delete(p_values, np.where(p_values == p))
            # continue
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

    
    print(f"amin energia: {min(a_values)}")
    plot_a_vs_p(p_values, a_values, pi, pii)
    plot_t_vs_p(p_values, t_values)

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
    grav_param = 1
    

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

    #Exercicio 1
    r1_vec = np.array([1, 1, 0]) # u.d.
    r2_vec = np.array([-1, 1, 0]) # u.d.

    # # Exercicio 5
    # r1_vec = np.array([.5*6378, .6*6378, .7*6378]) # u.d.
    # r2_vec = np.array([0, -1*6378, 0]) # u.d.
    # dt = 25 * 60
    # p0 = None

    solve_gauss(r1_vec, r2_vec, grav_param=grav_param)
    

