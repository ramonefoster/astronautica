import numpy as np
import pandas as pd

def solve_gauss(r1_vec, r2_vec, delta_t, p0, grav_param=398600):
    # Misc
    data = {'p (km)': [], 'a (km)': [], 'g (s)': [], 'DeltaE (rad)': [], 'Deltat(s)': []}
    table = pd.DataFrame(data)

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
    p = p0
    tolerance = 1e-4
    max_iter = 100

    for i in range(max_iter):
        a = (m*k*p)/((2*m-l**2)*p**2 + 2*k*l*p - k**2)
        g = r1*r2*np.sin(delta_f)/(np.sqrt(grav_param*p))
        if a > 0:
            cosDeltaE = 1 - (k/(p*a))
            senDeltaE = -(k - p*l)*np.tan(delta_f/2)/(p*np.sqrt(p*a))
            deltaE = np.arctan2(senDeltaE,cosDeltaE)
            delta_tn = g + np.sqrt(a**3/grav_param)*(deltaE - senDeltaE)
            dDtDp = -g/(2*p) - (3*a*(delta_t - g)/2)*((k**2 + (2*m - l**2)*p**2)/(m*k*p**2)) + np.sqrt(a**3/grav_param)*(2*k*senDeltaE)/(p*(k-l*p))
        else:
            coshDeltaF = 1 - (k/(p*a))
            deltaF = np.arccosh(coshDeltaF)
            delta_tn = g + np.sqrt(((-a)**3)/grav_param)*(np.sinh(deltaF) - deltaF)
            dDtDp = -g/(2*p) - (3*a*(delta_t - g)/2)*((k**2 + (2*m - l**2)*p**2)/(m*k*p**2)) + np.sqrt((-a)**3/grav_param)*(2*k*np.sinh(deltaE))/(p*(k-l*p))

        row = {
            'p (km)': p,
            'a (km)': a,
            'g (s)': g,
            'DeltaE (rad)': deltaE if a > 0 else deltaF,
            'Deltat(s)': delta_tn
        }
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)

        # Update p        
        p_new = p + (delta_t - delta_tn)/dDtDp
        if abs(p_new - p) < tolerance:
            print(f"Converged p: {p_new} after {i+1} iterations")
            break
        p = p_new
    else:
        print("Did not converge for p")

    # Coeficientes de Lagrange
    f = 1 - (a/r1)*(1-cosDeltaE)

    # Velocity
    v1_vec = (r2_vec - f*r1_vec)/g
    print(f"v1_vec = {v1_vec}")

    # excentricidade
    e = np.sqrt(1 - (p/a))
    print(f"Excentricidade e: {e}")

    # raio pericentro
    rp = a*(1-e)
    print(f"Raio do pericentro rp: {rp}")

    if rp < 6378:
        print("Voo suborbital")

    # Constnate A
    A = np.sqrt((r1*r2)/(1 - np.cos(delta_f))) * np.sin(delta_f)
    print(f"Constant A: {A}")

if __name__ == "__main__":
    r1_vec = np.array([1, 1, 0]) # u.d.
    r2_vec = np.array([-1, 1]) # u.d.
    dt = 0
    p0 = 0

    # EXAMPLE
    r1_vec = np.array([5, 0, 5]) * 1e3
    r2_vec = np.array([4, 4, 5]) * 1e3
    dt = 45 * 60
    p0 = 1000
    solve_gauss(r1_vec, r2_vec, dt, p0)
    