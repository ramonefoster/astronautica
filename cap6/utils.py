import numpy as np

def kepler_f(E, M, e):
    return M - E + e * np.sin(E)

def kepler_fp(E, e):
    return -1 + e * np.cos(E)

# Newton–Raphson for elliptical orbits
def newton_raphson(E0, M, e, tol=1e-15, max_iter=100):
    E = E0
    for i in range(max_iter):
        f  = kepler_f(E, M, e)
        fp = kepler_fp(E, e)
        dE = -f / fp
        En = E + dE
    
        if abs(En - E) < tol:
            break
        
        E = En
    return E

import numpy as np

def calc_elements(r_vec, v_vec, grav_parameter=398600):
    """
    Calcula os elementos orbitais a partir dos vetores posição e velocidade.
    
    Args:
        r_vec (`np.array`): Vetor posição (km).
        v_vec (`np.array`): Vetor velocidade (km/s).
        grav_parameter (`float`): Parâmetro gravitacional do corpo central (km³/s²). Default is 398600 (Terra).

    Returns:
        orbital_elements (`dict`): Dicionário com os elementos orbitais:
            - 'a': semi-eixo maior (km)
            - 'e': excentricidade (adimensional)
            - 'i': inclinação (graus)
            - 'Omega': longitude do nodo ascendente (graus)
            - 'w': argumento do pericentro (graus)
            - 'f': anomalia verdadeira (graus)
            - 'h_vec': vetor momentum angular específico (km²/s)
    """
    # Constantes
    μ = grav_parameter
        
    # Passo 1: Determinação de r e v
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    # Passo 2: Determinação das Integrais Primeiras
    # Energia específica
    ε = 0.5 * v**2 - μ/r
    
    # Momentum angular específico
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # Vetor Laplace Runge-Lenz B
    B_vec = np.cross(v_vec, h_vec) - μ * r_vec / r
    B = np.linalg.norm(B_vec)
    
    # Passo 3: Determinação do vetor nodal N
    k = np.array([0, 0, 1])
    N_vec = np.cross(k, h_vec)
    N = np.linalg.norm(N_vec)
    
    # Passo 4: Determinação do parâmetro adimensional Q
    Q = (r * v**2) / μ
    
    # Passo 5: Determinação do semi-eixo maior
    a = r / (2 - Q)
    
    # Passo 6: Determinação da excentricidade
    e = B / μ
    
    # Passo 7: Determinação da inclinação do plano da órbita
    I = np.degrees(np.arccos(np.dot(k, h_vec)/h))
    
    # Passo 8: Determinação da longitude do nodo ascendente
    i = np.array([1, 0, 0])
    Omegao_cos = np.dot(i, N_vec)/N
    Omegao = np.degrees(np.arccos(Omegao_cos))
    
    # Determinar o quadrante (j·N < 0 -> 4º quadrante)
    j = np.array([0, 1, 0])
    if np.dot(j, N_vec) < 0:
        Omegao = 360 - Omegao
        
    # Passo 9: Determinação do argumento do pericentro
    w_cos = np.dot(B_vec, N_vec)/(B*N)
    w = np.degrees(np.arccos(w_cos))
    
    # Determinar o quadrante (k·B < 0 -> 4º quadrante)
    if np.dot(k, B_vec) < 0:
        w = 360 - w
    
    # Passo 10: Determinação da anomalia verdadeira
    f_cos = np.dot(B_vec, r_vec)/(B*r)
    f = np.degrees(np.arccos(f_cos))
    # Determinar o quadrante (r·v > 0 -> 2º quadrante)
    if np.dot(r_vec, v_vec) < 0:
        f = 360 - f

    orbital_elements = {
        'a': a,
        'e': e,
        'I': I,
        'Omega': Omegao,
        'w': w,
        'f': f,
        'h': h_vec
    }
    return orbital_elements

import pandas as pd

def p_NR(r1, r2, df, k, l, m, dt, tolerance, guess, maxIte, printMode=1):
    mu = 398600  # [km^3/s^2] standard gravitational parameter
    p = guess
    numberOfIterations = 0
    error = 100

    p_vec = []
    a_vec = []
    g_vec = []
    dX_vec = []
    dtn_vec = []
    error_vec = []

    while error > tolerance:
        denominator = ((2 * m - l ** 2) * p ** 2 + 2 * k * l * p - k ** 2)
        a = m * k * p / denominator
        g = r1 * r2 * np.sin(df) / np.sqrt(mu * p)

        if a > 0:
            dX = np.arctan2(-np.tan(df / 2) * (k - l * p) / (p * np.sqrt(p * a)), 1 - k / (p * a))
            if dX < 0:
                dX += 2 * np.pi
            dtn = g + np.sqrt(a ** 3 / mu) * (dX - np.sin(dX))
            ddt = (-g / (2 * p)
                   - (3 / 2) * a * (dtn - g) * ((k ** 2 + (2 * m - l ** 2) * p ** 2) / (m * k * p ** 2))
                   + np.sqrt(a ** 3 / mu) * 2 * k * np.sin(dX) / (p * (k - l * p)))
        else:
            dX = np.arccosh(1 - k / (p * a))
            dtn = g + np.sqrt((-a) ** 3 / mu) * (np.sinh(dX) - dX)
            ddt = (-g / (2 * p)
                   - (3 / 2) * a * (dtn - g) * ((k ** 2 + (2 * m - l ** 2) * p ** 2) / (m * k * p ** 2))
                   + np.sqrt((-a) ** 3 / mu) * 2 * k * np.sinh(dX) / (p * (k - l * p)))

        numberOfIterations += 1
        p_try = p + (dt - dtn) / ddt

        p_vec.append(p_try)
        a_vec.append(a)
        g_vec.append(g)
        dX_vec.append(dX)
        dtn_vec.append(dtn)
        error = abs(dt - dtn)
        error_vec.append(error)

        p = p_try

        if numberOfIterations == maxIte:
            print(f"Did not converge within {maxIte} iterations.")
            return None, None, numberOfIterations

    # Optional detailed printout
    T, Tpres = None, None
    if printMode == 1:
        T = pd.DataFrame({
            'Iteracao': np.arange(1, numberOfIterations + 1),
            'p [km]': np.round(p_vec, 4),
            'a [km]': np.round(a_vec, 4),
            'g [s]': np.round(g_vec, 4),
            'ΔE [rad]': np.round(dX_vec, 4),
            'Δt [s]': np.round(dtn_vec, 4),
            'Δt error [s]': np.round(error_vec, 4)
        })
        Tpres = pd.DataFrame({
            'Iteracao': np.arange(1, numberOfIterations + 1),
            'p [km]': p_vec,
            'a [km]': a_vec,
            'g [s]': g_vec,
            'ΔE [rad]': dX_vec,
            'Δt [s]': dtn_vec,
            'Δt error [s]': error_vec
        })
        print(T)

    return T, Tpres, numberOfIterations