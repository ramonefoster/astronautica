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
    
    # print("\n===== Resultados Finais =====")
    # print(f"Semi-eixo maior (a): {(f'{a:.2f} km')} ")
    # print(f"Excentricidade (e): {(f'{e:.4f}')}")
    # print(f"Inclinação (I): {(f'{I:.2f}°')}")
    # print(f"Longitude do nodo ascendente (Ω): {(f'{Omegao:.2f}°')}")
    # print(f"Argumento do pericentro (ω): {(f'{w:.2f}°')}")
    # print(f"Anomalia verdadeira (f): {(f'{f:.2f}°')}")

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