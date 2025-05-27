"""

Author: Ramon Carlos Gargalhone
@ Instituto Tecnologico de Aeronautica - ITA

Esse script foi baseado nos Exercicios Resolvidos do Capitulo 3, do livro: 'Fundamentos de Astronáutica e suas Aplicações Volume I', 
                                        de Maria Cecília França de Paula Santos Zanardi, Sandro da Silva Fernandes

Semi-eixo maior (a): Este é o semi-eixo da elipse que define a órbita. É metade da distância máxima entre o objeto e o centro da órbita (foco). 

Excentricidade (e): A excentricidade determina a forma da órbita, sendo 0 para uma órbita perfeitamente circular e 1 para uma órbita parabólica. 

Inclinação (I): A inclinação é o ângulo entre o plano orbital do objeto e o plano de referência (normalmente o equador do corpo central). 
I = 0, indica orbita equatorial, enquanto I = 90 indica uma órbita polar.

Longitude do nodo ascendente (Ω): Este é o ângulo no plano de referência a partir de uma direção de referência até o ponto onde a órbita 
cruza o plano de referência de sul para norte.

Argumento do pericentro (ω): O argumento do pericentro é o ângulo medido ao longo da órbita a partir do nodo ascendente até o pericentro 
(ponto mais próximo do corpo central).

Anomalia verdadeira (f): A anomalia verdadeira é o ângulo medido a partir do pericentro até a posição atual do objeto na órbita. 
"""
import numpy as np
import os
from termcolor import colored

os.system('color')

def print_step(step_num, description, value=None, unit=None):
    print(f"\n=== Passo {step_num} ===")
    print(colored(description, 'yellow'))
    if value is not None:
        if unit is not None:
            print(f"Resultado: {colored(value, 'green')} {unit}")
        else:
            print(f"Resultado: {colored(value, 'green')}")

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
    
    print("===== Cálculo dos Elementos Orbitais =====")
    
    # Passo 1: Determinação de r e v
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    print_step(1, "Cálculo das magnitudes dos vetores posição e velocidade r=sqrt(r_i²+r_j²+r_k²) e v=sqrt(v_i²+v_j²+v_k²)", 
              (f"r = {r:.3f} km", f"v = {v:.3f} km/s"))
    
    # Passo 2: Determinação das Integrais Primeiras
    # Energia específica
    ε = 0.5 * v**2 - μ/r
    print_step(2, "Cálculo da energia específica (ε = ½v² - μ/r)", f"{ε:.2f} km²/s²")
    
    # Momentum angular específico
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    print_step(2, "Cálculo do vetor momentum angular (h = r × v)", 
              (f"Vetor h: {h_vec} km²/s", f"Magnitude h: {h:.3f} km²/s"))
    
    # Vetor Laplace Runge-Lenz B
    B_vec = np.cross(v_vec, h_vec) - μ * r_vec / r
    B = np.linalg.norm(B_vec)
    print_step(2, "Cálculo do Vetor Laplace Runge-Lenz (B = v × h - μr/r)", 
              (f"Vetor B: {B_vec} km³/s²", f"Magnitude B: {B:.3f} km³/s²"))
    
    # Passo 3: Determinação do vetor nodal N
    k = np.array([0, 0, 1])
    N_vec = np.cross(k, h_vec)
    N = np.linalg.norm(N_vec)
    print_step(3, "Cálculo do vetor nodal (N = k × h)", 
              (f"Vetor N: {N_vec} km²/s", f"Magnitude N: {N:.3f} km²/s"))
    
    # Passo 4: Determinação do parâmetro adimensional Q
    Q = (r * v**2) / μ
    print_step(4, "Cálculo do parâmetro adimensional Q = rv²/μ", f"{Q:.3f}")
    
    # Passo 5: Determinação do semi-eixo maior
    a = r / (2 - Q)
    print_step(5, "Cálculo do semi-eixo maior (a = r/(2-Q))", f"{a:.2f} km")
    
    # Passo 6: Determinação da excentricidade
    e = B / μ
    print_step(6, "Cálculo da excentricidade (e = B/μ)", f"{e:.3f}")
    
    # Passo 7: Determinação da inclinação do plano da órbita
    I = np.degrees(np.arccos(np.dot(k, h_vec)/h))
    print_step(7, "Cálculo da inclinação do plano da órbita (I = arccos(k·h/h))", f"{I:.2f}°")
    
    # Passo 8: Determinação da longitude do nodo ascendente
    i = np.array([1, 0, 0])
    Omegao_cos = np.dot(i, N_vec)/N
    Omegao = np.degrees(np.arccos(Omegao_cos))
    
    # Determinar o quadrante (j·N < 0 -> 4º quadrante)
    j = np.array([0, 1, 0])
    if np.dot(j, N_vec) < 0:
        Omegao = 360 - Omegao
    
    print_step(8, "Cálculo da longitude do nodo ascendente (Ω = arccos(i·N/N)))", f"{Omegao:.2f}°")
    
    # Passo 9: Determinação do argumento do pericentro
    w_cos = np.dot(B_vec, N_vec)/(B*N)
    w = np.degrees(np.arccos(w_cos))
    
    # Determinar o quadrante (k·B < 0 -> 4º quadrante)
    if np.dot(k, B_vec) < 0:
        w = 360 - w
    print_step(9, "Cálculo do argumento do pericentro (ω = arccos(B·N/BN))", f"{w:.2f}°")
    
    # Passo 10: Determinação da anomalia verdadeira
    f_cos = np.dot(B_vec, r_vec)/(B*r)
    f = np.degrees(np.arccos(f_cos))
    
    # Determinar o quadrante (r·v > 0 -> 2º quadrante)
    if np.dot(r_vec, v_vec) > 0:
        f = 360 - f
    print_step(10, "Cálculo da anomalia verdadeira (f = arccos(B·r/Br))", f"{f:.2f}°")
    
    print("\n===== Resultados Finais =====")
    print(f"Semi-eixo maior (a): {colored(f'{a:.2f} km', 'green')} ")
    print(f"Excentricidade (e): {colored(f'{e:.3f}', 'green')}")
    print(f"Inclinação (I): {colored(f'{I:.2f}°', 'green')}")
    print(f"Longitude do nodo ascendente (Ω): {colored(f'{Omegao:.2f}°', 'green')}")
    print(f"Argumento do pericentro (ω): {colored(f'{w:.2f}°', 'green')}")
    print(f"Anomalia verdadeira (f): {colored(f'{f:.2f}°', 'green')}")

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


if __name__ == "__main__":
    from plots.plot_orbit import plot_orbit 

    # # SAMPLE 1    
    r_vec = np.array([1.0, 0.0, np.sqrt(3)]) * 1e4  # km (vetor posição)
    v_vec = np.array([2.0, 4.0, 4.0])  # km/s (vetor velocidade)
    μ = 398600  # km³/s² (parâmetro gravitacional da Terra)

    # EX 1:
    r_vec = np.array([1.0, -1.0, -1])  # u.d (vetor posição)
    v_vec = np.array([.4, .2, .4])  # u.d (vetor velocidade)
    μ = 1 

    # # Ex 2:
    # r_vec = np.array([6.0, 6.0, 0]) * 1e3  # km (vetor posição)
    # v_vec = np.array([-4.0, -4.0, 6.0])  # km/s (vetor velocidade)
    # μ = 398600  # km³/s² (parâmetro gravitacional da Terra)

    orbital_elements = calc_elements(r_vec, v_vec, grav_parameter=μ)    
    plot_orbit(**orbital_elements)