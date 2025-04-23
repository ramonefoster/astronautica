import numpy as np

def print_step(step_num, description, value=None, unit=None, matrix=None):
    print(f"\n=== Passo {step_num} ===")
    print(description)
    if value is not None:
        if unit is not None:
            print(f"Resultado: {value} {unit}")
        else:
            print(f"Resultado: {value}")
    if matrix is not None:
        print("Matriz:")
        print(np.array2string(matrix, precision=4, suppress_small=True))

def calc_r_v(a,e,i,Omega,w,f,grav_parameter=398600):
    # Constantes
    μ = grav_parameter  # km³/s² (parâmetro gravitacional da Terra)
    
    # Dados iniciais (elementos orbitais)
    I = np.radians(i)  # inclinação (convertido para radianos)
    Omega = np.radians(Omega)  # longitude do nodo ascendente
    u = w + f
    u = np.radians(u)
    w = np.radians(w)  # argumento do pericentro
    f = np.radians(f)  # anomalia verdadeira
    
    print("===== Cálculo dos Vetores Posição e Velocidade - Exemplo 3.4 =====")
    
    # Passo 1: Cálculo do semi-latus rectum
    p = a * (1 - e**2)
    print_step(1, "Cálculo do semi-latus rectum (p = a(1-e²))", f"{p:.3f} km")
    
    # Passo 2: Cálculo da distância radial
    r = p / (1 + e * np.cos(f))
    print_step(2, "Cálculo da distância radial (r = p/(1+e·np.cos(f)))", f"{r:.4f} km")
    
    # Passo 3: Cálculo das componentes de velocidade
    sqrt_u_p = np.sqrt(μ/p)
    v_r = sqrt_u_p * e * np.sin(f)
    v_s = sqrt_u_p * (1 + e * np.cos(f))
    print_step(3, "Cálculo das componentes de velocidade:", 
              (f"Componente radial (v_r): {v_r:.4f} km/s", 
               f"Componente circunferencial (v_s): {v_s:.4f} km/s"))
    
    # Passo 4: Determinação da matriz de transformação    
    # Matriz de rotação R3(Ω)
    R3_Omega = np.array([
        [np.cos(Omega), np.sin(Omega), 0],
        [-np.sin(Omega), np.cos(Omega), 0],
        [0, 0, 1]
    ])
    
    # Matriz de rotação R1(I)
    R1_I = np.array([
        [1, 0, 0],
        [0, np.cos(I), np.sin(I)],
        [0, -np.sin(I), np.cos(I)]
    ])
    
    # Matriz de rotação R3(u)
    R3_u = np.array([
        [np.cos(u), np.sin(u), 0],
        [-np.sin(u), np.cos(u), 0],
        [0, 0, 1]
    ])
    
    # Matriz de transformação completa (Gauss para Geocêntrico)
    T_Gauss_Geo = R3_u @ R1_I @ R3_Omega
    
    print_step(4, "Matrizes de transformação:", 
              matrix=None, unit=None)
    print("\nMatriz R3(Ω) Rotação em torno do eixo z de um ângulo Ω -> Ox1y1z1:")
    print(np.array2string(R3_Omega, precision=4, suppress_small=True))
    print("\nMatriz R1(I) Rotação em torno do eixo x1 de um ângulo I -> Ox2y2z2:")
    print(np.array2string(R1_I, precision=4, suppress_small=True))
    print("\nMatriz R3(u) Rotação em torno do eixo z2 de um ângulo u -> Ox'y'z':")
    print(np.array2string(R3_u, precision=4, suppress_small=True))
    print("\nMatriz de transformação completa T_Gauss_Geo = R3(u) @ R1(I) @ R3(Ω):")
    print(np.array2string(T_Gauss_Geo, precision=4, suppress_small=True))
    
    # Passo 5: Determinação dos vetores no sistema geocêntrico-equatorial
    # Vetor posição no sistema de Gauss (perifocal)
    r_Gauss = np.array([r, 0, 0])
    
    # Vetor velocidade no sistema de Gauss (perifocal)
    v_Gauss = np.array([v_r, v_s, 0])
    
    # Transformação para sistema geocêntrico-equatorial
    T_Gauss_Geo_T = T_Gauss_Geo.T
    r_Geo = T_Gauss_Geo_T @ r_Gauss
    v_Geo = T_Gauss_Geo_T @ v_Gauss

    print("\nMatriz Transposta:")
    print(np.array2string(T_Gauss_Geo_T, precision=4, suppress_small=True))
    
    print_step(5, "Vetores no sistema geocêntrico-equatorial:", 
              (f"Posição: {np.array2string(r_Geo, precision=2)} km",
               f"Velocidade: {np.array2string(v_Geo, precision=4)} km/s"))
    
    print("\n===== Resultados Finais =====")
    print("Vetor posição no sistema geocêntrico-equatorial (km):")
    print(np.array2string(r_Geo, precision=2))
    print("\nVetor velocidade no sistema geocêntrico-equatorial (km/s):")
    print(np.array2string(v_Geo, precision=4))

if __name__ == "__main__":
    from plots.plot_orbit import plot_orbit 
    # Dados iniciais (elementos orbitais)
    # Ex: 3
    # a = 10000  # km (semi-eixo maior)
    # e = 0.05    # excentricidade
    # i = 30  # inclinação (graus)
    # Omega = 90  # longitude do nodo ascendente (graus)
    # w = 60  # argumento do pericentro (graus)
    # f = 45  # anomalia verdadeira (graus)
    # μ = 398600  # km³/s² (parâmetro gravitacional da Terra)

    # Ex: 4
    # a = 1.26
    # e = 0.417
    # i = 53.96
    # Omega = 345.96
    # w = 118.02
    # f = 45    
    # μ = 1  # km³/s² (parâmetro gravitacional da Terra)

    # Ex: 5
    a = 15359.76  # km (semi-eixo maior)
    e = 0.759    # excentricidade
    i = 90  # inclinação (graus)
    Omega = 45  # longitude do nodo ascendente (graus)
    w = 107.92  # argumento do pericentro (graus)
    f = 60  # anomalia verdadeira (graus)
    μ = 398600  # km³/s² (parâmetro gravitacional da Terra)

    calc_r_v(a,e,i,Omega,w,f,grav_parameter=μ)
    plot_orbit(a,e,i,Omega,w,f)
    