import numpy as np

def calc_flight_time(ha = None, hp=None, f=None, P=None):
    f= np.radians(f) # rad
    Rt = 6378 # km

    rp = Rt + hp # km
    grav_param = 398600 # km³/s² (parâmetro gravitacional da Terra)

    if P:
        P = P * 60 # T em minutos    
        # Semi eixo maior
        a = ((grav_param * P**2) / (4 * np.pi**2))**(1/3) # km 
    else:
        # Semi eixo maior
        a = Rt + (ha + hp)/2
    
    # Excentricidade
    e = 1 - (rp/a) # excentricidade

    # Anomalia Excentrica
    E = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(f/2)) # rad

    # Anomalia Media
    M = E - e * np.sin(E) # rad

    if P:
        # Tempo de voo, t 
        t = P*M/(2*np.pi) # s
        n = None
    else:
        n = np.sqrt(grav_param / (a**3))
        # Tempo de voo, t
        t = M/n

    t = t / 60 # min

    dec = t - (int(t)) # parte decimal

    print(f"Semi-eixo maior: {a:.4f} km")
    print(f"Excentricidade: {e:.5f}")
    print(f"Anomalia Excentrica: {E:.5f} rad, ou {np.degrees(E):.4f} graus")
    print(f"Anomalia Media: {M:.5f} rad, ou {np.degrees(M):.4f} graus")
    if n:
        print(f"Movimento médio n: {n}")
    print(f"Tempo de voo: {t} = {(int(t))} min {dec*60:.2f} s")
    return t

if __name__ == "__main__":
    # Exercicio 1
    input = {
        "ha": 0, # Altura em km
        "hp": 750, # Altura em km
        "f": 90, # Anomalia verdadeira em graus
        "P": 120 # Período em minutos
    }

    # # Exercicio 2
    input = {
        "ha": 900, # Altura em km
        "hp": 700, # Altura em km
        "f": 90, # Anomalia verdadeira em graus
        "P": None # Período em minutos
    }
    
    calc_flight_time(**input)