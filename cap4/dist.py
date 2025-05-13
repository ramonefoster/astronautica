import numpy as np

def calc_dist(**kargs):

    hc = kargs.get("hc", 0) # Altura em km
    t = kargs.get("t", 0) # Tempo em minutos
    
    Rt = 6378 # km
    rc = Rt + hc # km
    grav_param = 398600

    # raio da orbita
    print(f"Raio da orbita: {rc} km")

    # semi-lactus rectum
    p = 2*rc
    print(f"Semi-lactus rectum: {p} km")
    # Anomalia media parabolica
    Mp = np.sqrt(grav_param/p**3) * t * 3600 # rad
    print(f"Anomalia Parabolica Media Mp: {Mp} rad, ou {np.degrees(Mp)} graus")

    # Equacao de Barker
    D = np.sqrt(p)*( (3*Mp + np.sqrt(1 + 9*Mp**2))**(1/3) - (3*Mp + np.sqrt(1 + 9*Mp**2))**(-1/3))
    print(f"Anomalia Parabolica D: {D}")

    # Distancia do centro da Terra
    r = 0.5*(p + D**2)
    print(f"Distancia do centro da Terra: {r} km")

    vc = np.sqrt(grav_param/rc) # km/s
    print(f"Velocidade orbital circular: {vc} km/s")

    v_escape = np.sqrt(2*grav_param/rc) # km/s
    print(f"Velocidade de escape: {v_escape} km/s")

    # delta v
    delta_v = v_escape - vc # km/s
    print(f"Incremento, Delta v: {delta_v} km/s")


if __name__ == "__main__":
    # Exercicio 3
    input = {
        "hc": 350, # Altura em km
        "t": 5 # Tempo em horas
    }
    calc_dist(**input)

    # Exercicio 2
