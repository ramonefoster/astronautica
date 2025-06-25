import numpy as np

MU = 398600
r0 = 6600
rf = 90000

sigma_f = rf/r0

if sigma_f < 11.94:
    print("Manobra de Hohmann")
    # primeiro incremento
    dv0 = np.sqrt(MU/r0)*(-1 + np.sqrt(2*sigma_f/(1 + sigma_f)))
    print(f"Primeiro incremento: {dv0:.4f} km/s")

    #segundo incremento
    dvf = np.sqrt(MU/(r0*sigma_f))*(1-np.sqrt(2/(1+sigma_f)))
    print(f"Segundo Incremento: {dvf:.4f} km/s")

    Cf = dv0 + dvf
    print(f"Delta V total: {Cf:.4f} km/s")

    #semi-eixo maior:
    a = (r0+rf)/2
    print(f"Semi-eixo maior a: {1:.4f} km")

    #excentricidade
    e = (rf-r0)/(rf+r0)
    print(f"Excentricidade: {e:.4f}")

    #duracao da manobra
    delta_t = (np.pi/2)*np.sqrt((r0+rf)**3/(2*MU))
    #tempo em hh:mm:ss
    delta_t_h = int(delta_t/3600)
    delta_t_m = int((delta_t % 3600)/60)
    delta_t_s = (delta_t % 60)
    print(f"Tempo de voo: {delta_t_h}hs {delta_t_m} min {delta_t_s:.2f}s")

    text = f"""
        A manobra de menor consumo de combustível - mínimo delta-v - que transfere
        o satélite de uma órbita circular de {r0} para uma órbita circular a {rf}
        km é realizada mediante a aplicação de dois impulsos: o primeiro impulso,
        aplicado tangencialmente à órbita circular inicial, fornece um incremento de velocidade
        de {dv0} km/s e coloca o satélite no pericentro de uma órbita elíptica de semi-eixo
        maior {a} km e excentricidade {e}. O segundo impulso, aplicado
        tangencialmente à órbita de transferência em seu apocentro, fornece um incremento de
        velocidade de {dvf} km/s e coloca o satélite na órbita circular final. O tempo de vôo
        para a realização da manobra é de {delta_t_h}hs {delta_t_m} min {delta_t_s:.2f}s.
    """
    print(text)
    

elif sigma_f >= 11.94:
    print("Bi-Parabolica")
    dv0 = np.sqrt(MU/r0)*(-1 + np.sqrt(2))
    print(f"Primeiro incremento: {dv0:.4f} km/s")

    #segundo incremento
    dvf = np.sqrt(MU/(rf))*(np.sqrt(2) -1)
    print(f"Segundo Incremento: {dvf:.4f} km/s")

    Cf = dv0 + dvf
    print(f"Delta V total: {Cf:.4f} km/s")

    #semi-eixo maior:
    a = (r0+rf)/2
    print(f"Semi-eixo maior a: {1:.4f} km")

    e = (rf-r0)/(rf+r0)
    #semi-latus rectum
    p = rf*2
    print(f"Semi-latus rectum p: {p:.4f} km")

    text = f"""
A manobra de menor consumo de combustível - mínimo delta-v - que transfere
o satélite de uma órbita circular de {r0} km de raio para uma órbita circular de {rf}
km de raio é realizada mediante a aplicação de dois impulsos finitos e de um impulso
intermediário infinitesimal: o primeiro impulso, aplicado tangencialmente à órbita
circular inicial, fornece um incremento de velocidade de {dv0} km/s e coloca o satélite
no pericentro de uma órbita parabólica de semi-latus rectum {r0*2} km. O impulso
intermediário, aplicado a uma distância infinita, coloca o satélite em uma segunda
órbita parabólica de semi-latus rectum {rf*2} km. O segundo impulso finito, aplicado
tangencialmente à segunda órbita parabólica em seu pericentro, fornece um incremento
de velocidade de {dvf} km/s, desacelerando o satélite e colocando-o na órbita circular
final. 
"""
    
    print(text)
