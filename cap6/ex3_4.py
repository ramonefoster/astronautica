import numpy as np
import matplotlib.pyplot as plt

def min_energy_transfer(r1, r2, theta):
    k = r1 * r2 * (1 - np.cos(theta))
    l = r1 + r2
    m = r1 * r2 * (1 + np.cos(theta))
    a = m/(2*l - 2*np.sqrt(l**2 - 2*m))

    return a

def flight_time(a, mu=1.0):
    """Calculate flight time for half-ellipse transfer"""
    dt = np.pi/ np.sqrt(mu/a**3)
    return dt

def calc_amin(r_earth, r_mars):    
    df_values= np.linspace(0, 2*np.pi, 360)
    a_values = [min_energy_transfer(r_earth, r_mars, df) for df in df_values]

    print(f"amin energia: {max(a_values)}")
    print(f"test: {(r_earth+r_mars)/2}")

    t_flight = flight_time(max(a_values))
    print(t_flight)
    plt.figure(figsize=(8, 5))
    plt.plot(np.degrees(df_values), a_values)
    plt.xlabel('theta (graus)')
    plt.ylabel('Semi-major axis a (km)')
    plt.title('Semi-major axis a vs Theta')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    r_earth = 1 # u.d.
    r_mars = 1.524 # u.d.

    calc_amin(r_earth, r_mars)

