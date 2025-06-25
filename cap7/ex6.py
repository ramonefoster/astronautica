import numpy as np

# Constantes e condições iniciais
MU = 398600  # km^3/s^2
r0 = 7160  # km
rf = 3 * r0
THETA_DEG = 30
THETA = np.radians(THETA_DEG)

a = (r0 + rf) / 2
e = (rf - r0) / (rf + r0)

v1_abs = np.sqrt((MU * (1 + e)) / r0)
v2_abs = np.sqrt((MU * (1 - e)) / rf)
v0 = np.sqrt(MU / r0)
vf = np.sqrt(MU / rf)

print(f"Semi-eixo maior: {a:.4f} km")
print(f"Excentricidade: {e:.4f}")
print(f"Velocidades nos apsis da órbita de transferência: {v1_abs:.4f}, {v2_abs:.4f} km/s")
print(f"Velocidades nas órbitas circular inicial e final: {v0:.4f}, {vf:.4f} km/s")

# Função f(alpha1)
def f(alpha1):
    term1 = v1_abs * v0 * np.sin(alpha1)
    inner1 = np.sqrt(v2_abs**2 + vf**2 - 2*v2_abs*vf*np.cos(THETA - alpha1))
    term2 = v2_abs * vf * np.sin(THETA - alpha1)
    inner2 = np.sqrt(v1_abs**2 + v0**2 - 2*v1_abs*v0*np.cos(alpha1))
    return term1 * inner1 - term2 * inner2

# Derivada df/dalpha1 numericamente
def df(alpha1, h=1e-6):
    return (f(alpha1 + h) - f(alpha1 - h)) / (2 * h)

# Método de Newton-Raphson
def newton_raphson(alpha0, tol=1e-8, max_iter=100):
    alpha = alpha0
    for i in range(max_iter):
        f_val = f(alpha)
        df_val = df(alpha)
        if abs(df_val) < 1e-12:
            print("Derivada próxima de zero, parada.")
            break
        alpha_new = alpha - f_val / df_val
        if abs(alpha_new - alpha) < tol:
            print(f"Convergiu em {i+1} iterações.")
            return alpha_new
        alpha = alpha_new
    raise RuntimeError("Não convergiu.")

# Chamada do método
alpha0_guess = np.radians(10)  # chute inicial
alpha1_solution = newton_raphson(alpha0_guess)
print(f"Ângulo ótimo alpha1: {np.degrees(alpha1_solution):.4f} radians")

delta_v0 = np.sqrt(v1_abs**2 + v0**2 -2*v1_abs*v0*np.cos(alpha1_solution))
print(f"deltaV0: {delta_v0:.4f} km/s")

delta_vf = np.sqrt(v2_abs**2 + vf**2 -2*v2_abs*vf*np.cos(THETA-alpha1_solution))
print(f"deltaVf: {delta_vf:.4f} km/s")

print(f"Cf = {delta_v0+delta_vf}")