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