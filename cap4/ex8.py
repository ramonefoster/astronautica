import numpy as np
import matplotlib.pyplot as plt
import time

# Função de Kepler e iteração de Newton–Raphson (1ª ordem)
def kepler_f(E, M, e):
    return M - E + e * np.sin(E)

def kepler_fp(E, e):
    return -1 + e * np.cos(E)

def kepler_fs(E, e):
    return -e * np.sin(E)

def newton_raphson_sec(E0, M, e, tol=1e-15, max_iter=100):
    E = E0
    for i in range(max_iter):
        f  = kepler_f(E, M, e)
        fp = kepler_fp(E, e)
        fs = kepler_fs(E, e)
        dE = -f / fp
        dE2 = -f/(fp+0.5*dE*fs)
        En = E + dE2

        if abs(En - E) < tol:
            break

        E = En
    return i + 1

# Exemplo livro
e_vals = np.linspace(0, 0.99, 300)
M_vals = np.linspace(0, np.pi, 300)

# Parâmetros de grade
# e_vals = [0.05, 0.25, 0.75, 0.99]
# M_vals = np.linspace(0, 2 * np.pi, 50)

# Cria malha
M_mesh, e_mesh = np.meshgrid(M_vals, e_vals)
iter_mesh = np.zeros_like(M_mesh)

# Computa número de iterações para cada ponto da malha
t0 = time.time()
for i, e in enumerate(e_vals):
    for j, M in enumerate(M_vals):
        # Eo = M
        # iter_mesh[i, j] = newton_raphson_sec(M, M, e)

        # Eo = M + e
        iter_mesh[i, j] = newton_raphson_sec(M + e, M, e)

        # Eo = M + e * np.sin(M)
        # iter_mesh[i, j] = newton_raphson_sec(M + e * np.sin(M), M, e)
print(f"Tempo de execução: {time.time() - t0:.4f} segundos")
# Plots combinados
fig = plt.figure(figsize=(14, 6))

# 3D surface plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surface = ax1.plot_surface(M_mesh, e_mesh, iter_mesh, cmap='plasma', vmin=iter_mesh.min(), vmax=iter_mesh.max())
ax1.set_xlabel('Anomalia Média M (rad)')
ax1.set_ylabel('Excentricidade e')
ax1.set_zlabel('Número de Iterações')
ax1.set_title('Convergência NR')
fig.colorbar(surface, ax=ax1, pad=0.1, label='Iterações')

# 2D colormap plot
ax2 = fig.add_subplot(1, 2, 2)
c = ax2.pcolormesh(M_mesh, e_mesh, iter_mesh, shading='auto', cmap='plasma')
ax2.set_xlabel('Anomalia Média M (rad)')
ax2.set_ylabel('Excentricidade e')
ax2.set_title('Convergência NR')
fig.colorbar(c, ax=ax2, pad=0.1, label='Iterações')

plt.tight_layout()
plt.show()

# Encontrando o máximo
max_idx = np.unravel_index(iter_mesh.argmax(), iter_mesh.shape)

M_max = M_mesh[max_idx]
e_max = e_mesh[max_idx]
max_val = iter_mesh[max_idx]

print(f"Máximo número de iterações: {max_val} ocorre em M = {np.degrees(M_max):.2f}, e = {e_max:.2f}")
print(f"Media de iterações: {iter_mesh.mean():.2f}")