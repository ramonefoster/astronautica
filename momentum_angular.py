import numpy as np
import matplotlib.pyplot as plt

# Definindo os vetores
r = np.array([-1, 1, 0])
v = np.array([2, -4, -4])
h = np.cross(r, v)  # produto vetorial

# Calculando e imprimindo h como uma equação
r_equation = f"{r[0]}i {'+' if r[1] >= 0 else ''}{r[1]}j {'+' if r[2] >= 0 else ''}{r[2]}k"
v_equation = f"{v[0]}i {'+' if v[1] >= 0 else ''}{v[1]}j {'+' if v[2] >= 0 else ''}{v[2]}k"
h_equation = f"{h[0]}i {'+' if h[1] >= 0 else ''}{h[1]}j {'+' if h[2] >= 0 else ''}{h[2]}k"

print(f"Vector radial position is:\n r = {r_equation}")
print(f"Vector velocity is:\n v = {v_equation}")
print(f"rxv = Angular Momentum is:\n h = {h_equation}")

# Criando o gráfico
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Origem
origin = np.array([0, 0, 0])

# Plotando os vetores
ax.quiver(*origin, *r, color='blue', label='r (posição)', linewidth=2)
ax.quiver(*origin, *v, color='green', label='v (velocidade)', linewidth=2)
ax.quiver(*origin, *h, color='red', label='h = r × v (momento angular)', linewidth=2)

# Ajustando o ângulo de visão
ax.view_init(elev=20, azim=30)
# Ajustando limites e rótulos
max_range = 6
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualização do Produto Vetorial: r × v')
ax.legend()

plt.tight_layout()
plt.show()
