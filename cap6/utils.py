import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

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

import numpy as np

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
        
    # Passo 1: Determinação de r e v
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    # Passo 2: Determinação das Integrais Primeiras
    # Energia específica
    ε = 0.5 * v**2 - μ/r
    
    # Momentum angular específico
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # Vetor Laplace Runge-Lenz B
    B_vec = np.cross(v_vec, h_vec) - μ * r_vec / r
    B = np.linalg.norm(B_vec)
    
    # Passo 3: Determinação do vetor nodal N
    k = np.array([0, 0, 1])
    N_vec = np.cross(k, h_vec)
    N = np.linalg.norm(N_vec)
    
    # Passo 4: Determinação do parâmetro adimensional Q
    Q = (r * v**2) / μ
    
    # Passo 5: Determinação do semi-eixo maior
    a = r / (2 - Q)
    
    # Passo 6: Determinação da excentricidade
    e = B / μ
    
    # Passo 7: Determinação da inclinação do plano da órbita
    I = np.degrees(np.arccos(np.dot(k, h_vec)/h))
    
    # Passo 8: Determinação da longitude do nodo ascendente
    i = np.array([1, 0, 0])
    Omegao_cos = np.dot(i, N_vec)/N
    Omegao = np.degrees(np.arccos(Omegao_cos))
    
    # Determinar o quadrante (j·N < 0 -> 4º quadrante)
    j = np.array([0, 1, 0])
    if np.dot(j, N_vec) < 0:
        Omegao = 360 - Omegao
        
    # Passo 9: Determinação do argumento do pericentro
    w_cos = np.dot(B_vec, N_vec)/(B*N)
    w = np.degrees(np.arccos(w_cos))
    
    # Determinar o quadrante (k·B < 0 -> 4º quadrante)
    if np.dot(k, B_vec) < 0:
        w = 360 - w
    
    # Passo 10: Determinação da anomalia verdadeira
    f_cos = np.dot(B_vec, r_vec)/(B*r)
    f = np.degrees(np.arccos(f_cos))
    # Determinar o quadrante (r·v > 0 -> 2º quadrante)
    if np.dot(r_vec, v_vec) < 0:
        f = 360 - f

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

import pandas as pd

def p_NR(r1, r2, df, k, l, m, dt, tolerance, guess, maxIte, printMode=1):
    mu = 398600  # [km^3/s^2] standard gravitational parameter
    p = guess
    numberOfIterations = 0
    error = 100

    p_vec = []
    a_vec = []
    g_vec = []
    dX_vec = []
    dtn_vec = []
    error_vec = []

    while error > tolerance:
        denominator = ((2 * m - l ** 2) * p ** 2 + 2 * k * l * p - k ** 2)
        a = m * k * p / denominator
        g = r1 * r2 * np.sin(df) / np.sqrt(mu * p)

        if a > 0:
            dX = np.arctan2(-np.tan(df / 2) * (k - l * p) / (p * np.sqrt(p * a)), 1 - k / (p * a))
            if dX < 0:
                dX += 2 * np.pi
            dtn = g + np.sqrt(a ** 3 / mu) * (dX - np.sin(dX))
            ddt = (-g / (2 * p)
                   - (3 / 2) * a * (dtn - g) * ((k ** 2 + (2 * m - l ** 2) * p ** 2) / (m * k * p ** 2))
                   + np.sqrt(a ** 3 / mu) * 2 * k * np.sin(dX) / (p * (k - l * p)))
        else:
            dX = np.arccosh(1 - k / (p * a))
            dtn = g + np.sqrt((-a) ** 3 / mu) * (np.sinh(dX) - dX)
            ddt = (-g / (2 * p)
                   - (3 / 2) * a * (dtn - g) * ((k ** 2 + (2 * m - l ** 2) * p ** 2) / (m * k * p ** 2))
                   + np.sqrt((-a) ** 3 / mu) * 2 * k * np.sinh(dX) / (p * (k - l * p)))

        numberOfIterations += 1
        p_try = p + (dt - dtn) / ddt

        p_vec.append(p_try)
        a_vec.append(a)
        g_vec.append(g)
        dX_vec.append(dX)
        dtn_vec.append(dtn)
        error = abs(dt - dtn)
        error_vec.append(error)

        p = p_try

        if numberOfIterations == maxIte:
            print(f"Did not converge within {maxIte} iterations.")
            return None, None, numberOfIterations

    # Optional detailed printout
    T, Tpres = None, None
    if printMode == 1:
        T = pd.DataFrame({
            'Iteracao': np.arange(1, numberOfIterations + 1),
            'p [km]': np.round(p_vec, 4),
            'a [km]': np.round(a_vec, 4),
            'g [s]': np.round(g_vec, 4),
            'ΔE [rad]': np.round(dX_vec, 4),
            'Δt [s]': np.round(dtn_vec, 4),
            'Δt error [s]': np.round(error_vec, 4)
        })
        Tpres = pd.DataFrame({
            'Iteracao': np.arange(1, numberOfIterations + 1),
            'p [km]': p_vec,
            'a [km]': a_vec,
            'g [s]': g_vec,
            'ΔE [rad]': dX_vec,
            'Δt [s]': dtn_vec,
            'Δt error [s]': error_vec
        })
        print(T)

    return T, Tpres, numberOfIterations

class AngleAnnotation3D:
    def __init__(self, ax, origin, v1, v2, radius=0.5, color='k', text=None):
        self.ax = ax
        self.origin = np.array(origin)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.radius = radius
        self.color = color
        self.text = text
        self.arc_points = 30
        
        # Normalize vectors
        self.v1_u = self.v1 / np.linalg.norm(self.v1)
        self.v2_u = self.v2 / np.linalg.norm(self.v2)
        
        # Calculate angle between vectors
        cross = np.cross(self.v1_u, self.v2_u)
        dot = np.dot(self.v1_u, self.v2_u)
        angle = np.arctan2(np.linalg.norm(cross), dot)
        
        # Determine rotation axis and correct angle direction
        self.rot_axis = np.cross(self.v1_u, self.v2_u)
        rot_norm = np.linalg.norm(self.rot_axis)
        self.rot_axis = self.rot_axis / rot_norm if rot_norm > 1e-8 else np.array([0, 0, 1])
        
        # Check if we need to flip the angle direction
        # Angle from v1 to v2 in the conventional direction
        if self.rot_axis[2] < 0:
            angle = 2 * np.pi - angle
            self.rot_axis = -self.rot_axis
        
        self.angle = angle
        self.text = self.text or f"{np.degrees(angle):.2f}°"
        
        self.generate_arc()
        self.draw()
    
    def generate_arc(self):
        # Create arc in the plane containing v1 and v2
        angles = np.linspace(0, self.angle, self.arc_points)
        self.arc_pts = []
        
        # Find orthogonal basis
        u = self.v1_u
        w = np.cross(self.rot_axis, u)
        w = w / np.linalg.norm(w)
        
        for angle in angles:
            pt = self.radius * (np.cos(angle) * u + np.sin(angle) * w)
            self.arc_pts.append(pt)
        
        self.arc_pts = np.array(self.arc_pts)
    
    def draw(self):
        # Plot the arc
        x = self.origin[0] + self.arc_pts[:,0]
        y = self.origin[1] + self.arc_pts[:,1]
        z = self.origin[2] + self.arc_pts[:,2]
        self.ax.plot(x, y, z, color=self.color, linewidth=2)
        
        # Add text label at midpoint
        if self.text:
            mid_idx = len(self.arc_pts) // 2
            mid_pt = self.origin + self.arc_pts[mid_idx] * 1.1
            self.ax.text(mid_pt[0], mid_pt[1], mid_pt[2], 
                        self.text, color=self.color)

# Custom class for arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return min(zs[0], zs[1])
    
def plot_orbit(a, e, I, Omega, w, f, h=None):
    # Input data in radians
    i = np.radians(I)  # inclination
    Omegao = np.radians(Omega)  # Longitude of the ascending node
    omega = np.radians(w)  # Argument of periapsis 
    f = np.radians(f)  # true anomaly

    # mag r
    r = a * (1 - e**2) / (1 + e * np.cos(f))

    # vector
    r_peri = np.array([
        r * np.cos(f),
        r * np.sin(f),
        0
    ])

    # Rotation matrices
    def rotation_matrix_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    def rotation_matrix_x(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c]
        ])

    R_Ω = rotation_matrix_z(Omegao)
    R_i = rotation_matrix_x(i)
    R_ω = rotation_matrix_z(omega)
    R = R_Ω @ R_i @ R_ω

    # Inertial system
    r_inertial = R @ r_peri

    # Plot whole orbit
    theta = np.linspace(0, 2*np.pi, 1000)
    r_vals = a * (1 - e**2) / (1 + e * np.cos(theta))
    x_peri = r_vals * np.cos(theta)
    y_peri = r_vals * np.sin(theta)
    z_peri = np.zeros_like(theta)

    # Transform to inertial system
    orbit_points = np.vstack([x_peri, y_peri, z_peri])
    orbit_inertial = R @ orbit_points

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    def plot_vector(origin, direction, color, label, length=0.7*a):
        arrow = Arrow3D(
            [origin[0], origin[0]+direction[0]*length],
            [origin[1], origin[1]+direction[1]*length],
            [origin[2], origin[2]+direction[2]*length],
            mutation_scale=15, lw=2, arrowstyle="-|>", color=color)
        ax.add_artist(arrow)
        ax.text(origin[0]+direction[0]*length*1.1,
                origin[1]+direction[1]*length*1.1,
                origin[2]+direction[2]*length*1.1,
                label, color=color)

    # Plot Orbit
    # Find the index corresponding to the current true anomaly
    # ax.plot(orbit_inertial[0], orbit_inertial[1], orbit_inertial[2], 
    #         label='Orbit', linewidth=.5)
    earth_radius = 6378
    # Calcular distância ao centro da Terra para cada ponto
    distances = np.linalg.norm(orbit_inertial, axis=0)
    above_surface = distances > earth_radius

    # Identificar transições (de fora para dentro ou vice-versa)
    transitions = np.where(np.diff(above_surface.astype(int)) != 0)[0]

    # Plotar segmentos fora da Terra
    for i in range(len(orbit_inertial[0]) - 1):
        if above_surface[i] and above_surface[i+1]:
            ax.plot(orbit_inertial[0][i:i+2],
                    orbit_inertial[1][i:i+2],
                    orbit_inertial[2][i:i+2],
                    color='C0', linewidth=0.8)

    # Marcar pontos de intersecção com a superfície da Terra
    for idx in transitions:
        # Interpolação linear para estimar ponto de intersecção
        p1 = orbit_inertial[:, idx]
        p2 = orbit_inertial[:, idx + 1]
        d1 = np.linalg.norm(p1)
        d2 = np.linalg.norm(p2)
        alpha = (earth_radius - d1) / (d2 - d1) if d2 != d1 else 0.5
        intersection = p1 + alpha * (p2 - p1)

        ax.scatter(*intersection, color='orange', s=20, label='Intersection' if idx == transitions[0] else "")

    f_mod = f % (2 * np.pi)
    idx = np.searchsorted(theta, f_mod)
    half_idx = idx // 2 if idx > 0 else len(theta) // 4

    # Indices for 50% of the orbit ending at the current position
    start_idx = idx - half_idx
    if start_idx < 0:
        start_idx = 0
    end_idx = idx + 1

    # Extract the segment
    seg_x = orbit_inertial[0][start_idx:end_idx]
    seg_y = orbit_inertial[1][start_idx:end_idx]
    seg_z = orbit_inertial[2][start_idx:end_idx]

    # Compute distances to the small body for linewidth modulation
    seg_points = np.vstack([seg_x, seg_y, seg_z]).T
    small_body_pos = r_inertial
    dists = np.linalg.norm(seg_points - small_body_pos, axis=1)
    dists = np.max(dists) - dists  # closer = higher value

    # Normalize linewidths between 1 and 4
    lw_min, lw_max = 1, 3.5
    if len(dists) > 1:
        lws = lw_min + (lw_max - lw_min) * (dists - np.min(dists)) / (np.ptp(dists) if np.ptp(dists) > 0 else 1)
        # Plot segment with varying linewidth
        for i in range(len(seg_x)-1):
            ax.plot(seg_x[i:i+2], seg_y[i:i+2], seg_z[i:i+2],
                    color='C9', linewidth=lws[i], label='Orbit' if i == 0 else "", alpha=0.2)
    else:
        ax.plot(seg_x, seg_y, seg_z, color='C9', linewidth=lw_min, label='Orbit', alpha=0.2)

    # Small Body (this case, a satellite)
    ax.scatter(r_inertial[0], r_inertial[1], r_inertial[2], 
            color='red', s=25, label='Satellite')
    
    # Plot Earth at the origin
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    
    x_earth = earth_radius * np.cos(u) * np.sin(v)
    y_earth = earth_radius * np.sin(u) * np.sin(v)
    z_earth = earth_radius * np.cos(v)
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.6, linewidth=0)
    
    # Apsis Line
    r_inertial = R @ r_peri
    pericenter = R @ np.array([a*(1-e), 0, 0])
    apocenter = R @ np.array([-a*(1+e), 0, 0])
    ax.plot([pericenter[0], apocenter[0]], 
            [pericenter[1], apocenter[1]], 
            [pericenter[2], apocenter[2]], 
            'm--', linewidth=2, label='Apsis')

    # Draw Equatorial Plane
    xx, yy = np.meshgrid(np.linspace(-2*a, 2*a, 2), np.linspace(-2*a, 2*a, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='grey', alpha=0.3, label='Equatorial Plane')

    # Axis arrows
    ax.quiver(0, 0, 0, .3*a, 0, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, .3*a, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, .3*a, color='k', arrow_length_ratio=0.1)
    plot_vector([0,0,0], r_inertial, 'red', 'R', 1)

    x_axis = np.array([1,0,0])
    
    ax.set_xlim([-1.5*a, 1.5*a])
    ax.set_ylim([-1.5*a, 1.5*a])
    ax.set_zlim([-1.5*a, 1.5*a])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Orbital elements')
    ax.legend()

    ax.view_init(elev=5, azim=310)

    plt.tight_layout()
    plt.show()
    return fig