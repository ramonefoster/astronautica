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

def kepler_f_hyperbolic(F, N, e):
    return e * np.sinh(F) - F - N

def kepler_fp_hyperbolic(F, e):
    return e * np.cosh(F) - 1

# Newton–Raphson for hyperbolic orbits
def newton_raphson_hyperbolic(F0, N, e, tol=1e-15, max_iter=100):
    F = F0
    for i in range(max_iter):
        f  = kepler_f_hyperbolic(F, N, e)
        fp = kepler_fp_hyperbolic(F, e)
        dF = -f / fp
        Fn = F + dF

        if abs(Fn - F) < tol:
            break

        F = Fn
    return F

def newton_raphson_universal(guess, deltaT, a, r0, v0, grav_param, tol=1e-15, max_iter=100):
    X = guess

    def dt(X, z, order):
        if z > 0:
            C = (1 - np.cos(np.sqrt(z))) / z
            S = (np.sqrt(z) - np.sin(np.sqrt(z))) / np.sqrt(z**3)
        elif round(z, 10) == 0:
            C = 0.5
            S = 1/6
        else:
            C = (1 - np.cosh(np.sqrt(-z))) / z
            S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / np.sqrt(-z**3)
        
        if order == 0:
            return ((X**3 * S) + (np.dot(r0,v0)*X**2*C/np.sqrt(grav_param)) + np.linalg.norm(r0)*X*(1-z*S))/np.sqrt(grav_param)
        else:
            return ((X**2*C) + (np.dot(r0,v0)*X*(1-z*S)/np.sqrt(grav_param)) + (np.linalg.norm(r0)*(1-z*C)))/np.sqrt(grav_param)

    for i in range(max_iter):  
        z = X**2/a      
        Xnp = X + ((deltaT - dt(X, z, 0))/dt(X, z, 1))
    
        if abs(Xnp - X) < tol:
            break

        X = Xnp
        
    return X, z

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def plot_comparison(time_steps, r_exact, v_exact, r_series, v_series):
    """Gera gráficos comparativos para distância radial e velocidade."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Gráfico da Distância Radial
    ax1.plot(time_steps, r_exact, label='Solução Exata', color='blue')
    ax1.plot(time_steps, r_series, label='Solução com Séries f e g', linestyle='--', color='red')
    ax1.set_title('Comparação da Distância Radial', fontsize=16)
    ax1.set_xlabel('Tempo (s)', fontsize=12)
    ax1.set_ylabel('Distância Radial (km)', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Gráfico da Velocidade
    ax2.plot(time_steps, v_exact, label='Solução Exata', color='blue')
    ax2.plot(time_steps, v_series, label='Solução com Séries f e g', linestyle='--', color='red')
    ax2.set_title('Comparação da Velocidade', fontsize=16)
    ax2.set_xlabel('Tempo (s)', fontsize=12)
    ax2.set_ylabel('Velocidade (km/s)', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
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
    ax.plot(orbit_inertial[0], orbit_inertial[1], orbit_inertial[2], 
            label='Orbit', linewidth=.5)
    
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
    earth_radius = 6378
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
