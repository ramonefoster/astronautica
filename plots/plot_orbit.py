import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

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

    # XYZ reference coordinates
    x_axis = np.array([1,0,0])
    y_axis = np.array([0,1,0])
    z_axis = np.array([0,0,1])

    # Nodes line (Ω)
    nodal_axis = np.array([np.cos(Omegao), np.sin(Omegao), 0])
    plot_vector([0,0,0], nodal_axis, 'green', 'Ascending Node')

    # Pericenter direction (ω)
    pericentro = R @ x_axis
    plot_vector([0,0,0], pericentro, 'orange', 'Periapsis', 0.6*a)

    # Plot Orbit
    ax.plot(orbit_inertial[0], orbit_inertial[1], orbit_inertial[2], 
            label='Orbit', linewidth=1)

    # Small Body (this case, a satellite)
    ax.scatter(r_inertial[0], r_inertial[1], r_inertial[2], 
            color='red', s=50, label='Satellite')

    # Massive Body (this case, a planet)
    ax.scatter(0, 0, 0, 
            color='orange', s=100, label='Planet')
    
    # Apsis Line
    r_inertial = R @ r_peri
    pericenter = R @ np.array([a*(1-e), 0, 0])
    apocenter = R @ np.array([-a*(1+e), 0, 0])
    ax.plot([pericenter[0], apocenter[0]], 
            [pericenter[1], apocenter[1]], 
            [pericenter[2], apocenter[2]], 
            'm--', linewidth=2, label='Apsis')

    # 5. Nodes (intersection of orbital and equatorial planes)
    ax.plot([0, nodal_axis[0]*a*1.2],
        [0, nodal_axis[1]*a*1.2],
        [0, nodal_axis[2]*a*1.2],
        'g--', linewidth=2, label='Nodes')

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
    AngleAnnotation3D(ax, [0,0,0], x_axis, nodal_axis, radius=0.3*a, color='green', 
                 text=f'Ω = {np.degrees(Omegao):.1f}°')
    AngleAnnotation3D(ax, [0,0,0], nodal_axis, pericenter, radius=0.4*a, color='orange', 
                 text=f'ω = {np.degrees(omega):.1f}°')
    AngleAnnotation3D(ax, [0,0,0], pericenter, r_inertial, radius=0.2*a, color='red', 
                 text=f'f = {np.degrees(f):.1f}°')
    # AngleAnnotation3D(ax, [0,0,0], z_axis, h_vec, radius=0.2*a, color='cyan', 
    #              text=f'I = {np.degrees(i):.1f}°')
    
    ax.set_xlim([-1.5*a, 1.5*a])
    ax.set_ylim([-1.5*a, 1.5*a])
    ax.set_zlim([-1.5*a, 1.5*a])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Orbital elements')
    ax.legend()

    plt.tight_layout()
    plt.show()
