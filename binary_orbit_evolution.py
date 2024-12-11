import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import astropy.units as u
import astropy.constants as const
from matplotlib.animation import FuncAnimation
import argparse

# Use a scientific plotting style
plt.style.use(['science'])

def v_orb(primary_mass, secondary_mass, semi_major_axis):
    """
    Computes the orbital velocity.

    Parameters:
        primary_mass (astropy mass): Mass of the primary star .
        secondary_mass (astropy mass): Mass of the secondary star.
        semi_major_axis (astropy distance): Orbital semi-major axis.

    Returns:
        astropy speed: Orbital velocity (km/s).
    """
    return (np.sqrt(G * (primary_mass + secondary_mass) / semi_major_axis)).to('km/s')

def P_orb(primary_mass, secondary_mass, semi_major_axis):
    """
    Computes the orbital period.

    Parameters:
        primary_mass (astropy mass): Mass of the primary star .
        secondary_mass (astropy mass): Mass of the secondary star.
        semi_major_axis (astropy distance): Orbital semi-major axis.

    Returns:
        astropy time: Orbital period (days).
    """
    return (2 * pi * np.sqrt(semi_major_axis**3 / (G * (primary_mass + secondary_mass)))).to('day')

def pre_SN_a(M1, M2, P):
    """
    Computes the orbital semi-major axis before the supernova explosion.

    Parameters:
        M1 (astropy mass): Mass of the primary star .
        M2 (astropy mass): Mass of the secondary star.
        P (astropy time): Orbital period.

    Returns:
        astropy distance: Orbital semi-major axis (R_sun).
    """
    return (((P / (2 * pi))**2 * G * (M1 + M2))**(1/3)).to('Rsun')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Simulate a supernova event in a binary system.")
parser.add_argument("--m1", type=float, default=12, help="Mass of the primary star (in solar masses).")
parser.add_argument("--m2", type=float, default=6, help="Mass of the secondary star (in solar masses).")
parser.add_argument("--final_m1", type=float, default=4, help="Remnant mass of primary star after explosion (in solar masses).")
parser.add_argument("--P", type=float, default=5, help="Orbital period (in days).")
parser.add_argument("--kick", type=float, nargs=3, default=[0, 0, 0], help="Kick velocity vector (in km/s).")
parser.add_argument("--P_explode", type=float, default=3, help="Time of explosion (in orbital periods).")
parser.add_argument("--P_end", type=float, default=20, help="End of simulation (in orbital periods).")
parser.add_argument("--N", type=int, default=2500, help="Number of points in simulation")
parser.add_argument("--output", type=str, default="SN_explosion_3D_no_kick", help="Output file name for the animation.")
args = parser.parse_args()

# Constants
pi = np.pi
msun = u.M_sun
rsun = u.R_sun
G = const.G
day = u.day
meter = u.meter
kms = (u.km) / (u.s)

# Variables from command-line arguments
m1 = args.m1 * msun
m2 = args.m2 * msun
m1_post_SN = args.final_m1 * msun
P = args.P * day
P_explode = args.P_explode
P_end = args.P_end
N = args.N
kick = (np.array(args.kick) * kms).si.value
output_file = args.output

# Initial conditions
# Compute the pre-supernova semi-major axis
a = pre_SN_a(m1, m2, P)
# Compute the orbital velocity
vorb = v_orb(m1, m2, a)

# Initial positions and velocities of the binary components
X1_init = ((m2 / (m1 + m2)) * a).si.value
Y1_init = (0 * rsun).si.value
Z1_init = (0 * rsun).si.value

X2_init = (-(m1 / (m1 + m2)) * a).si.value
Y2_init = (0 * rsun).si.value
Z2_init = (0 * rsun).si.value

VX1_init = (0 * kms).si.value
VY1_init = ((m2 / (m1 + m2)) * vorb).si.value
VZ1_init = (0 * kms).si.value

VX2_init = (0 * kms).si.value
VY2_init = (-(m1 / (m1 + m2)) * vorb).si.value
VZ2_init = (0 * kms).si.value

# Combine initial conditions into a single array
Y_init = np.array([X1_init, Y1_init, Z1_init, X2_init, Y2_init, Z2_init, VX1_init, VY1_init, VZ1_init, VX2_init, VY2_init, VZ2_init])

def func(Y, t, G, m1, m2):
    """
    Defines the differential equations for the binary system.

    Parameters:
        Y (array): State vector containing positions and velocities of the binary components.
        t (float): Time variable.
        G (float): Gravitational constant.
        m1 (float): Mass of the primary star (kg).
        m2 (float): Mass of the secondary star (kg).

    Returns:
        array: Derivatives of the state vector.
    """
    # Unpack the state vector
    X1, Y1, Z1, X2, Y2, Z2, VX1, VY1, VZ1, VX2, VY2, VZ2 = Y
    
    # Compute the distance between the two stars
    r = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)

    # Compute derivatives of positions
    dX1dt = VX1
    dY1dt = VY1
    dZ1dt = VZ1
    dX2dt = VX2
    dY2dt = VY2
    dZ2dt = VZ2

    # Compute derivatives of velocities using gravitational acceleration
    dVX1dt = G * m2 * (X2 - X1) / (r**3)
    dVY1dt = G * m2 * (Y2 - Y1) / (r**3)
    dVZ1dt = G * m2 * (Z2 - Z1) / (r**3)
    dVX2dt = G * m1 * (X1 - X2) / (r**3)
    dVY2dt = G * m1 * (Y1 - Y2) / (r**3)
    dVZ2dt = G * m1 * (Z1 - Z2) / (r**3)

    # Combine derivatives into a single array
    dYdt = [dX1dt, dY1dt, dZ1dt, dX2dt, dY2dt, dZ2dt,
            dVX1dt, dVY1dt, dVZ1dt, dVX2dt, dVY2dt, dVZ2dt]
    
    return dYdt

# Time array for the simulation
t = np.linspace(0 * day, P_end * P, N).si.value

# Time of the supernova explosion
t_explode = P_explode * P.si.value

# Simulate the orbit before the explosion
t_pre_SN = t[t <= t_explode]
t_end = t_pre_SN[-1]
sol_pre_SN = sp.integrate.odeint(func, Y_init, t_pre_SN, args=(G.si.value, m1.si.value, m2.si.value))

# Simulate the orbit after the explosion
# Add the kick velocity to the primary star
sol_pre_SN[-1, 6:9] += kick
Y_init_post_SN = sol_pre_SN[-1]
t_post_SN = t[t > t_explode]
t_post_SN = np.insert(t_post_SN, 0, t_end)
sol_post_SN = sp.integrate.odeint(func, Y_init_post_SN, t_post_SN, args=(G.si.value, m1_post_SN.si.value, m2.si.value))

#stack the pre and post SN solutions
sol = np.vstack((sol_pre_SN[:-1],sol_post_SN))
m1_array = np.ones(N) * m1
m1_array[len(sol_pre_SN):] = m1_post_SN

#primary star
X1 = (sol[:,0] * meter).to('Rsun')
Y1 = (sol[:,1] * meter).to('Rsun')
Z1 = (sol[:,2] * meter).to('Rsun')

#secondary star
X2 = (sol[:,3] * meter).to('Rsun')
Y2 = (sol[:,4] * meter).to('Rsun')
Z2 = (sol[:,5] * meter).to('Rsun')

#track the center of mass motion
X_com = (m1_array * X1 + m2 * X2) / (m1_array + m2)
Y_com = (m1_array * Y1 + m2 * Y2) / (m1_array + m2)
Z_com = (m1_array * Z1 + m2 * Z2) / (m1_array + m2)

VX_com = (m1_array * sol[:,6] + m2 * sol[:,9]) / (m1_array + m2)
VY_com = (m1_array * sol[:,7] + m2 * sol[:,10]) / (m1_array + m2)
VZ_com = (m1_array * sol[:,8] + m2 * sol[:,11]) / (m1_array + m2)

V_com = np.sqrt(VX_com**2 + VY_com**2 + VZ_com**2)

print("Post-SN systemic velocity:", (V_com[-1] * u.m/u.s).to('km/s'))
print("Post-SN eccentricity:", (V_com[-1] * u.m/u.s).to('km/s') / (vorb * m2 / (m1+m2)))

# Specific frame index where the transition happens
transition_frame = len(t_pre_SN)  # Replace with your value
scale = 0.5

# Set up the figure and 3D axes for animation in the inertial frame
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Initialize lines and markers
ln1_line, = ax.plot([], [], [], color='xkcd:blue', label='Progenitor')  # Line for Star 1
ln2_line, = ax.plot([], [], [], color='xkcd:red', label='Companion')    # Line for Star 2
ln3_line, = ax.plot([], [], [], color='xkcd:orange', label='Center of Mass')    # Line for Center of Mass

ln1_marker, = ax.plot([], [], [], color='xkcd:blue', marker='*', markersize=(m1.value / scale)**(2/3))  # Marker for Star 1
ln2_marker, = ax.plot([], [], [], color='xkcd:red', marker='*', markersize=(m2.value / scale)**(2/3))  # Marker for Star 2
ln3_marker, = ax.plot([], [], [], color='xkcd:orange', marker='o', markersize=((m1+m2).value / scale)**(2/3))  # Marker for Center of Mass

# Initialization function
def init():
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$X(R_{\odot})$')
    ax.set_ylabel(r'$Y(R_{\odot})$')
    ax.set_zlabel(r'$Z(R_{\odot})$')
    return [ln1_line, ln2_line, ln3_line, ln1_marker, ln2_marker, ln3_marker]

# Update function
def update(frame):
    # Update line trajectories
    ln1_line.set_data(X1[:frame], Y1[:frame])
    ln1_line.set_3d_properties(Z1[:frame])
    
    ln2_line.set_data(X2[:frame], Y2[:frame])
    ln2_line.set_3d_properties(Z2[:frame])

    ln3_line.set_data(X_com[:frame], Y_com[:frame])
    ln3_line.set_3d_properties(Z_com[:frame])

    # Update marker for the last point
    ln1_marker.set_data([X1[:frame][-1].value], [Y1[:frame][-1].value])
    ln1_marker.set_3d_properties([Z1[:frame][-1].value])
    
    ln2_marker.set_data([X2[:frame][-1].value], [Y2[:frame][-1].value])
    ln2_marker.set_3d_properties([Z2[:frame][-1].value])

    ln3_marker.set_data([X_com[:frame][-1].value], [Y_com[:frame][-1].value])
    ln3_marker.set_3d_properties([Z_com[:frame][-1].value])

    # Change marker style and color for Star 1 after the transition frame
    if frame < transition_frame:
        ln1_marker.set_marker('*')
        ln1_marker.set_color('xkcd:blue')
    else:
        if m1 != m1_post_SN:
            ln1_marker.set_marker('o')  # Black circle
            ln1_marker.set_color('xkcd:black')
            ln1_marker.set_markersize((m1_post_SN.value / scale)**(2/3))

    # Dynamically adjust axis limits
    xmin = min(X1[:frame].min(), X2[:frame].min()) - 1*rsun
    xmax = max(X1[:frame].max(), X2[:frame].max()) + 1*rsun
    ymin = min(Y1[:frame].min(), Y2[:frame].min()) - 1*rsun
    ymax = max(Y1[:frame].max(), Y2[:frame].max()) + 1*rsun
    zmin = min(Z1[:frame].min(), Z2[:frame].min()) - 1*rsun
    zmax = max(Z1[:frame].max(), Z2[:frame].max()) + 1*rsun

    ax.set_xlim(xmin.value, xmax.value)
    ax.set_ylim(ymin.value, ymax.value)
    ax.set_zlim(zmin.value, zmax.value)

    return [ln1_line, ln2_line, ln3_line, ln1_marker, ln2_marker, ln3_marker]

# Create the animation for orbit in CoM frame
ani = FuncAnimation(fig, update, frames=np.arange(1, N, 10),  # Start from 1
                    init_func=init, blit=False, interval=5)

# Save the animation as a GIF
ani.save(output_file + '.gif', writer='pillow', dpi=200)

# Set up the figure and 3D axes
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Initialize lines and markers
ln1_line, = ax.plot([], [], [], color='xkcd:blue', label='Progenitor')  # Line for Star 1
ln2_line, = ax.plot([], [], [], color='xkcd:red', label='Companion')    # Line for Star 2

ln1_marker, = ax.plot([], [], [], color='xkcd:blue', marker='*', markersize=(m1.value / scale)**(2/3))  # Marker for Star 1
ln2_marker, = ax.plot([], [], [], color='xkcd:red', marker='*', markersize=(m2.value / scale)**(2/3))  # Marker for Star 2

# Initialization function
def init():
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$X(R_{\odot})$')
    ax.set_ylabel(r'$Y(R_{\odot})$')
    ax.set_zlabel(r'$Z(R_{\odot})$')
    return [ln1_line, ln2_line, ln1_marker, ln2_marker]

# Update function
def update(frame):
    # Update line trajectories
    ln1_line.set_data(X1[:frame] - X_com[:frame], Y1[:frame] - Y_com[:frame])
    ln1_line.set_3d_properties(Z1[:frame] - Z_com[:frame])
    
    ln2_line.set_data(X2[:frame] - X_com[:frame], Y2[:frame] - Y_com[:frame])
    ln2_line.set_3d_properties(Z2[:frame] - Z_com[:frame])

    # Update marker for the last point
    ln1_marker.set_data([X1[:frame][-1].value - X_com[:frame][-1].value], [Y1[:frame][-1].value - Y_com[:frame][-1].value])
    ln1_marker.set_3d_properties([Z1[:frame][-1].value - Z_com[:frame][-1].value])
    
    ln2_marker.set_data([X2[:frame][-1].value] - X_com[:frame][-1].value, [Y2[:frame][-1].value - Y_com[:frame][-1].value])
    ln2_marker.set_3d_properties([Z2[:frame][-1].value - Z_com[:frame][-1].value])

    # Change marker style and color for Star 1 after the transition frame
    if frame < transition_frame:
        ln1_marker.set_marker('*')
        ln1_marker.set_color('xkcd:blue')
    else:
        if m1 != m1_post_SN:
            ln1_marker.set_marker('o')  # Black circle
            ln1_marker.set_color('xkcd:black')
            ln1_marker.set_markersize((m1_post_SN.value / scale)**(2/3))

    # Dynamically adjust axis limits
    xmin = min((X1[:frame] - X_com[:frame]).min(), (X2[:frame] - X_com[:frame]).min()) - 1*rsun
    xmax = max((X1[:frame] - X_com[:frame]).max(), (X2[:frame] - X_com[:frame]).max()) + 1*rsun
    ymin = min((Y1[:frame] - Y_com[:frame]).min(), (Y2[:frame] - Y_com[:frame]).min()) - 1*rsun
    ymax = max((Y1[:frame] - Y_com[:frame]).max(), (Y2[:frame] - Y_com[:frame]).max()) + 1*rsun
    zmin = min((Z1[:frame] - Z_com[:frame]).min(), (Z2[:frame] - Z_com[:frame]).min()) - 1*rsun
    zmax = max((Z1[:frame] - Z_com[:frame]).max(), (Z2[:frame] - Z_com[:frame]).max()) + 1*rsun

    ax.set_xlim(xmin.value, xmax.value)
    ax.set_ylim(ymin.value, ymax.value)
    ax.set_zlim(zmin.value, zmax.value)

    return [ln1_line, ln2_line, ln1_marker, ln2_marker]

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(1, N, 10),  # Start from 1
                    init_func=init, blit=False, interval=20)

# Save the animation as a GIF
ani.save(output_file + '_relative.gif', writer='pillow', dpi=200)