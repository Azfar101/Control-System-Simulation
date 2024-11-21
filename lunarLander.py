# Lunar Lander Simulation with Target Position, Without Predefined Control System

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
g = 9.81  # Gravitational acceleration (m/s^2)
m = 1.0   # Mass of the lander (kg)
I = 0.1   # Moment of inertia (kg·m^2)
L = 0.5   # Distance between the side thrusters (m)
lander_size = 0.5  # Size of the lander for visualization
disturbance_force = 1.0  # Disturbance magnitude
disturbance_interval = 2.0  # Time interval for disturbances (seconds)

# Placeholder for student-defined control system
def control_system(state):
    """
    Placeholder for a control system to stabilize the lunar lander.
    Parameters:
    - state: [x, y, vx, vy, theta, omega] where:
        - x, y: Position (m)
        - vx, vy: Velocities (m/s)
        - theta: Orientation angle (rad)
        - omega: Angular velocity (rad/s)
    Returns:
    - [T_m, T_left, T_right]: Main thruster and left and right side thruster forces.
    """
    # Students can define their own control logic here
    T_m = 0.0    # Main thruster force (N)
    T_left = 0.0 # Left thruster force (N)
    T_right = 0.0 # Right thruster force (N)
    return [T_m, T_left, T_right]

# State-space dynamics for the lunar lander
def lunar_lander_dynamics(t, state):
    """
    Computes the dynamics of the lunar lander with disturbances.
    Parameters:
    - t: Current time (seconds).
    - state: [x, y, vx, vy, theta, omega].
    Returns:
    - Derivatives: [vx, vy, ax, ay, angular velocity, angular acceleration].
    """
    x, y, vx, vy, theta, omega = state
    T_m, T_left, T_right = control_system(state)  # Get control forces

    # Total side thrust
    T_side = T_left - T_right

    # Equations of motion
    ax = T_m / m * np.sin(theta) + T_side / m
    ay = -g + T_m / m * np.cos(theta)
    alpha = (T_left - T_right) * L / I  # Angular acceleration from torque

    # Add periodic disturbance to horizontal acceleration
    if int(t) % disturbance_interval == 0 and int(t * 100) % 100 == 0:
        ax += disturbance_force / m

    # Collision detection with the ground
    left_edge_y = y - (lander_size / 2) * np.cos(theta)
    right_edge_y = y + (lander_size / 2) * np.cos(theta)
    if left_edge_y <= 0 or right_edge_y <= 0:
        ay = 0
        vy = 0
        ax = 0
        vx = 0
        omega = 0

    return [vx, vy, ax, ay, omega, alpha]

# Initial conditions
initial_state = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]  # Starting position far from the target

# Simulation parameters
t_span = (0, 20)  # Simulate for 20 seconds
dt = 0.01         # Time step
time = np.arange(t_span[0], t_span[1], dt)

# Solve the system
solution = solve_ivp(lunar_lander_dynamics, t_span, initial_state, t_eval=time)

# Extract results
x = solution.y[0]
y = solution.y[1]
theta = solution.y[4]

# Visualization: Animate the Lunar Lander
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 5)
ax.set_ylim(0, 12)
ax.set_xlabel("Horizontal Position (x)")
ax.set_ylabel("Vertical Position (y)")
ax.set_title("Lunar Lander Simulation with Target at (1, 1) (No Control)")

# Ground line
ground, = ax.plot([-5, 5], [0, 0], 'k-', lw=2)

# Target marker
target, = ax.plot([1.0], [1.0], 'go', label="Target")

# Lander representation
lander_body, = ax.plot([], [], 's-', markersize=15, label="Lander")

# Thrust representation
main_thrust, = ax.plot([], [], 'r-', lw=2, label="Main Thrust")
left_thrust, = ax.plot([], [], 'b-', lw=2, label="Left Thrust")
right_thrust, = ax.plot([], [], 'g-', lw=2, label="Right Thrust")

# Initialize the animation
def init():
    lander_body.set_data([], [])
    main_thrust.set_data([], [])
    left_thrust.set_data([], [])
    right_thrust.set_data([], [])
    return lander_body, main_thrust, left_thrust, right_thrust, ground, target

# Update function for the animation
def update(frame):
    lander_x = x[frame]
    lander_y = y[frame]
    lander_theta = theta[frame]

    # Lander corners for square visualization
    corners_x = lander_x + lander_size * np.array([-0.5, 0.5, 0.5, -0.5, -0.5]) * np.cos(lander_theta) \
                        - lander_size * np.array([0.5, 0.5, -0.5, -0.5, 0.5]) * np.sin(lander_theta)
    corners_y = lander_y + lander_size * np.array([-0.5, 0.5, 0.5, -0.5, -0.5]) * np.sin(lander_theta) \
                        + lander_size * np.array([0.5, 0.5, -0.5, -0.5, 0.5]) * np.cos(lander_theta)
    lander_body.set_data(corners_x, corners_y)

    # Thrust visualization
    T_m, T_left, T_right = control_system([lander_x, lander_y, 0, 0, lander_theta, 0])
    main_thrust.set_data([lander_x, lander_x], [lander_y, lander_y - 0.1 * T_m])
    left_thrust.set_data([lander_x, lander_x - 0.1 * T_left], [lander_y, lander_y])
    right_thrust.set_data([lander_x, lander_x + 0.1 * T_right], [lander_y, lander_y])

    return lander_body, main_thrust, left_thrust, right_thrust, ground, target

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=dt * 1000)

plt.legend()
plt.show()
