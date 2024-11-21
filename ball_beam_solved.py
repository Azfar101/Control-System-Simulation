# Ball and Beam Simulation with Wind-Like Disturbance (No Noise)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# System parameters
g = 9.81       # Gravity (m/s^2)
R = 0.02       # Ball radius (m)
L = 1.0        # Beam length (m)
alpha = g / (1 + (2 * R**2) / (5 * L**2))  # System constant

# Control parameters
K = np.array([50.0, 20.0])  # State feedback gain

# Simulation parameters
dt = 0.01  # Time step

# Initial conditions
initial_state = [L / 2 + 0.3 , 0.0]  # Ball starts at the center of the beam with zero velocity

# State variables
current_state = initial_state.copy()
beam_angle = [0.0]  # Mutable variable to allow updates

# Disturbance parameters
disturbance_start = 2.0  # Time when disturbance starts (seconds)
disturbance_duration = 2.0  # Duration of the disturbance (seconds)
disturbance_force = 0.2  # Magnitude of the wind-like disturbance

# State-space model with wind disturbance (no noise)
def ball_and_beam_dynamic(t, state):
    x, x_dot = state
    x_relative = x - L / 2  # Recenter the system around the beam's midpoint
    # Control input (beam angle) using proper feedback
    u = K[0] * x_relative + K[1] * x_dot  # Corrected control sign
    u = np.clip(u, -0.2, 0.2)  # Allow a slightly larger range for the beam angle
    beam_angle[0] = u  # Update the global beam angle

    # Add wind-like disturbance during the specified time interval
    disturbance = disturbance_force if disturbance_start <= t <= (disturbance_start + disturbance_duration) else 0

    # Compute dynamics with disturbance
    x_ddot = -alpha * np.sin(u) + disturbance
    return [x_dot, x_ddot]

# Reset simulation if the ball falls off the beam
def reset_simulation():
    global current_state, solution_time, solution_states
    current_state = [L / 2, 0.0]  # Reset state to the midpoint of the beam
    solution_time = [0]
    solution_states = [current_state]

# Solve the system iteratively
solution_time = [0]  # Start at t = 0
solution_states = [initial_state]  # Start with initial conditions

def update_solution():
    global solution_time, solution_states, current_state
    t_last = solution_time[-1]
    # Solve for the next small time step
    sol = solve_ivp(ball_and_beam_dynamic, [t_last, t_last + dt], current_state, t_eval=[t_last + dt])
    solution_time.append(sol.t[-1])
    solution_states.append(sol.y[:, -1])
    current_state = sol.y[:, -1]
    # Check if the ball falls off the beam
    if abs(current_state[0] - L / 2) > L / 2:
        reset_simulation()

# Matplotlib setup for animation
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-L / 2 - 0.1, L / 2 + 0.1)  # Add margin for visualization
ax.set_ylim(-0.2, 0.2)
ax.set_xlabel("Beam (m)")
ax.set_ylabel("Height (m)")
ax.set_title("Ball and Beam System with Wind-Like Disturbance (No Noise)")

# Beam representation
beam_line, = ax.plot([], [], 'k-', lw=2)
# Ball representation
ball, = ax.plot([], [], 'ro', markersize=8)

# Initialize the animation
def init():
    beam_line.set_data([], [])
    ball.set_data([], [])
    return beam_line, ball

# Update function for the animation
def update_dynamic(frame):
    update_solution()
    state = current_state
    x = state[0]

    # Beam ends with current angle
    angle = beam_angle[0]
    beam_x = [-L / 2 * np.cos(angle), L / 2 * np.cos(angle)]
    beam_y = [-L / 2 * np.sin(angle), L / 2 * np.sin(angle)]
    beam_line.set_data(beam_x, beam_y)

    # Ball position with respect to the beam's rotation
    ball_x = beam_x[0] + x * np.cos(angle)
    ball_y = beam_y[0] + x * np.sin(angle)
    ball.set_data(ball_x, ball_y)
    return beam_line, ball

# Create the animation
ani_dynamic = animation.FuncAnimation(
    fig, update_dynamic, frames=2000, init_func=init, blit=True, interval=dt * 1000
)

plt.show()
