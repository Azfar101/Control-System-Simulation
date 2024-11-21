# Mountain Car Simulation With Robust Control to Stabilize at Peak

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# System parameters
g = 9.81  # Gravitational constant (m/s^2)
m = 1.0   # Mass of the car (kg)
k = 3.0   # Scaling factor for the slope's curvature

# Shift the mountain profile so the peak is at the center
peak_shift = np.pi / (2 * k)  # Horizontal shift to align peak at x = 0

# Control parameters for robust control
Kp = 50.0  # Proportional gain
Kd = 10.0  # Derivative gain

# Robust control system
def robust_control(state):
    """
    Robust control system to stabilize the car at the top of the mountain.
    Parameters:
    - state: [x, v] where x is position and v is velocity.
    Returns:
    - Force applied to the car (N).
    """
    x, v = state
    x_target = 0.0  # Target position at the peak of the mountain
    # Control law: Robust PD controller
    F = -Kp * (x - x_target) - Kd * v
    return np.clip(F, -10.0, 10.0)  # Limit force for stability

# State-space dynamics of the Mountain Car with control input
def mountain_car_dynamics(t, state):
    """
    Computes the dynamics of the mountain car with robust control.
    Parameters:
    - t: Current time (seconds).
    - state: [x, v] where x is position and v is velocity.
    Returns:
    - [v, a]: Derivatives of position and velocity.
    """
    x, v = state
    F = robust_control(state)  # Control input
    a = -g * np.cos(k * (x + peak_shift)) + F / m  # Acceleration with control
    return [v, a]

# Initial conditions
initial_state = [0.5, 0.0]  # Starting near the peak with no velocity

# Simulation parameters
t_span = (0, 10)  # Simulate for 10 seconds
dt = 0.01         # Time step
time = np.arange(t_span[0], t_span[1], dt)

# Solve the system
solution = solve_ivp(mountain_car_dynamics, t_span, initial_state, t_eval=time)

# Extract results
x = solution.y[0]
v = solution.y[1]

# Visualization: Animate the Mountain Car
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.0, 1.0)
ax.set_xlabel("Position (x)")
ax.set_ylabel("Height")
ax.set_title("Mountain Car Stabilized at Peak Using Robust Control")

# Mountain and Car representation
mountain_x = np.linspace(-1.2, 1.2, 500)
mountain_y = np.sin(k * (mountain_x + peak_shift)) / k  # Adjusted mountain profile
car, = ax.plot([], [], 'ro', markersize=8)
mountain_line, = ax.plot(mountain_x, mountain_y, 'k-', lw=2)

# Initialize the animation
def init():
    car.set_data([], [])
    return car, mountain_line

# Update function for the animation
def update(frame):
    car_x = x[frame]
    car_y = np.sin(k * (car_x + peak_shift)) / k  # Compute height from the adjusted mountain profile
    car.set_data(car_x, car_y)
    return car, mountain_line

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=dt * 1000)

plt.show()
