import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
m = 1.0   # Mass (kg)
I = 0.01  # Moment of inertia (kg·m^2)
l = 0.5   # Arm length (m)
g = 9.81  # Gravitational acceleration (m/s^2)
dt = 0.01  # Time step

# Quadrotor Environment
class Quadrotor2DEnv:
    def __init__(self):
        # Initial state: 15 degrees (converted to radians) for theta
        self.state = np.array([0, 0, np.radians(15), 0, 0, 0])  # [x, y, theta, dx, dy, dtheta]
    
    def dynamics(self, state, u):
        x, y, theta, dx, dy, dtheta = state
        u1, u2 = u
        # Linear accelerations
        ddx = (u1 + u2) * np.sin(theta) / m  # Horizontal force
        ddy = (u1 + u2) * np.cos(theta) / m - g  # Upward thrust counters gravity
        ddtheta = l * (u2 - u1) / I  # Torque due to thrust difference
        return np.array([dx, dy, dtheta, ddx, ddy, ddtheta])
    
    def step(self, u):
        u = np.clip(u, 0, 10)  # Limit thrust to non-negative range
        self.state += dt * self.dynamics(self.state, u)
        return self.state

    def reset(self):
        self.state = np.array([0, 0, np.radians(15), 0, 0, 0])  # Reset to 15 degrees initial orientation
        return self.state

# Initialize environment
env = Quadrotor2DEnv()

# Placeholder functions for student to implement their control system
def control_x(state, target_x):
    """
    Compute the control signal for stabilizing x-axis.
    :param state: Current state of the drone [x, y, theta, dx, dy, dtheta]
    :param target_x: Desired x-position
    :return: Target theta to stabilize x-axis
    """
    return 0  # Placeholder, students should implement this

def control_theta(state, target_theta):
    """
    Compute the control signal for stabilizing theta.
    :param state: Current state of the drone [x, y, theta, dx, dy, dtheta]
    :param target_theta: Desired theta
    :return: Torque to stabilize theta
    """
    return 0  # Placeholder, students should implement this

def control_y(state, target_y):
    """
    Compute the control signal for stabilizing y-axis.
    :param state: Current state of the drone [x, y, theta, dx, dy, dtheta]
    :param target_y: Desired y-position
    :return: Total thrust to stabilize y-axis
    """
    return 0  # Placeholder, students should implement this

# Simulation parameters
time_steps = 500
state_history = []
u_history = []

# Initial thrust values
u1, u2 = 5.0, 5.0

# Targets
target_x = 0  # Target x-position
target_y = 2  # Target y-position
target_theta = 0  # Target theta (level orientation)

# Run simulation
state = env.reset()
for t in range(time_steps):
    x, y, theta, dx, dy, dtheta = state

    # Compute control signals (students will implement these functions)
    theta_target = control_x(state, target_x)  # Target theta based on x-axis error
    torque = control_theta(state, theta_target)  # Adjust for orientation
    total_thrust = control_y(state, target_y)  # Adjust for altitude

    # Compute motor thrusts
    u1 = (total_thrust - torque / l) / 2
    u2 = (total_thrust + torque / l) / 2

    # Apply thrusts
    state = env.step([u1, u2])
    state_history.append(state.copy())
    u_history.append([u1, u2])

# Convert to arrays
state_history = np.array(state_history)
u_history = np.array(u_history)

# Visualization
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)  # Widened view
ax.set_ylim(-2, 5)
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2, markersize=10, label="Quadrotor")
thrust_left, = ax.plot([], [], 'r-', lw=2, label="Left Motor Thrust (Downward)")
thrust_right, = ax.plot([], [], 'b-', lw=2, label="Right Motor Thrust (Downward)")

def init():
    line.set_data([], [])
    thrust_left.set_data([], [])
    thrust_right.set_data([], [])
    return line, thrust_left, thrust_right

def update(frame):
    state = state_history[frame]
    u = u_history[frame]
    x, y, theta = state[0], state[1], state[2]

    # Quadrotor body
    left_x = x - l * np.cos(theta)
    left_y = y - l * np.sin(theta)
    right_x = x + l * np.cos(theta)
    right_y = y + l * np.sin(theta)
    line.set_data([left_x, right_x], [left_y, right_y])

    # Thrust visualization (always downward)
    thrust_scale = 0.1  # Scale factor for visualization
    thrust_left.set_data([left_x, left_x], [left_y, left_y - u[0] * thrust_scale])
    thrust_right.set_data([right_x, right_x], [right_y, right_y - u[1] * thrust_scale])

    return line, thrust_left, thrust_right

ani = FuncAnimation(fig, update, frames=time_steps, init_func=init, blit=True, interval=20)
plt.legend()
plt.show()
