import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
m = 1.0   # Mass (kg)
I = 0.01  # Moment of inertia (kg·m^2)
l = 0.5   # Arm length (m)
g = 9.81  # Gravitational acceleration (m/s^2)
dt = 0.01  # Time step

# PID Controller
class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def update(self, measurement):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

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

# PID Controllers
x_pid = PIDController(kp=1.5, ki=0.1, kd=0.5, setpoint=0)  # Stabilize x-axis at 0
theta_pid = PIDController(kp=2.0, ki=0.1, kd=1, setpoint=0)  # Stabilize orientation to target theta
altitude_pid = PIDController(kp=15.0, ki=5.0, kd=10.0, setpoint=2.0)  # Hover at y=2.0

# Simulation parameters
time_steps = 500
state_history = []
u_history = []

# Initial thrust values
u1, u2 = 5.0, 5.0

# Run simulation with PID control
state = env.reset()
for t in range(time_steps):
    x, y, theta, dx, dy, dtheta = state

    # PID outputs
    theta_target = x_pid.update(x)  # Target theta based on x-axis error
    torque = theta_pid.update(theta - theta_target)  # Adjust for orientation
    total_thrust = altitude_pid.update(y)  # Adjust for altitude

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
