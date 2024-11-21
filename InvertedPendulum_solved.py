import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import linalg

# Create the environment
env = gym.make('CartPole-v1', render_mode='human').unwrapped

# Define the system matrices
A = np.array([[0, 1, 0, 0],
              [0, 0, -0.709, 0],
              [0, 0, 0, 1],
              [0, 0, 15.775, 0]])

B = np.array([[0],
              [0.974],
              [0],
              [-1.463]])

# Weight matrices for the LQR controller
Q = np.array([[50, 0, 0, 0],
              [0, 5, 0, 0],
              [0, 0, 5, 0],
              [0, 0, 0, 5]])

R = np.array([[0.05]])

# Solve the Riccati equation to compute the robust control gain
P = linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ np.transpose(B) @ P

print(f"Robust Control Gain K:\n{K}\n")

# Define a robust state controller function
def apply_robust_controller(K, x):
    """
    Implements robust state feedback control using LQR gain K.
    """
    u = -K @ x
    # CartPole expects a discrete action (0: left, 1: right)
    action = 1 if u > 0 else 0
    return action, u

# Run the simulation for 15 seconds
obs = np.array(env.reset(seed=1)[0])
x_array = []
t_array = []

# Track start time
start_time = time.time()

while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Render the environment
    env.render()

    # Apply robust controller
    action, force = apply_robust_controller(K, obs)

    # Apply action to environment
    obs, _, done, _, _ = env.step(action)

    # Log the states
    x_array.append(obs)
    t_array.append(elapsed_time)

    # Reset the environment if the episode ends
    if done:
        obs = np.array(env.reset(seed=1)[0])

    # Stop the simulation after 15 seconds
    if elapsed_time >= 15:  # 15 seconds
        print(f'Simulation ended after 15 seconds.')
        break

env.close()

# Convert logged states to numpy arrays for plotting
x_array = np.array(x_array)
t_array = np.array(t_array)

# Plot the system response
fig, ax = plt.subplots(x_array.shape[1], sharex=True, figsize=(10, 8))
fig.suptitle("System Response's Plot with Robust Control")

for i in range(x_array.shape[1]):
    ax[i].plot(t_array, x_array[:, i], '-r', lw=1)
    ax[i].set_ylabel(f"State {i + 1}")
    ax[i].grid()

ax[-1].set_xlabel("Time (s)")
plt.show()
