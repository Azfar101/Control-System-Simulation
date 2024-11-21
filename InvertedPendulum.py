import gym
import numpy as np
import matplotlib.pyplot as plt
import time

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

# Placeholder function for applying control (students can implement their logic here)
def apply_state_controller(x):
    """
    Placeholder for control logic.
    Replace this with your control system implementation.
    """
    # Default: Random action as placeholder
    action = env.action_space.sample()  # Random action (0 or 1)
    return action

# Run the simulation for 15 minutes (900 seconds)
obs = np.array(env.reset(seed=1)[0])
reward_total = 0
x_array = []
t_array = []

# Track start time
start_time = time.time()

while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Render the environment
    env.render()

    # Apply random action (placeholder)
    action = apply_state_controller(obs)

    # Step in the environment
    obs, reward, done, _, _ = env.step(action)

    # Log the states
    x_array.append(obs)
    t_array.append(elapsed_time)

    # Accumulate reward
    reward_total += reward

    # Reset the environment if the episode ends
    if done:
        obs = np.array(env.reset(seed=1)[0])

    # Stop the simulation after 15 sec
    if elapsed_time >= 15:  # 
        print(f'Simulation ended after 15 minutes.')
        print(f"Total reward: {reward_total}")
        break

env.close()

# Convert logged states to numpy arrays for plotting
x_array = np.array(x_array)
t_array = np.array(t_array)

# Plot the system response
fig, ax = plt.subplots(x_array.shape[1], sharex=True, figsize=(10, 8))
fig.suptitle("System Response's Plot")

for i in range(x_array.shape[1]):
    ax[i].plot(t_array, x_array[:, i], '-r', lw=1)
    ax[i].set_ylabel(f"State {i + 1}")
    ax[i].grid()

ax[-1].set_xlabel("Time (s)")
plt.show()
