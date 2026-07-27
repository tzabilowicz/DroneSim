import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from stable_baselines3 import PPO
from environment.environment import QuadrotorEnv

MODEL = "model/ppo_quad.zip"

# Create the environment
env = QuadrotorEnv(render_mode="human")

# Load the best model
model = PPO.load(MODEL)

# Number of episodes to simulate
episodes = 10

all_rewards = []
all_distance = []

# Initialize a single 3D figure before running episodes
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection="3d")

# Plot target point once
target = env.target_position
ax.scatter(
    target[0], target[1], target[2],
    color="red",
    marker="*",
    s=150,
    label=f"Target ({target[0]}, {target[1]}, {target[2]})"
)

for ep in range(episodes):
    print(f"Processing episode: {ep+1}")
    obs, _ = env.reset()
    done = False

    trajectory = []
    actions = []
    reward_total = 0

    while not done:
        # Run the model
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        time.sleep(env.dt)

        reward_total += reward

        trajectory.append(env.data.qpos[:3].copy())
        actions.append(action)

    # Drone trajectory
    trajectory = np.array(trajectory)

    # All rewards & distances
    all_rewards.append(reward_total)
    all_distance.append(info["distance"])

    # Add this episode's trajectory to the shared plot
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        label=f"Ep {ep+1}",
        alpha=0.8
    )

# Formatting the combined 3D plot
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Combined Drone Trajectories")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("model/trajectory.png")
plt.show()
plt.close()


print("===================")
print("Average reward:", np.mean(all_rewards))
print("Average final distance:", np.mean(all_distance))