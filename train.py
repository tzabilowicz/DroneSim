# train.py

import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from environment.environment import QuadrotorEnv

MODEL_DIR="model"

os.makedirs(MODEL_DIR, exist_ok=True)

# Create the environment
env = QuadrotorEnv()
env = Monitor(env, f"{MODEL_DIR}/monitor.csv")

eval_env = QuadrotorEnv()

# Add the training callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=MODEL_DIR,
    eval_freq=10000,
    deterministic=True
)

# Model network capacity
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

# Create the PPO model
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.005,
    device="cpu",
    tensorboard_log="./model/tensorboard"
)

# Train the model
model.learn(
    total_timesteps=2_000_000,
    callback=eval_callback
)

# Save the best model
model.save(f"{MODEL_DIR}/ppo_quad")

print("[train.py] Training complete.")