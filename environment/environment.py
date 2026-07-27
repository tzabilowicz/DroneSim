import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from utils.utils import randomize_target


class QuadrotorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    HOVER_THRUST = 3.2495625
    MAX_THRUST = 13.0

    def __init__(self, render_mode=None, random_target=False, frame_skip=4):
        """ Initialize the model parameters. """

        super().__init__()

        # Read and extract the model
        self.model = mujoco.MjModel.from_xml_path("assets/mujoco_menagerie/skydio_x2/scene.xml")
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * self.frame_skip

        # Set the target position
        self.random_target = random_target
        if self.random_target:
            self.target_position = randomize_target(50, 2)
        else:
            self.target_position = np.array([5, 5, 5], dtype=np.float32)

        target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target"
        )
        if target_body_id != -1:
            self.model.body_pos[target_body_id] = self.target_position

        # Set the models action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        self.max_steps = 500
        self.steps = 0

        self.render_mode = render_mode
        self.viewer = None


    def _action_to_thrust(self, action):
        """ Convert the action to drone trust values. """

        thrust = np.zeros_like(action)
        low_mask = action <= 0.0
        high_mask = action > 0.0

        thrust[low_mask] = self.HOVER_THRUST * (1.0 + action[low_mask])
        thrust[high_mask] = self.HOVER_THRUST + action[high_mask] * (self.MAX_THRUST - self.HOVER_THRUST)

        return thrust


    def get_obs(self):
        """ Get sensor observations. """

        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]

        target_delta = self.target_position - pos

        obs = np.concatenate([pos, vel, quat, ang_vel, target_delta])
        return obs.astype(np.float32)


    def reset(self, seed=None, options=None):
        """ Reset the simulation state. """

        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Sample a new target position
        if self.random_target:
            self.target_position = randomize_target(50, 2)

        # Spawn drone resting on ground
        self.data.qpos[:7] = [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.prev_distance = np.linalg.norm(self.target_position - self.data.qpos[:3])

        if self.render_mode == "human":
            self.render()

        return self.get_obs(), {}


    def step(self, action):
        """ Step the physics simulation. """

        # Convert [-1, 1] action to motor forces
        thrust = self._action_to_thrust(action)
        self.data.ctrl[:] = thrust

        # Sub-step physics engine
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1

        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]

        # Calculate upward z-axis projection to measure tilt
        w, x, y, z = quat
        z_axis_z = 1.0 - 2.0 * (x**2 + y**2)  # World Z projection of body frame

        distance = np.linalg.norm(self.target_position - pos)

        # Progress Reward (Dense)
        # Rewards the agent for closing the gap, penalizes for moving away.
        progress = self.prev_distance - distance
        self.prev_distance = distance

        # Scale progress so moving 1 meter yields ~10 points
        reward = progress * 10.0

        # Alive bonus
        is_airborne = pos[2] > 0.2
        alive_bonus = 0.1 if is_airborne else 0.0
        reward += alive_bonus

        # Small stability penalties
        reward -= 0.05 * np.linalg.norm(vel)
        reward -= 0.02 * np.linalg.norm(ang_vel)
        reward -= 0.1 * (1.0 - max(0.0, z_axis_z))

        # Crash conditions (allows resting at ground height pos[2] ~ 0.05 without crashing):
        # - Flipping past 90 degrees (z_axis_z < 0.0)
        # - Clipping below ground plane (z < 0.0) or flying out of vertical bounds (z > 80.0)
        # - Drifting too far horizontally (|x| > 80.0 or |y| > 80.0)
        crashed = (
            pos[2] < 0.0 or pos[2] > 80.0 or
            abs(pos[0]) > 80.0 or abs(pos[1]) > 80.0 or
            z_axis_z < 0.0
        )

        # Reward for target reached
        target_reached = distance < 0.1 and np.linalg.norm(vel) < 0.2

        terminated = False
        truncated = False

        # Process the end state
        if crashed:
            reward = -100.0
            terminated = True
        elif target_reached:
            time_bonus = (self.max_steps - self.steps) * alive_bonus
            reward += 100.0 + time_bonus
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True

        # Render sim
        if self.render_mode == "human":
            self.render()

        return (
            self.get_obs(),
            float(reward),
            terminated,
            truncated,
            {"distance": distance},
        )

    def render(self):
        """ Render the simulation. """

        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self.viewer.sync()

        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mujoco.Renderer(self.model)

            self.viewer.update_scene(self.data)

            return self.viewer.render()

    def close(self):
        """ Close the simulation. """

        if self.viewer is not None:
            if hasattr(self.viewer, "close"):
                self.viewer.close()
            self.viewer = None