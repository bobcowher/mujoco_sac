import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

class RoboGymEnv(gym.Env):

    def __init__(self, robot):
        model_path = f"robots/{robot}/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Viewer setup (non-blocking)
        self.viewer = None

        # Setup action and observation spaces (example: torque control)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = self.model.nq + self.model.nv  # Position + velocity
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Reset simulation to initial state
        mujoco.mj_resetData(self.model, self.data)

        # Optional: add randomization here
        return self._get_obs(), {}

    def step(self, action):
        # Apply control input
        self.data.ctrl[:] = action

        # Step the simulation
        mujoco.mj_step(self.model, self.data)

        # Collect observation and compute reward
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        terminated = False   # Set this to True if task is done
        truncated = False    # Set to True if time limit or failure
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Simple observation: joint pos + vel
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _compute_reward(self, obs, action):
        # Example: move forward in +x
        base_pos = self.data.qpos[:3]
        return base_pos[0]  # Reward = forward movement

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data).__enter__() 
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.__exit__(None, None, None)
