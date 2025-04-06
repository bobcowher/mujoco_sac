import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import sys

class RoboGymEnv(gym.Env):

    def __init__(self, robot, max_episode_steps):
        model_path = f"robots/{robot}/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        obs, info = self.reset()
        self.success_threshold = 1
        self.max_episode_steps = max_episode_steps
#        self.goal_id = self.model.body(name="goal").id

#        self.goal_pos = self.data.xpos[self.goal_id]

        print(f"self.goal_pos: {self.goal_pos}")

        print(self.goal_pos) 

        # Viewer setup (non-blocking)
        self.viewer = None

        # Setup action and observation spaces (example: torque control)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = obs.shape[0]  # Position + velocity + target position
        # print(f"Obs dim: {obs_dim}"
        
        # print(f"Obs dim: {self.model.nq}")
        # print(f"Obs dim: {self.model.nv}")
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def get_body_position(self, name):
        return self.data.xpos[self.model.body(name).id]
    
    def get_distance_to_goal(self):
        robot_pos = self.data.qpos[:3]
        distance = np.linalg.norm(robot_pos[:2] - self.goal_pos[:2])
        return distance


    def reset(self, *, seed=None, options=None):
        # Reset simulation to initial state
        mujoco.mj_resetData(self.model, self.data)

        mujoco.mj_forward(self.model, self.data) # Data comes back as 0 without this.

        self.goal_pos = self.get_body_position("goal")
        self.last_distance = self.get_distance_to_goal() 

        self.current_step = 0

        # Optional: add randomization here
        return self._get_obs(), {}

    def step(self, action):
        # Apply control input
        self.data.ctrl[:] = action

        # Step the simulation
        mujoco.mj_step(self.model, self.data)

        # Collect observation and compute reward
        obs = self._get_obs()
        # reward = self._compute_reward(obs, action)
        current_goal_distance = self.get_distance_to_goal()
        reward = self.last_distance - current_goal_distance
        self.last_distance = current_goal_distance

        done = False

        if(current_goal_distance <= self.success_threshold):
            reward += 1
            done = True

        reward = reward * 100    

        truncated = False    # Set to True if time limit or failure
        info = {}

        if not done:
            self.current_step += 1
            if self.current_step >= self.max_episode_steps:
                done = True

        return obs, reward, done, truncated, info

    def _get_obs(self):
        # Simple observation: joint pos + vel

        obs = np.concatenate([self.data.qpos, self.data.qvel, self.goal_pos])

        for i in range(self.model.nbody):
            if not '_' in self.model.body(i).name:
                obs = np.concatenate([obs, self.data.xpos[i]])

        return obs 


    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data).__enter__() 
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.__exit__(None, None, None)
