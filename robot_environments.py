import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import time

class RoboGymEnv(gym.Env):

    def __init__(self, robot, max_episode_steps, step_repeat=2):
        model_path = f"robots/{robot}/scene.xml"
        
        self.viewer = None
        self.step_repeat = step_repeat
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.success_threshold = 1
        self.max_episode_steps = max_episode_steps
        self.renderer = mujoco.Renderer(self.model)
        obs, info = self.reset()
        
        #        self.goal_id = self.model.body(name="goal").id

#        self.goal_pos = self.data.xpos[self.goal_id]

        # print(f"self.goal_pos: {self.goal_pos}")
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = obs['camera'].shape[0]  # Position + velocity + target position
        # print(f"Obs dim: {obs_dim}"
        
        # print(f"Obs dim: {self.model.nq}")
        # print(f"Obs dim: {self.model.nv}")
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def get_body_position(self, name):
        return self.data.xpos[self.model.body(name).id]
    
    def get_distance_to_goal(self):
        #robot_pos = self.data.qpos[:3]
        #print(f"Robot pos: {robot_pos}")
        robot_pos = self.get_body_position("front_camera_mount")
        #print(f"Robot pos 2: {robot_pos}")
        #print(f"Goal pos: {self.goal_pos}")
        #time.sleep(1)

        distance = np.linalg.norm(robot_pos[:2] - self.goal_pos[:2])
        distance = distance * 1000

        return distance


    def reset(self, *, seed=None, options=None):
        # Reset simulation to initial state
        mujoco.mj_resetData(self.model, self.data)

        mujoco.mj_forward(self.model, self.data) # Data comes back as 0 without this.

        self.goal_pos = self.get_body_position("goal")
        self.nearest_distance = self.get_distance_to_goal() 

        self.current_step = 0

        # Optional: add randomization here
        return self._get_obs(), {}

    def step(self, action):

        total_reward = 0
        
        for i in range(self.step_repeat):
            reward, done, truncated, info = self._step(action)
            total_reward += reward
            
            if done:
                break

        obs = self._get_obs()

        return obs, total_reward, done, truncated, info

    def _step(self, action):
        # Start with done as false.
        done = False

        # Apply control input
        self.data.ctrl[:] = action

        # Step the simulation
        mujoco.mj_step(self.model, self.data)

        # Get current Goal Distance and Compute Reward
        current_goal_distance = self.get_distance_to_goal()
        reward = max(0, self.nearest_distance - current_goal_distance)
        self.nearest_distance = min(self.nearest_distance, current_goal_distance)


        # Get raw reward from the environment and multiply it by 1000.
        #reward = reward * 100    
        #reward = np.clip(reward, -0.05, 1) # Clip upper considerably higher than lower. Don't over-penalize lower scores. 

        # Penalize Thrashing
        # reward -= 0.1 * np.linalg.norm(action)

        # Reward success highly
        if(current_goal_distance <= self.success_threshold):
            reward += 100
            done = True

        truncated = False    # Set to True if time limit or failure
        info = {}

        if not done:
            self.current_step += 1
            if self.current_step >= self.max_episode_steps:
                done = True

        return reward, done, truncated, info


    def _get_image_obs(self):
        #obs = mujoco._render(self.data, self.model, width=128, height=128, camera="front_camera")
        self.renderer.update_scene(self.data, camera="forward_camera")
        front_img = self.renderer.render()
        
        self.renderer.update_scene(self.data, camera="down_camera")
        bottom_img = self.renderer.render()

        img = np.concatenate([front_img, bottom_img], dtype=np.uint8)
        #print(img.shape)
        #sys.exit(1)
        img = cv2.resize(img, (160, 240), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img 


    def _get_obs(self):
        # Simple observation: joint pos + vel
        obs = {'camera': self._get_image_obs(),
               'joint_pos': self.data.qpos,
               'joint_vel': self.data.qvel}

        return obs 


    def render(self, front_camera=False):

        if not front_camera:
           plt.imshow(self._get_image_obs())
           plt.axis('off')
           plt.pause(0.0001)
           plt.clf() 
        else:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data).__enter__() 
            self.viewer.sync()


    def close(self):
        if self.viewer:
            self.viewer.__exit__(None, None, None)
