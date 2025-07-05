import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import time
import math


from numpy.random import f

class RoboGymEnv(gym.Env):

    def __init__(self, robot, max_episode_steps, step_repeat=2):
        model_path = f"robots/{robot}/scene.xml"
        
        self.viewer = None
        self.step_repeat = step_repeat
        self.model = mujoco.MjModel.from_xml_path(model_path)
        # Order must match the actuator order in spot.xml
        joint_names = ["fl_hx","fl_hy","fl_kn",
                       "fr_hx","fr_hy","fr_kn",
                       "hl_hx","hl_hy","hl_kn",
                       "hr_hx","hr_hy","hr_kn"]

        self.jnt_ids  = [self.model.joint(name).id for name in joint_names]
        jnt_ranges    = self.model.jnt_range[self.jnt_ids]          # (12, 2)

        self.jnt_mid  = 0.5 * (jnt_ranges[:, 0] + jnt_ranges[:, 1]) # (12,)
        self.jnt_half = 0.5 * (jnt_ranges[:, 1] - jnt_ranges[:, 0]) # (12,)

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
        img_shape = self._get_image_obs().shape
        self.observation_space = gym.spaces.Dict({
            'camera': gym.spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8),
            'joint_pos': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq,), dtype=np.float32),
            'joint_vel': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nv,), dtype=np.float32),
        })

        # print(f"Obs dim: {obs_dim}"
        
        # print(f"Obs dim: {self.model.nq}")
        # print(f"Obs dim: {self.model.nv}")
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def get_body_position(self, name):
        return self.data.xpos[self.model.body(name).id]

    def get_robot_height(self, name="front_camera_mount"):
        return self.get_body_position("front_camera_mount")[2]

    def get_distance_to_goal(self):
        #robot_pos = self.data.qpos[:3]
        #print(f"Robot pos: {robot_pos}")
        robot_pos = self.get_body_position("front_camera_mount")
        #print(f"Robot pos 2: {robot_pos}")
        #print(f"Goal pos: {self.goal_pos}")
        #time.sleep(1)

        distance = np.linalg.norm(robot_pos[:2] - self.goal_pos[:2])

        return distance


    def reset(self, *, seed=None, options=None):
        # Reset simulation to initial state
        # mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        mujoco.mj_forward(self.model, self.data) # Data comes back as 0 without this.

        self.goal_pos = self.get_body_position("goal")
        self.last_goal_distance = self.get_distance_to_goal() 

        self.current_step = 0

        # Optional: add randomization here
        return self._get_obs(), {}


    def step(self, action):

        reward = 0
    
        done = False
        truncated = False    # Set to True if time limit or failure
        
        for i in range(self.step_repeat):
            self._step(action)
            
        obs = self._get_obs()
        
        if not done:
            self.current_step += 1
            if self.current_step >= self.max_episode_steps:
                done = True
                truncated = True

        # Get current Goal Distance and Compute Reward
        current_goal_distance = self.get_distance_to_goal()
        progress = self.last_goal_distance - current_goal_distance
        reward = np.clip(100 * progress, -20.0, 20.0)                             # keep range stable

        # Set last goal distance to current goal distance. 
        self.last_goal_distance = current_goal_distance
        
        # Reward success highly
        if(current_goal_distance <= self.success_threshold):
            reward += 100
            done = True

        if(self.get_robot_height() < 0.2):
            reward = -10
            done = True
            truncated = True

        info = {}

        return obs, reward, done, truncated, info


    def _step(self, action):
        # Start with done as false.
        done = False

        action = self.jnt_mid + action * self.jnt_half
        # Apply control input
        
        self.data.ctrl[:] = action

        # Step the simulation
        mujoco.mj_step(self.model, self.data)


    def _get_image_obs(self):
        #obs = mujoco._render(self.data, self.model, width=128, height=128, camera="front_camera")
        self.renderer.update_scene(self.data, camera="forward_camera")
        front_img = self.renderer.render()
        
        #self.renderer.update_scene(self.data, camera="down_camera")
        #bottom_img = self.renderer.render()

        # img = np.concatenate([front_img, bottom_img], dtype=np.uint8)
        #print(img.shape)
        #sys.exit(1)
        img = cv2.resize(front_img, (80, 80), interpolation=cv2.INTER_AREA)
        # print(f"Image shape after resize: {img.shape}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(f"Image shape after cvtColor: {img.shape}")
        img = np.expand_dims(img, 0)
        # print(f"Image after expansion: {img}")
        
        return img 


    def _get_obs(self):
        # Simple observation: joint pos + vel
        # Normalizing joint_pos and joint_vel
        obs = {'camera': self._get_image_obs(),
               'joint_pos': self.data.qpos / math.pi,
               'joint_vel': self.data.qvel / 25}

        return obs 


    def render(self, front_camera=False):

        if not front_camera:
           plt.imshow(self._get_image_obs().squeeze(0))
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
