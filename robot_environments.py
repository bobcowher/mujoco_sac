import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2

class RoboGymEnv(gym.Env):

    def __init__(self, robot, max_episode_steps, step_repeat=2):
        model_path = f"robots/{robot}/scene.xml"
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

        total_reward = 0
        
        for i in range(self.step_repeat):
            reward, done, truncated, info = self._step(action)
            total_reward += reward
            
            if done:
                break

        obs = self._get_obs()

        return obs, reward, done, truncated, info

    def _step(self, action):
        # Apply control input
        self.data.ctrl[:] = action

        # Step the simulation
        mujoco.mj_step(self.model, self.data)

        current_goal_distance = self.get_distance_to_goal()
        reward = self.last_distance - current_goal_distance
        self.last_distance = current_goal_distance

        done = False

        reward = reward * 1000    
        
        if(current_goal_distance <= self.success_threshold):
            reward += 1
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
