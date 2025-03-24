import mujoco
import mujoco.viewer
import numpy as np
import time
from robot_environments import RoboGymEnv

# Load the Humanoid model
env = RoboGymEnv(robot="boston_dynamics_spot")

env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # print(f"Obs: {obs} Reward: {reward} Term: {terminated} Trunc: {truncated} Info: {info}")
    env.render()
    time.sleep(0.01)