import time
import os
import gym
import numpy as np
import datetime
from agent import SAC
import torch
from robot_environments import RoboGymEnv

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    env_name = "boston_dynamics_spot"
    replay_buffer_size = 10000000
    episodes = 1
    warmup = 20
    batch_size = 64
    pretrain_batch_size = 64
    updates_per_step = 1
    gamma = 0.99
    tau = 0.005
    alpha = 0.1 # Temperature parameter.
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = False
    hidden_size = 64 
    learning_rate = 0.0001
    max_episode_steps=3000 # max episode steps

    env = RoboGymEnv(robot="boston_dynamics_spot", max_episode_steps=max_episode_steps)


    print(env.observation_space.shape[0])
    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate, alpha_decay=0.01)

    agent.load_checkpoint()

    agent.test(env=env)

    env.close()