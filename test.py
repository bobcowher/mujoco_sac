import time
import os
import gym
import numpy as np
from buffer import ReplayBuffer, CombinedReplayBuffer
import datetime
from agent import SAC
import torch
from robot_environments import RoboGymEnv

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    env_name = "Stack"
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
    hidden_size = 756
    learning_rate = 0.0001
    max_episode_steps=500 # max episode steps

    env = RoboGymEnv(robot="boston_dynamics_spot", max_episode_steps=max_episode_steps)


    print(env.observation_space.shape[0])
    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate)

    agent.load_checkpoint()

    # Memory
    memory = ReplayBuffer(replay_buffer_size, input_shape=env.observation_space.shape, n_actions=env.action_space.shape[0])

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in range(episodes):
        episode_reward = 0
        episode_steps = 0
        done = False
        state, info = env.reset()

        while not done:
            
            action = agent.select_action(state)  # Sample action from policy

            next_state, reward, done, _, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            env.render()

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)

            state = next_state



    env.close()