import time
import os
import gym
import numpy as np
from buffer import ReplayBuffer 
import datetime
from agent import SAC
import torch
from robot_environments import RoboGymEnv
import sys

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    env_name = "boston_dynamics_spot"
    replay_buffer_size = 100000
    episodes = 3000
    warmup = 10
    batch_size = 64
    pretrain_batch_size = 64
    updates_per_step = 1 
    gamma = 0.99
    tau = 0.005
    alpha = 0.1 # Temperature parameter.
    min_alpha = alpha
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = False
    hidden_size = 512 
    learning_rate = 0.0001
    max_episode_steps=1500 # max episode steps
    alpha_decay = 0.0001
    step_repeat = 1 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = RoboGymEnv(robot=env_name, max_episode_steps=max_episode_steps, step_repeat=step_repeat)

    state, info = env.reset()

    joint_obs_size = state['joint_pos'].shape[0] + state['joint_vel'].shape[0]

    print(joint_obs_size)

    # print("Camera OBS Shape:", state['camera'].shape)
    # print(state['camera'])
    # print(state['joint_pos'])
    # print(state['joint_vel'])
    # sys.exit(1)

    # print(f"State shape: {state.shape}")

    
    # Agent
    agent = SAC(joint_obs_size, env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate, alpha_decay=alpha_decay, min_alpha=min_alpha,
                device=device, env=env)

    # Tesnorboard
    episode_identifier = f"Adam - lr: {learning_rate} HL: {hidden_size} A: {alpha} UPS: {updates_per_step} TUI: {target_update_interval} SR: {step_repeat} - clipped-rewards"

    summary_writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{episode_identifier}')



    # Memory
    memory = ReplayBuffer(replay_buffer_size, 
                          camera_shape=state['camera'].shape, 
                          joint_pos_dim=state['joint_pos'].shape[0],
                          joint_vel_dim=state['joint_vel'].shape[0],  
                          n_actions=env.action_space.shape[0],
                          input_device='cpu',
                          output_device=device)

    # Training Loop
    total_numsteps = 0
    updates = 0

    agent.train(episodes=episodes, 
                memory=memory, 
                updates_per_step=updates_per_step, 
                batch_size=batch_size, 
                summary_writer=summary_writer, 
                max_episode_steps=max_episode_steps,
                warmup=warmup)



    env.close()
