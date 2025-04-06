import time
import os
import gym
import numpy as np
from buffer import ReplayBuffer, CombinedReplayBuffer
import datetime
from agent import SAC
import torch
from robot_environments import RoboGymEnv
import sys

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    env_name = "boston_dynamics_spot"
    replay_buffer_size = 10000000
    episodes = 3000
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
    hidden_size = 256
    learning_rate = 0.0001
    max_episode_steps=3000 # max episode steps

    env = RoboGymEnv(robot=env_name, max_episode_steps=max_episode_steps)


    print(env.observation_space.shape[0])
    
    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate)

    # agent.load_checkpoint()

    # Tesnorboard
    episode_identifier = f"Adam - lr: {learning_rate} HL: {hidden_size}"

    writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{episode_identifier}_temp={alpha}')

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

            if memory.can_sample(batch_size=batch_size):
                # Number of updates per step in environment
                for i in range(updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         batch_size,
                                                                                                         updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == max_episode_steps else float(not done)

            memory.store_transition(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        writer.add_scalar('score/live_train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))
        if i_episode % 10 == 0:
            agent.save_checkpoint(env_name=env_name)



    env.close()