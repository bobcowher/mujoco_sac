import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, AdamW
from sac_utils import *
from model import *
import time
from robot_environments import RoboGymEnv


class SAC(object):
    def __init__(self, joint_obs_size, action_space, gamma, tau, alpha, policy, target_update_interval,
                 automatic_entropy_tuning, hidden_size, learning_rate, device, env, entropy_scalar):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.env : RoboGymEnv = env

        self.policy_type = policy
        self.target_update_interval = target_update_interval

        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = device
        self.aet_warmup_episodes = 150
        self.aet_warmup_steps = self.aet_warmup_episodes * self.env.max_episode_steps 

        if self.automatic_entropy_tuning:
            # target_entropy ≈ −|A|
            self.target_entropy = -entropy_scalar * action_space.shape[0]
        
            # log α is the trainable parameter; start from log(α0)
            self.log_alpha = torch.tensor(np.log(alpha),
                                            requires_grad=True,
                                            device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=learning_rate * 0.5)


        self.critic = QNetwork(joint_obs_size=joint_obs_size, 
                               camera_obs_shape=(1, 80, 80),
                               num_actions=action_space.shape[0], 
                               hidden_dim=hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)
        
        self.critic_target = QNetwork(joint_obs_size=joint_obs_size, 
                               camera_obs_shape=(1, 80, 80),
                               num_actions=action_space.shape[0], 
                               hidden_dim=hidden_size).to(device=self.device)
 
        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(joint_obs_size=joint_obs_size, 
                                     camera_obs_shape=(1, 80, 80),
                                     num_actions=action_space.shape[0], 
                                     hidden_dim=hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        # Successful Initiation 
        print("Successfully initialized the agent")
        print("-" * 20)
        print("AET:                 ", self.automatic_entropy_tuning)
        
        if(self.automatic_entropy_tuning):
            print("AET Warmup Episodes: ", self.aet_warmup_episodes)
            print("AET Warmup Steps:    ", self.aet_warmup_steps)
            print("Target Entropy:      ", self.target_entropy)
        print("Alpha:               ", self.alpha)
        print("-" * 20)
        

    def select_action(self, state, evaluate=False, random=False):
        #state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        #state = state.to(self.device).unsqueeze(0)

        if random:
            action = self.env.action_space.sample()
            return action
        else:
            if evaluate is False:
                action, _, _, _ = self.policy.sample(state)
            else:
                _, _, action, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]


    def obs_to_tensor(self, obs):
        return {
            'camera': torch.from_numpy(obs['camera']).unsqueeze(0).to(self.device),
            'joint_pos': torch.from_numpy(obs['joint_pos']).unsqueeze(0).to(self.device),
            'joint_vel': torch.from_numpy(obs['joint_vel']).unsqueeze(0).to(self.device)
        }


    def test(self):
    
        episode_reward = 0
        episode_steps = 0
        done = False
        state, info = self.env.reset()

        while not done:
            
            action = self.select_action(self.obs_to_tensor(obs=state))  # Sample action from policy
            next_state, reward, done, _, _ = self.env.step(action)  # Step
            episode_steps += 1
            episode_reward += reward

            self.env.render()
            self.env.render(front_camera=True)
            print(f"Ground Distance: {self.env.get_robot_height()}. Distance to Goal: {self.env.get_distance_to_goal()} Reward: {reward} Action: {action} State: {state['joint_pos']}, {state['joint_vel']}")
            # print(f"QPos: {self.env.data.qpos}")


            # img = self.sim.render(width=128, height=128, camera_name="front_camera")
            #img = env._get_image_obs()


            time.sleep(0.005)
            # time.sleep(0.1)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)

            state = next_state

        print(f"Test run completed with score {episode_reward}")

    def train(self, episodes, memory, update_interval, batch_size, summary_writer, max_episode_steps, warmup):
        # Training Loop
        total_numsteps = 0
        updates = 0
        warmup_episode = True
        debug = False

        for i_episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, info = self.env.reset()

            if(os.path.exists('./debug')):
                debug = True
            else:
                debug = False

            if i_episode > warmup:
                warmup_episode = False

            while not done:

                action = self.select_action(self.obs_to_tensor(obs=state), random=warmup_episode)  # Sample action from policy

                if memory.can_sample(batch_size=batch_size) and not warmup_episode and episode_steps % update_interval == 0:
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, alpha = self.update_parameters(memory, batch_size, updates)

                    summary_writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    summary_writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    summary_writer.add_scalar('loss/policy', policy_loss, updates)
                    summary_writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

                #if i_episode % 20 == 0 and episode_steps < 5:
                #    print(f"Sampled action: {action}")
                
                next_state, reward, done, _, _ = self.env.step(action)  # Step
                
                if(debug):
                    # print(f"State: {state}")
                    # print(f"Next State: {next_state}")
                    print(f"Reward: {reward}")
                    # print(f"Done: {done}")
                    print(f"Robot Height: {self.env.get_robot_height()}")
                    # print(f": {done}")
                    show_observation_image(state['camera'])


                #print(f"Action: {action}")
                #print(f"Reward: {reward} - Joint Pos: {state['joint_pos']}")
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == max_episode_steps else float(not done)

                memory.store_transition(state, action, reward, next_state, mask)  # Append transition to memory

                state = next_state

            summary_writer.add_scalar('score/live_train', episode_reward, i_episode)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                        episode_steps,
                                                                                        round(episode_reward, 2)))
            if i_episode % 10 == 0:
                self.save_checkpoint()

    def update_parameters(self, memory, batch_size, updates, human=False):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=batch_size)
                # state_batch = state_batch.to(self.device)
        # next_state_batch = next_state_batch.to(self.device)
        # action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        #print(f"qf1 {qf1.shape}")
        #print(f"next_q_values {next_q_value.shape}")
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _, log_std = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        #print(f"Mean log_std: {log_std.mean().item():.3f}")
        #print(f"Action sample (mean/std): {pi.mean().item():.3f} / {pi.std().item():.3f}")

        if human == True:
            policy_loss = F.mse_loss(pi, action_batch)
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning and updates > self.aet_warmup_steps:
            alpha_loss = (self.log_alpha.exp() *
                         (-log_pi - self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()   # scalar for later

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), self.alpha 


    # Save model parameters
    def save_checkpoint(self, suffix=""):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        print('Saving models')
        self.policy.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()


    # Load model parameters
    def load_checkpoint(self, evaluate=False):

        try:
            print('Loading models...')
            self.policy.load_checkpoint()
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
            print('Successfully loaded models')
        except:
            print("Unable to load models. Starting from scratch")

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()



