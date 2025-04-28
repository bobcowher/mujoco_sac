import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from sac_utils import *
from model import *
import time


class SAC(object):
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy, target_update_interval,
                 automatic_entropy_tuning, hidden_size, learning_rate, alpha_decay, min_alpha=0.05):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = AdamW(self.critic.parameters(), lr=0.001)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha


        # else:
        #     self.alpha = 0
        #     self.automatic_entropy_tuning = False
        #     self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
        #     self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state, evaluate=False):
        # state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state = torch.Tensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]


    def test(self, env):
    
        episode_reward = 0
        episode_steps = 0
        done = False
        state, info = env.reset()

        while not done:
            
            action = self.select_action(state)  # Sample action from policy

            next_state, reward, done, _, _ = env.step(action)  # Step
            episode_steps += 1
            episode_reward += reward

            env.render()
            env.render(front_camera=True)

            # img = self.sim.render(width=128, height=128, camera_name="front_camera")
            #img = env._get_image_obs()


            time.sleep(0.01)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)

            state = next_state

        print(f"Test run completed with score {episode_reward}")

    def train(self, episodes, env, memory, updates_per_step, batch_size, summary_writer, max_episode_steps):
        # Training Loop
        total_numsteps = 0
        updates = 0

        for i_episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, info = env.reset()

            while not done:
                
                action = self.select_action(state)  # Sample action from policy

                if memory.can_sample(batch_size=batch_size):
                    # Number of updates per step in environment
                    for i in range(updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, alpha = self.update_parameters(memory, batch_size, updates)

                        summary_writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        summary_writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        summary_writer.add_scalar('loss/policy', policy_loss, updates)
                        summary_writer.add_scalar('entropy_temprature/alpha', alpha, updates)
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

            summary_writer.add_scalar('score/live_train', episode_reward, i_episode)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                        episode_steps,
                                                                                        round(episode_reward, 2)))
            if i_episode % 10 == 0:
                self.save_checkpoint()

    def update_parameters(self, memory, batch_size, updates, human=False):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        if human == True:
            policy_loss = F.mse_loss(pi, action_batch)
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update alpha. 
        if(self.alpha > self.min_alpha and updates % 200 == 0):
            #self.alpha = self.alpha * (1 - (self.alpha_decay * (10 * self.alpha)))
            self.alpha = self.alpha * (1 - self.alpha_decay)
        
        if(updates % int(5e5) == 0):
            self.alpha_decay = self.alpha_decay * 0.75

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), self.alpha 


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
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



