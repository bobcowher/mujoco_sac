import torch

class ReplayBuffer:
    def __init__(self, max_size, camera_shape, joint_pos_dim, joint_vel_dim, n_actions, device='cpu'):
        self.device = device
        self.mem_size = max_size
        self.mem_ctr = 0

        self.state_memory = {
            'camera': torch.zeros((self.mem_size, *camera_shape), dtype=torch.uint8, device=self.device),
            'joint_pos': torch.zeros((self.mem_size, joint_pos_dim), dtype=torch.float32, device=self.device),
            'joint_vel': torch.zeros((self.mem_size, joint_vel_dim), dtype=torch.float32, device=self.device),
        }
        self.new_state_memory = {
            'camera': torch.zeros((self.mem_size, *camera_shape), dtype=torch.uint8, device=self.device),
            'joint_pos': torch.zeros((self.mem_size, joint_pos_dim), dtype=torch.float32, device=self.device),
            'joint_vel': torch.zeros((self.mem_size, joint_vel_dim), dtype=torch.float32, device=self.device),
        }
        self.action_memory = torch.zeros((self.mem_size, n_actions), dtype=torch.float32, device=self.device)
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float32, device=self.device)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool, device=self.device)

    def can_sample(self, batch_size):
        return self.mem_ctr > (batch_size * 5)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_ctr % self.mem_size

        self.state_memory['camera'][index] = torch.from_numpy(state['camera']).to(self.device)
        self.state_memory['joint_pos'][index] = torch.from_numpy(state['joint_pos']).to(self.device)
        self.state_memory['joint_vel'][index] = torch.from_numpy(state['joint_vel']).to(self.device)

        self.new_state_memory['camera'][index] = torch.from_numpy(next_state['camera']).to(self.device)
        self.new_state_memory['joint_pos'][index] = torch.from_numpy(next_state['joint_pos']).to(self.device)
        self.new_state_memory['joint_vel'][index] = torch.from_numpy(next_state['joint_vel']).to(self.device)

        self.action_memory[index] = torch.from_numpy(action).to(self.device)
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def sample_buffer(self, batch_size, augment_data=False, noise_ratio=0.1):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = torch.randint(0, max_mem, (batch_size,), device=self.device)

        states = {
            'camera': self.state_memory['camera'][batch],
            'joint_pos': self.state_memory['joint_pos'][batch],
            'joint_vel': self.state_memory['joint_vel'][batch],
        }
        new_states = {
            'camera': self.new_state_memory['camera'][batch],
            'joint_pos': self.new_state_memory['joint_pos'][batch],
            'joint_vel': self.new_state_memory['joint_vel'][batch],
        }
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        if augment_data:
            states = self._augment_states(states, noise_ratio)
            actions = actions + noise_ratio * torch.randn_like(actions)
            rewards = rewards + noise_ratio * torch.randn_like(rewards)

        return states, actions, rewards, new_states, dones

    def _augment_states(self, states, noise_ratio):
        # Only augment joint_pos and joint_vel; don't touch images
        states['joint_pos'] += noise_ratio * torch.randn_like(states['joint_pos'])
        states['joint_vel'] += noise_ratio * torch.randn_like(states['joint_vel'])
        return states
