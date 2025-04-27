import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir='checkpoints', name='q_network'):

        print(f"Num inputs: {num_inputs}")
        print(f"Num actions: {num_actions}")

        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)  # Third convolutional layer
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)

        # Pool Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim * 2)
        self.linear5 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.apply(weights_init_)

    def calculate_conv_output(self, observation_shape):
        x = torch.zeros(1, *observation_shape)
        x = self.pool(F.relu(self.conv1(x)))  # Pooling after first conv layer
        x = F.relu(self.conv2(x))             # No pooling after second to control size
        x = self.pool(F.relu(self.conv3(x)))  # Pooling after third conv layer
        x = F.relu(self.conv4(x))             # No pooling after second to control size

        return x.view(-1).shape[0]

    def forward(self, obs, action):

        camera_obs = obs['camera']
        joint_pos = obs['joint_pos']
        joint_vel = obs['joint_vel']        

        x = camera_obs.permute(0, 3, 1, 2)  # (batch, 3, H, W) for CNN
        x = x.float() / 255.0

        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x))) 
        x = F.relu(self.conv4(x)) 
        x = x.view(x.size(0), -1)  

        x = torch.cat([x, joint_pos, joint_vel, action])

        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(x))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, checkpoint_dir='checkpoints', name='policy_network'):
        
        super(GaussianPolicy, self).__init__()
 
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)  # Third convolutional layer
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)

        # Pooling layer.  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FC Layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, obs):

        camera_obs = obs['camera']
        joint_pos = obs['joint_pos']
        joint_vel = obs['joint_vel']        

        x = camera_obs.permute(0, 3, 1, 2)  # (batch, 3, H, W) for CNN
        x = x.float() / 255.0

        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x))) 
        x = F.relu(self.conv4(x)) 
        x = x.view(x.size(0), -1)  

        x = torch.cat([x, joint_pos, joint_vel])

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
