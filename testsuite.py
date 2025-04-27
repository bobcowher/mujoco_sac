import unittest
import torch
import numpy as np
from robot_environments import RoboGymEnv
from model import QNetwork, GaussianPolicy  # replace 'your_model_file' with your filename

class TestModelEnvIntegration(unittest.TestCase):

    def setUp(self):
        self.env = RoboGymEnv(robot="boston_dynamics_spot", max_episode_steps=100)

        # Get an initial observation
        obs, info = self.env.reset()

        # Convert single env observation into batch format for testing
        self.obs = {
            'camera': torch.from_numpy(obs['camera']).unsqueeze(0),  # (1, H, W, 3)
            'joint_pos': torch.from_numpy(obs['joint_pos']).unsqueeze(0),  # (1, nq)
            'joint_vel': torch.from_numpy(obs['joint_vel']).unsqueeze(0)   # (1, nv)
        }

        # Setup dummy action input
        action_dim = self.env.action_space.shape[0]
        self.dummy_action = torch.randn(1, action_dim)

        # Setup model
        self.hidden_dim = 256
        self.q_net = QNetwork(num_inputs=self.obs['joint_pos'].shape[1] + self.obs['joint_vel'].shape[1], 
                              num_actions=action_dim, 
                              hidden_dim=self.hidden_dim)
        self.policy = GaussianPolicy(num_inputs=self.obs['joint_pos'].shape[1] + self.obs['joint_vel'].shape[1], 
                                     num_actions=action_dim, 
                                     hidden_dim=self.hidden_dim)

    def tearDown(self):
        self.env.close()

    def test_q_network_forward(self):
        try:
            q1, q2 = self.q_net(self.obs, self.dummy_action)
            self.assertEqual(q1.shape, (1, 1))
            self.assertEqual(q2.shape, (1, 1))
        except Exception as e:
            self.fail(f"QNetwork forward pass failed with error: {e}")

    def test_policy_network_forward(self):
        try:
            mean, log_std = self.policy(self.obs)
            self.assertEqual(mean.shape[0], 1)
            self.assertEqual(log_std.shape[0], 1)
        except Exception as e:
            self.fail(f"GaussianPolicy forward pass failed with error: {e}")

    def test_policy_sampling(self):
        try:
            action, log_prob, mean = self.policy.sample(self.obs)
            self.assertEqual(action.shape, (1, self.env.action_space.shape[0]))
            self.assertEqual(log_prob.shape, (1, 1))
        except Exception as e:
            self.fail(f"GaussianPolicy sampling failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
