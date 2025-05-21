import math
import torch
import matplotlib.pyplot as plt

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def show_observation_image(image_tensor_or_array, title="Observation"):
    """
    Displays an image from an observation. Accepts a NumPy array or torch.Tensor.
    """
    import numpy as np
    import torch

    # If it's a tensor, move to CPU and convert to NumPy
    if isinstance(image_tensor_or_array, torch.Tensor):
        img = image_tensor_or_array.detach().cpu().numpy()
    else:
        img = image_tensor_or_array

    # If shape is (1, H, W) or (H, W), squeeze out channel
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    elif img.ndim == 4 and img.shape[1] == 1:
        img = img.squeeze(1)[0]  # batch of 1, remove batch and channel

    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    # plt.show()
    plt.pause(0.001)
    plt.clf()

