import os
import torch
import numpy as np

from tqdm import tqdm

def save_args(args, workspace):
    args_path = os.path.join(workspace, 'args.txt')
    with open(args_path, 'w') as f:
        f.write(str(args).replace(', ', ',\n'))

def select_activation_fn(act_name):
    if act_name == 'relu':
        return torch.nn.ReLU()
    
    if act_name == 'tanh':
        return torch.nn.Tanh()
    
    if act_name == 'lrelu':
        return torch.nn.LeakyReLU(0.2)
    
    raise ValueError('Activation function {} not found.'.format(act_name))