import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional

from torch.nn.modules.dropout import Dropout
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((max_size, state_dim))
        self.action = torch.zeros((max_size, action_dim))
        self.next_state = torch.zeros((max_size, state_dim))
        self.reward = torch.zeros((max_size, 1))
        self.not_done = torch.zeros((max_size, 1))


        self.next_state_samples = torch.zeros((7,max_size, state_dim))

        self.mobile = 0

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        # print(self.size)

    def add_batch(self, batch):
        if batch is None:
            return 
        # s = batch["obss"].pin_memory()
        # ns = batch["next_obss"].pin_memory()
        # a = batch["actions"].pin_memory()
        # r = batch["rewards"].pin_memory()
        # d = batch["terminals"].pin_memory()

        s = batch["obss"]
        ns = batch["next_obss"]
        a = batch["actions"]
        r = batch["rewards"]
        d = batch["terminals"]



        # print(d)

        # print(s.shape)
        # print(s[:  min(self.ptr + len(s), self.max_size) - self.ptr].shape)
        # print(self.state.shape)
        # print(self.state[self.ptr:min(self.ptr + len(s), self.max_size)].shape )

        
        self.state[self.ptr:min(self.ptr + len(s), self.max_size)] = s[:  min(self.ptr + len(s), self.max_size) - self.ptr]
        self.action[self.ptr:min(self.ptr + len(s), self.max_size)] = a[:  min(self.ptr + len(s), self.max_size) - self.ptr]
        self.next_state[self.ptr:min(self.ptr + len(s), self.max_size)] = ns[:  min(self.ptr + len(s), self.max_size) - self.ptr]
        self.reward[self.ptr:min(self.ptr + len(s), self.max_size)] = r[:  min(self.ptr + len(s), self.max_size) - self.ptr]
        self.not_done[self.ptr:min(self.ptr + len(s), self.max_size)] = 1. - d[:  min(self.ptr + len(s), self.max_size) - self.ptr]





        used = min(self.ptr + len(s), self.max_size) - self.ptr

        self.ptr = (min(self.ptr + len(s), self.max_size)) % self.max_size
        
        self.size = min(self.size + used, self.max_size)

        if self.ptr == 0:
            self.state[0:len(s) - used] = s[used:]
            self.action[0:len(s) - used] = a[used:]
            self.next_state[0:len(s) - used] = ns[used:]
            self.reward[0:len(s) - used] = r[used:]
            self.not_done[0:len(s) - used] = 1. - d[used:]


            self.ptr = len(s) - used
        
    def add_batch_sep(self, s, a, ns, r, d):
        # s = batch["obss"]
        # ns = batch["next_obss"]
        # a = batch["actions"]
        # r = batch["rewards"]
        # d = batch["terminals"]

        
        self.state[self.ptr:min(self.ptr + len(s), self.max_size)] = s[:  min(self.ptr + len(s), self.max_size) - self.ptr]
        self.action[self.ptr:min(self.ptr + len(s), self.max_size)] = a[:  min(self.ptr + len(s), self.max_size) - self.ptr]
        self.next_state[self.ptr:min(self.ptr + len(s), self.max_size)] = ns[:  min(self.ptr + len(s), self.max_size) - self.ptr]
        self.reward[self.ptr:min(self.ptr + len(s), self.max_size)] = r[:  min(self.ptr + len(s), self.max_size) - self.ptr]
        self.not_done[self.ptr:min(self.ptr + len(s), self.max_size)] = 1. - d[:  min(self.ptr + len(s), self.max_size) - self.ptr]

        used = min(self.ptr + len(s), self.max_size) - self.ptr

        self.ptr = (min(self.ptr + len(s), self.max_size)) % self.max_size
        
        self.size = min(self.size + used, self.max_size)
        

        if self.ptr == 0:
            self.state[0:len(s) - used] = s[used:]
            self.action[0:len(s) - used] = a[used:]
            self.next_state[0:len(s) - used] = ns[used:]
            self.reward[0:len(s) - used] = r[used:]
            self.not_done[0:len(s) - used] = 1. - d[used:]

            self.ptr = len(s) - used

        


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.mobile == 1:
            return (
            torch.FloatTensor(self.state[ind]).to(self.device,non_blocking=True),
            torch.FloatTensor(self.action[ind]).to(self.device,non_blocking=True),
            torch.FloatTensor(self.next_state[ind]).to(self.device,non_blocking=True),
            torch.FloatTensor(self.reward[ind]).to(self.device,non_blocking=True),
            torch.FloatTensor(self.not_done[ind]).to(self.device,non_blocking=True),
            torch.FloatTensor(self.next_state_samples[:,ind]).to(self.device,non_blocking=True),
        )

        else:

            return (
                torch.FloatTensor(self.state[ind]).to(self.device,non_blocking=True),
                torch.FloatTensor(self.action[ind]).to(self.device,non_blocking=True),
                torch.FloatTensor(self.next_state[ind]).to(self.device,non_blocking=True),
                torch.FloatTensor(self.reward[ind]).to(self.device,non_blocking=True),
                torch.FloatTensor(self.not_done[ind]).to(self.device,non_blocking=True),
            )
    
    def sample_all(self, cuda = True):
        ind = np.arange(0, self.size)
        if cuda:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device, non_blocking=True),
                torch.FloatTensor(self.action[ind]).to(self.device, non_blocking=True),
                torch.FloatTensor(self.next_state[ind]).to(self.device, non_blocking=True),
                torch.FloatTensor(self.reward[ind]).to(self.device, non_blocking=True),
                torch.FloatTensor(self.not_done[ind]).to(self.device, non_blocking=True),
            )
        else:
            return (
                torch.FloatTensor(self.state[ind]),
                torch.FloatTensor(self.action[ind]),
                torch.FloatTensor(self.next_state[ind]),
                torch.FloatTensor(self.reward[ind]),
                torch.FloatTensor(self.not_done[ind]),
            )

       


        
    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        # self.state = torch.FloatTensor(self.state).pin_memory()
        self.state = torch.FloatTensor(self.state)

        self.action = dataset['actions']
        # self.action = torch.FloatTensor(self.action).pin_memory()
        # self.action = torch.FloatTensor(self.action)

        self.next_state = dataset['next_observations']
        # self.next_state = torch.FloatTensor(self.next_state).pin_memory()
        self.next_state = torch.FloatTensor(self.next_state)

        self.reward = dataset['rewards'].reshape(-1,1)
        # self.reward = torch.FloatTensor(self.reward).pin_memory()
        self.reward = torch.FloatTensor(self.reward)

        self.not_done = 1. - dataset['terminals'].reshape(-1,1)
        # self.not_done = torch.FloatTensor(self.not_done).pin_memory()
        self.not_done = torch.FloatTensor(self.not_done)
        self.size = self.state.shape[0]


# class ReplayBuffer(object):
#     def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0

#         self.state = torch.zeros((max_size, state_dim)).to(device)
#         self.action = torch.zeros((max_size, action_dim)).to(device)
#         self.next_state = torch.zeros((max_size, state_dim)).to(device)
#         self.reward = torch.zeros((max_size, 1)).to(device)
#         self.not_done = torch.zeros((max_size, 1)).to(device)

#         self.device = device

#     def add(self, state, action, next_state, reward, done):


#         state = torch.FloatTensor(state).to(self.device)

#         action = torch.FloatTensor(action).to(self.device)
#         next_state = torch.FloatTensor(next_state).to(self.device)
#         # print(reward)
#         reward = torch.FloatTensor([reward]).to(self.device)
#         done = torch.FloatTensor([done]).to(self.device)
        
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.next_state[self.ptr] = next_state
#         self.reward[self.ptr] = reward
#         self.not_done[self.ptr] = 1. - done

#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#         # print(self.size)

#     def add_batch(self, batch):
#         s = batch["obss"]
#         ns = batch["next_obss"]
#         a = batch["actions"]
#         r = batch["rewards"]
#         d = batch["terminals"]

        
#         self.state[self.ptr:min(self.ptr + len(s), self.max_size)] = s[:  min(self.ptr + len(s), self.max_size) - self.ptr]
#         self.action[self.ptr:min(self.ptr + len(s), self.max_size)] = a[:  min(self.ptr + len(s), self.max_size) - self.ptr]
#         self.next_state[self.ptr:min(self.ptr + len(s), self.max_size)] = ns[:  min(self.ptr + len(s), self.max_size) - self.ptr]
#         self.reward[self.ptr:min(self.ptr + len(s), self.max_size)] = r[:  min(self.ptr + len(s), self.max_size) - self.ptr]
#         self.not_done[self.ptr:min(self.ptr + len(s), self.max_size)] = 1. - d[:  min(self.ptr + len(s), self.max_size) - self.ptr]

#         used = min(self.ptr + len(s), self.max_size) - self.ptr

#         self.ptr = (min(self.ptr + len(s), self.max_size)) % self.max_size
        
#         self.size = min(self.size + used, self.max_size)

#         if self.ptr == 0:
#             self.state[0:len(s) - used] = s[used:]
#             self.action[0:len(s) - used] = a[used:]
#             self.next_state[0:len(s) - used] = ns[used:]
#             self.reward[0:len(s) - used] = r[used:]
#             self.not_done[0:len(s) - used] = 1. - d[used:]

#             self.ptr = len(s) - used
        
#     def add_batch_sep(self, s, a, ns, r, d):
#         # s = batch["obss"]
#         # ns = batch["next_obss"]
#         # a = batch["actions"]
#         # r = batch["rewards"]
#         # d = batch["terminals"]

#         s = torch.FloatTensor(s).to(self.device)

#         a = torch.FloatTensor(a).to(self.device)
#         ns = torch.FloatTensor(ns).to(self.device)
#         r = torch.FloatTensor(r).to(self.device)
#         d = torch.FloatTensor(d).to(self.device)
        
        

        
#         self.state[self.ptr:min(self.ptr + len(s), self.max_size)] = s[:  min(self.ptr + len(s), self.max_size) - self.ptr]
#         self.action[self.ptr:min(self.ptr + len(s), self.max_size)] = a[:  min(self.ptr + len(s), self.max_size) - self.ptr]
#         self.next_state[self.ptr:min(self.ptr + len(s), self.max_size)] = ns[:  min(self.ptr + len(s), self.max_size) - self.ptr]
#         self.reward[self.ptr:min(self.ptr + len(s), self.max_size)] = r[:  min(self.ptr + len(s), self.max_size) - self.ptr]
#         self.not_done[self.ptr:min(self.ptr + len(s), self.max_size)] = 1. - d[:  min(self.ptr + len(s), self.max_size) - self.ptr]

#         used = min(self.ptr + len(s), self.max_size) - self.ptr

#         self.ptr = (min(self.ptr + len(s), self.max_size)) % self.max_size
        
#         self.size = min(self.size + used, self.max_size)
        

#         if self.ptr == 0:
#             self.state[0:len(s) - used] = s[used:]
#             self.action[0:len(s) - used] = a[used:]
#             self.next_state[0:len(s) - used] = ns[used:]
#             self.reward[0:len(s) - used] = r[used:]
#             self.not_done[0:len(s) - used] = 1. - d[used:]

#             self.ptr = len(s) - used

        


#     def sample(self, batch_size):
#         # ind = torch.randint(0, self.size, size=batch_size)
#         ind = torch.randint(0, self.size, size=(batch_size,))


#         return (
#             self.state[ind],
#             self.action[ind],
#             self.next_state[ind],
#             self.reward[ind],
#             self.not_done[ind])
    
#     def convert_D4RL(self, dataset):
#         self.state = dataset['observations']
#         self.action = dataset['actions']
#         self.next_state = dataset['next_observations']
#         self.reward = dataset['rewards'].reshape(-1,1)
#         self.not_done = 1. - dataset['terminals'].reshape(-1,1)
#         self.size = self.state.shape[0]
        

class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        n_layers,
        activations: Callable = nn.ReLU,
        activate_final: int = False,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()

        self.affines = []
        self.affines.append(nn.Linear(in_dim, hidden_dim))
        for i in range(n_layers-2):
            self.affines.append(nn.Linear(hidden_dim, hidden_dim))
        self.affines.append(nn.Linear(hidden_dim, out_dim))
        self.affines = nn.ModuleList(self.affines)

        self.activations = activations()
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)
            self.norm_layer = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        for i in range(len(self.affines)):
            x = self.affines[i](x)
            if i != len(self.affines)-1 or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = self.dropout(x)
                    # x = self.norm_layer(x)
        return x

def identity(x):
    return x

def fanin_init(tensor, scale=1):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = scale / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def orthogonal_init(tensor, gain=0.01):
    torch.nn.init.orthogonal_(tensor, gain=gain)

class ParallelizedLayerMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_init=fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            final_init_scale=None,
            dropout_rate=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity
        
        self.layer_norm = layer_norm

        self.fcs = []

        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            for j in self.elites:
                hidden_init(fc.W[j], w_scale)
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                orthogonal_init(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)

        state_dim = inputs[0].shape[-1]
        
        dim=len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)
        
        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            # add dropout
            if self.dropout_rate:
                h = self.dropout(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        return output
    
    def sample(self, *inputs):
        preds = self.forward(*inputs)

        sample_idxs = np.random.choice(self.ensemble_size, 2, replace=False)
        preds_sample = preds[sample_idxs]
        
        return torch.min(preds_sample, dim=0)[0], sample_idxs
    

import os
import numpy as np
import torch
import torch.nn as nn

import numpy as np
import os.path as path
import torch


class StandardScaler(object):
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        
        self.mu = torch.mean(data, axis=0, keepdims=True)
        self.std = torch.mean(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

        print('scaler is not fit')

        self.mu = torch.zeros(self.mu.shape).to('cuda')

        self.std = torch.ones(self.std.shape).to('cuda')

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std
        # return data

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu
        # return data
    
    def save_scaler(self, save_path):
        mu_path = path.join(save_path, "mu.npy")
        std_path = path.join(save_path, "std.npy")

        try:
            np.save(mu_path, self.mu)
            np.save(std_path, self.std)
        except:
            np.save(mu_path, self.mu.cpu().numpy())
            np.save(std_path, self.std.cpu().numpy())
        
    def load_scaler(self, load_path):
        mu_path = path.join(load_path, "mu.npy")
        std_path = path.join(load_path, "std.npy")
        self.mu = np.load(mu_path,allow_pickle=True)
        self.std = np.load(std_path,allow_pickle=True)

        print(self.mu)
        print(type(self.mu), self.mu.dtype)

        self.mu = self.mu.astype(float)
        self.std = self.std.astype(float)

        print(self.mu)
        print(type(self.mu), self.mu.dtype)

        self.mu = torch.FloatTensor(self.mu).to('cuda')
        self.std = torch.FloatTensor(self.std).to('cuda')


        self.mu = 0
        self.std = 1
    def transform_tensor(self, data: torch.Tensor):
        device = data.device
        data = self.transform(data)
        # data = torch.tensor(data, device=device)
        return data