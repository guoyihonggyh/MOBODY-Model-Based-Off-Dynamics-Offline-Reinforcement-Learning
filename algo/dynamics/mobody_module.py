import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
# from offlinerlkit.nets import EnsembleLinear


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    # print("x",x.shape)
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

class GradReverse(torch.autograd.Function):
    """Extension of grad reverse layer."""
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()
        return grad_output, None

    def grad_reverse(x):
        return GradReverse.apply(x)


class MOBODYModule(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str = "cpu",
        reward_relu: bool = False,
        config = None,
    ) -> None:
        super().__init__()

        self.training = True
        self.config = config

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.device = torch.device(device)

        self.activation = activation()

        self.reward_relu = reward_relu
        
        self.encode_trg_diff = 0

        # assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        # hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        # if weight_decays is None:
        #     weight_decays = [0.0] * (len(hidden_dims) + 1)
        # for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
        #     module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        # self.backbones = nn.ModuleList(module_list)

        self.obs_dim = obs_dim
        # print("obs_dim",obs_dim)
        weight_decay = 5e-5

        latent_dim = 16
        # state encoder:
        self.zs1 = EnsembleLinear(obs_dim, hidden_dims, num_ensemble, weight_decay)
        self.zs2 = EnsembleLinear(hidden_dims, hidden_dims, num_ensemble, weight_decay)
        self.zs3 = EnsembleLinear(hidden_dims, 2 *  latent_dim, num_ensemble, weight_decay)
        module_list += [self.zs1,self.zs2,self.zs3]

        # self.zs1_trg = EnsembleLinear(obs_dim, hidden_dims, num_ensemble, weight_decay)
        # self.zs2_trg = EnsembleLinear(hidden_dims, hidden_dims, num_ensemble, weight_decay)
        # self.zs3_trg = EnsembleLinear(hidden_dims, hidden_dims, num_ensemble, weight_decay)
        # module_list += [self.zs1_trg,self.zs2_trg,self.zs3_trg]
        
        
        # action encoder:
        
        self.za_src1 = EnsembleLinear(latent_dim + action_dim, 32, num_ensemble, weight_decay)
        self.za_src2 = EnsembleLinear(32, 2 * latent_dim, num_ensemble, weight_decay)
        # self.zsa3 = EnsembleLinear(hidden_dims, latent_dim, num_ensemble, weight_decay)
        
        if self.config['mopo']:
            self.za_src1 = EnsembleLinear(self.obs_dim + action_dim, 256, num_ensemble, weight_decay)
            self.za_src2 = EnsembleLinear(256, 256, num_ensemble, weight_decay)
            self.za_src3 = EnsembleLinear(256, obs_dim, num_ensemble, weight_decay)
            module_list += [self.za_src1,self.za_src2,self.za_src3]
        else:   
            module_list += [self.za_src1,self.za_src2]
        # action decoder:
        self.za_de_src1 = EnsembleLinear(latent_dim, 8, num_ensemble, weight_decay)
        self.za_de_src2 = EnsembleLinear(8, action_dim, num_ensemble, weight_decay)
        # self.zsa3 = EnsembleLinear(hidden_dims, latent_dim, num_ensemble, weight_decay)
        module_list += [self.za_de_src1,self.za_de_src2]
        # print(list(self.za_de_src2.named_parameters()))

        # target action encoder:
        self.za_trg1 = EnsembleLinear(latent_dim + action_dim, 32, num_ensemble, weight_decay)
        self.za_trg2 = EnsembleLinear(32, 2 * latent_dim, num_ensemble, weight_decay)
        # self.za_trg3 = EnsembleLinear(hidden_dims, latent_dim, num_ensemble, weight_decay)

        if self.config['mopo']:
            self.za_trg1 = EnsembleLinear(self.obs_dim + action_dim, 256, num_ensemble, weight_decay)
            self.za_trg2 = EnsembleLinear(256, 256, num_ensemble, weight_decay)
            self.za_trg3 = EnsembleLinear(256, obs_dim, num_ensemble, weight_decay)
            module_list += [self.za_trg1,self.za_trg2,self.za_trg3]
        else:
            module_list += [self.za_trg1,self.za_trg2]
        # target action decoder:
        self.za_de_trg1 = EnsembleLinear(latent_dim, 8, num_ensemble, weight_decay)
        self.za_de_trg2 = EnsembleLinear(8, action_dim, num_ensemble, weight_decay)
        # self.zsa3 = EnsembleLinear(hidden_dims, latent_dim, num_ensemble, weight_decay)
        module_list += [self.za_de_trg1,self.za_de_trg2]

        # print(list(self.za_de_trg2.named_parameters()))

        
        # transition/ decoder:
        self._with_reward = 0
        self.transition1 = EnsembleLinear( latent_dim, hidden_dims, num_ensemble, weight_decay)
        self.transition2 = EnsembleLinear(hidden_dims, hidden_dims, num_ensemble, weight_decay)
        self.transition3 = EnsembleLinear(hidden_dims, 1 * (obs_dim + self._with_reward), num_ensemble, weight_decay)
        module_list += [self.transition1,self.transition2,self.transition3]

        # # transform:
        # self.transform1 = EnsembleLinear(latent_dim, latent_dim, num_ensemble, weight_decay)
        # self.transform2 = EnsembleLinear(latent_dim, latent_dim, num_ensemble, weight_decay)
        # self.transform3 = EnsembleLinear(latent_dim, latent_dim * 2, num_ensemble, weight_decay)
        # module_list += [self.transform1,self.transform2,self.transform3]
        
        # # transform:
        # self.src_transform1 = EnsembleLinear(latent_dim, latent_dim, num_ensemble, weight_decay)
        # self.src_transform2 = EnsembleLinear(latent_dim, latent_dim, num_ensemble, weight_decay)
        # self.src_transform3 = EnsembleLinear(latent_dim, latent_dim, num_ensemble, weight_decay)
        # module_list += [self.src_transform1,self.src_transform2,self.src_transform3]

        
        # self.domain_classfier1 = EnsembleLinear(latent_dim, hidden_dims, num_ensemble, weight_decay)
        # self.domain_classfier2 = EnsembleLinear(hidden_dims, hidden_dims, num_ensemble, weight_decay)
        # self.domain_classfier3 = EnsembleLinear(hidden_dims, 2, num_ensemble, weight_decay)
        # module_list += [self.domain_classfier1,self.domain_classfier2,self.domain_classfier3]

        # self.domain_classifier_list = [self.domain_classfier1,self.domain_classfier2,self.domain_classfier3]



        if self.config['latent_reward']:
            self.reward_model1 = EnsembleLinear(2*latent_dim + latent_dim, hidden_dims, num_ensemble, weight_decay)
        else:
            self.reward_model1 = EnsembleLinear(2*obs_dim + action_dim, hidden_dims, num_ensemble, weight_decay)
        self.reward_model2 = EnsembleLinear(hidden_dims, hidden_dims, num_ensemble, weight_decay)
        self.reward_model3 = EnsembleLinear(hidden_dims, 2, num_ensemble, weight_decay)
        module_list += [self.reward_model1,self.reward_model2,self.reward_model3]
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        
        
        self.module_list = module_list

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=False)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * -10, requires_grad=False)
        )

        self.register_parameter(
            "max_logvar_latent",
            nn.Parameter(torch.ones(latent_dim) * 20, requires_grad=False)
        )
        self.register_parameter(
            "min_logvar_latent",
            nn.Parameter(torch.ones(latent_dim) * -20, requires_grad=False)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self.device)


    def encode_state(self, state):
        if self.config['mopo']:
            return state, state, state
        zs = self.activation(self.zs1(state))
        zs = self.activation(self.zs2(zs))
        zs = self.zs3(zs)
        mu, logvar = torch.chunk(zs, 2, dim=-1)
        zs = self.reparameterize(mu, logvar)
        return zs, mu, logvar

    # def encode_statse_trg(self, state):
    #     zs = self.activation(self.zs1_trg(state))
    #     zs = self.activation(self.zs2_trg(zs))
    #     zs = AvgL1Norm(self.zs3_trg(zs))
    #     return zs
    def reward_relu_func(self, output):
        # print(output.shape)
        output[:,:,self.obs_dim] = F.sigmoid(output[:,:,self.obs_dim])
        return output

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)  # 计算标准差 σ
            eps = torch.randn_like(std)    # 采样噪声
            return mu + eps * std          # 
        else:
            return mu         # 采样 z
        
    def encode_src_action(self, s, a, reparam = True):
        if len(s.shape) == 3 and len(a.shape) == 2:
            a = a.unsqueeze(0).repeat(7,1,1)
        sa = torch.cat([s, a], dim=-1)
        za = self.activation(self.za_src1(sa))
        za = self.za_src2(za)
        if self.config['mopo']:
            za = self.activation(za)
            za = self.za_src3(za)
            return za
        mu, logvar = torch.chunk(za, 2, dim=-1)
        return mu
        
    def encode_trg_action(self, s, a, reparam = True):
        if len(s.shape) == 3 and len(a.shape) == 2:
            a = a.unsqueeze(0).repeat(7,1,1)

        sa = torch.cat([s, a], dim=-1)

        if self.config['mopo']:
            mu = self.encode_src_action(s, a, reparam)
            return mu
        za = self.activation(self.za_trg1(sa))
        za = self.za_trg2(za)

        mu, logvar = torch.chunk(za, 2, dim=-1)
        return mu
    
    def decode_src_action(self, z):
        # print(z[0])
        z = self.activation(self.za_de_src1(z))
        
        z = self.za_de_src2(z)
        # print(z[0].mean())
        return z
    def decode_trg_action(self, z):
        
        za1= self.activation(self.za_de_src1(z))
        za = self.za_de_src2(za1)

        return za

    def encode_transition(self, z):
        if self.config['mopo'] :
            return z
        z = self.activation(self.transition1(z))
        z = self.activation(self.transition2(z))
        z = self.transition3(z)
        return z
    
    def encode_reward(self, s, a, next_s, ):
        sas = torch.cat([s, a, next_s], dim=-1)
        sas_reward = self.activation(self.reward_model1(sas))
        sas_reward = self.activation(self.reward_model2(sas_reward))
        sas_reward = self.reward_model3(sas_reward)
        mu, logvar = torch.chunk(sas_reward, 2, dim=-1)
        logvar = soft_clamp(logvar, -10, 0.5)
        return mu, logvar
    
    # def encode_reward_latent(self, s, a, next_s, latent = True):
    #     sas = torch.cat([s, a, next_s], dim=-1)
    #     sas_reward = self.activation(self.reward_model1(sas))
    #     sas_reward = self.activation(self.reward_model2(sas_reward))
    #     sas_reward = self.reward_model3(sas_reward)
    #     mu, logvar = torch.chunk(sas_reward, 2, dim=-1)
    #     logvar = soft_clamp(logvar, -10, 0.5)
    #     return mu, logvar


    
    def forward_src(self, state, action):
        zs, zs_mu, zs_logvar = self.encode_state(state)
        za = self.encode_src_action(zs, action, False)   
        z_ns = zs + za
        output = self.encode_transition(z_ns)

        return output, zs_mu, zs_logvar

    def forward_trg(self, state, action):
        # with torch.no_grad():
        zs, zs_mu, zs_logvar = self.encode_state(state)
        za = self.encode_trg_action(zs, action, False)   
        z_ns = zs + za
        output = self.encode_transition(z_ns)

        return output, zs_mu, zs_logvar
    
    def encoder_decoder(self, state):
        zs, zs_mu, zs_logvar = self.encode_state(state)
        output = self.encode_transition(zs)
        return output, zs_mu, zs_logvar

    def load_save(self) -> None:
        for layer in self.module_list:
            layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.module_list:
            layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.module_list:
            decay_loss += layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
    def inference(self):
        self.training = False
    
    def uninference(self):
        self.training = True

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss
    

