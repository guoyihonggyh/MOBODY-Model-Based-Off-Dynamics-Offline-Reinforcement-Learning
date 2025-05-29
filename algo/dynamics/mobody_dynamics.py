import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Callable, List, Tuple, Dict

class Classifier(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256, gaussian_noise_std=1.0):
        super(Classifier, self).__init__()
        self.action_dim = action_dim
        self.gaussian_noise_std = gaussian_noise_std
        self.sa_classifier = MLPNetwork(state_dim + action_dim, 2, hidden_size)
        self.sas_classifier = MLPNetwork(2*state_dim + action_dim, 2, hidden_size)

    def forward(self, state_batch, action_batch, nextstate_batch, with_noise):
        sas = torch.cat([state_batch, action_batch, nextstate_batch], -1)

        if with_noise:
            sas += torch.randn_like(sas, device=state_batch.device) * self.gaussian_noise_std
        sas_logits = torch.nn.Softmax()(self.sas_classifier(sas))

        sa = torch.cat([state_batch, action_batch], -1)

        if with_noise:
            sa += torch.randn_like(sa, device=state_batch.device) * self.gaussian_noise_std
        sa_logits = torch.nn.Softmax()(self.sa_classifier(sa))

        return sas_logits, sa_logits

class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)
class BaseDynamics(object):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer
    ) -> None:
        super().__init__()
        self.model = model
        self.optim = optim
    
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        raise NotImplementedError



import os
import numpy as np
import torch
import torch.nn as nn

import numpy as np
import os.path as path
import torch



from typing import Callable, List, Tuple, Dict, Optional
# from offlinerlkit.dynamics import BaseDynamics
# from offlinerlkit.utils.scaler import StandardScaler
# from offlinerlkit.utils.logger import Logger
# from utils.logger import Logger
from algo.mb_utils.logger import Logger
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

        # print('scaler is fit')

        self.mu = torch.zeros(self.mu.shape).to('cuda')

        self.std = torch.ones(self.std.shape).to('cuda')

        print('scaler is not fit')

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        # return (data - self.mu) / self.std
        return data

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        # return self.std * data + self.mu
        return data
    
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

        self.mu = torch.FloatTensor(self.mu).to('cuda')
        self.std = torch.FloatTensor(self.std).to('cuda')

        self.mu = 0
        self.std = 1

    def transform_tensor(self, data: torch.Tensor):
        device = data.device
        data = self.transform(data)
        # data = torch.tensor(data, device=device)
        return data

class MOBODYEnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        config,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "pairwise-diff"
    ) -> None:
        super().__init__(model, optim)
        # self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode

        self.obs_scaler = StandardScaler()
        self.action_scaler = StandardScaler()


        self.encoder_loss_coef = config['encoder_loss_coef']
        self.domain_loss_coef = config['domain_loss_coef']
        self.cycle_loss_coef = config['cycle_loss_coef']

        self.config = config

        self.encode_trg_diff = self.model.encode_trg_diff

        # self.train_target_freq = 20

    @ torch.no_grad()
    def step(
        self,
        obs, # torch
        action, # torch
        use_penalty = True,
        use_trg = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        
        info = {}


        self.model.inference()
        "imagine single forward step"


        obs_standard = self.obs_scaler.transform_tensor(obs)

        if use_trg:
            mean, _, _  = self.model.forward_trg(obs_standard, action)
        else:
            mean, _, _  = self.model.forward_src(obs_standard, action)
        
        info['samples'] = mean

        std = torch.std(mean, dim=0, keepdim = True).repeat(7,1,1) 
        
        ensemble_samples = mean + torch.normal(mean=torch.zeros_like(mean), std=std)

        #MSE
        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        # samples = ensemble_samples.mean(0)
        
        next_obs = samples     
        next_obs = self.obs_scaler.inverse_transform(next_obs)
        if self.config['latent_reward']:
            zs, za,zs_next = self.get_latent_for_reward(obs_standard,action, True)
            reward, _  = self.model.encode_reward(zs.mean(0), za.mean(0),zs_next.mean(0))
        else:
            reward, _  = self.model.encode_reward(obs_standard, action, samples)
        reward = reward.mean(0)
        terminal = self.terminal_fn(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy())
        
        info["raw_reward"] = reward

        if self._uncertainty_mode == "aleatoric":
            # penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            penalty = torch.amax(torch.norm(std, dim=2), dim=0)

        elif self._uncertainty_mode == "pairwise-diff":
            next_obses_mean = mean[..., :-1]
            next_obs_mean = torch.mean(next_obses_mean, dim=0)
            diff = next_obses_mean - next_obs_mean
            penalty = torch.amax(torch.norm(diff, dim=2), dim=0)
        elif self._uncertainty_mode == "ensemble_std":
            next_obses_mean = mean[..., :-1]
            penalty = torch.sqrt(next_obses_mean.var(0).mean(1))

        else:
            raise ValueError
        
        penalty = penalty.reshape(len(penalty),1)
        
        info["penalty"] = penalty

        if self._penalty_coef and use_penalty:
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
        
        return next_obs, reward, terminal, info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        # mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss



    def kl_gaussian(self, mu1, logvar1, mu2, logvar2):
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        kl = 0.5 * torch.sum(
            torch.log(var2 / var1)
            + (var1 + (mu1 - mu2) ** 2) / var2
            - 1,
            dim=-1
        ).mean(dim=-1)
        return kl  # shape: (batch,)

    def encoder_loss(self, state_batch, action_batch, nextstate_batch, trg,writer=None):


        recon_state, state_mu, state_logvar = self.model.encoder_decoder(state_batch)
        recon_next_state, state_next_mu, state_next_logvar = self.model.encoder_decoder(nextstate_batch)


        # recon_state, state_mu, state_logvar = self.model.encoder_decoder(state_batch)

        recon_loss = ((recon_state - state_batch) ** 2).mean(dim=(1, 2)).sum() + \
            ((recon_next_state - nextstate_batch) ** 2).mean(dim=(1, 2)).sum() 
        loss = 100 * recon_loss
        kl_loss = self.get_kl_loss(state_mu, state_logvar) + self.get_kl_loss(state_next_mu, state_next_logvar)
        loss += kl_loss
        
        latent_state, latent_state_mu, latent_state_logvar= self.model.encode_state(state_batch)
        if trg:
            latent_action = self.model.encode_trg_action(latent_state, action_batch, True)
        else:
            latent_action = self.model.encode_src_action(latent_state, action_batch, True)
        # print(latent_action.shape, latent_state.shape)
        latent_state = latent_state + latent_action
        with torch.no_grad():
            latent_next_state, _, _  = self.model.encode_state(nextstate_batch)

        loss += ((latent_state - latent_next_state) ** 2).mean(dim=(1, 2)).sum()

        loss = 0.01 * self.model.get_decay_loss()
        return loss, recon_loss, kl_loss
    
    def get_kl_loss( self, mu, logvar):
        kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=(1, 2))
        kl_div = kl_div.sum()
        return 0.05 * kl_div
    
    def transition_loss(self, state_batch, action_batch, nextstate_batch, trg, logvar_loss_coef, writer=None):
        if trg:
            mean, zs_mu, zs_logvar = self.model.forward_trg(state_batch, action_batch)
        else:
            mean, zs_mu, zs_logvar = self.model.forward_src(state_batch, action_batch)


        loss = ((mean - nextstate_batch) ** 2).mean(dim=(1, 2)).sum()
        # KL_loss =  self.get_kl_loss(zs_mu, zs_logvar)
        # loss += KL_loss
        return loss

    def reward_loss(self, state_batch, action_batch, next_state_batch, rewards_batch, trg):
        # with torch.no_grad():
        if trg:
            mean,_, log_var = self.model.forward_trg(state_batch, action_batch)
        else:
            mean, _, log_var = self.model.forward_src(state_batch, action_batch)
        fake_next_state = mean + torch.randn_like(mean) * torch.std(mean, axis=0, keepdim=True)

        # if np.random.rand() < 0.7:
        #     train_transition = 10
        #     for param in self.model.reward_model1.parameters():
        #         param.requires_grad = False
        #     for param in self.model.reward_model2.parameters():
        #         param.requires_grad = False
        #     for param in self.model.reward_model3.parameters():
        #         param.requires_grad = False
        # else:
        #     train_transition = 1
        #     for param in self.model.reward_model1.parameters():
        #         param.requires_grad = True
        #     for param in self.model.reward_model2.parameters():
        #         param.requires_grad = True
        #     for param in self.model.reward_model3.parameters():
        #         param.requires_grad = True




        reward,_ = self.model.encode_reward(state_batch, \
                                           action_batch,\
                                            fake_next_state)
        loss = ((reward - rewards_batch) ** 2).mean(dim=(1, 2))
        
        loss = 1 * loss.sum()

        
        true_reward,_ = self.model.encode_reward(state_batch, action_batch, next_state_batch)
        loss_true = ((true_reward - rewards_batch) ** 2).mean(dim=(1, 2))
        loss += loss_true.sum()
        if not trg:
            loss = 0.01 * loss
        else:
            loss = loss
        return loss
    

    def get_latent_for_reward(self, state_batch, action_batch, trg):
        with torch.no_grad():
            zs, zmu, zlogvar = self.model.encode_state(state_batch)
            if trg:
                za, zamu, zalogvar = self.model.encode_trg_action(action_batch)
            else:
                za, zamu, zalogvar = self.model.encode_src_action(action_batch)
            zs_next_hat = zs + za
        return zs, za, zs_next_hat
    
    def reward_loss_with_latent(self, state_batch, action_batch, next_state_batch, rewards_batch, trg):
        with torch.no_grad():

            zs, zmu, zlogvar = self.model.encode_state(state_batch)
            if trg:
                za, zamu, zalogvar = self.model.encode_trg_action(action_batch)
            else:
                za, zamu, zalogvar = self.model.encode_src_action(action_batch)
            zs_next, _, _ = self.model.encode_state(next_state_batch)
            zs_next_hat = zs + za
        
        reward,_ = self.model.encode_reward(zs, \
                                            za,\
                                            zs_next_hat)
        loss = ((reward - rewards_batch) ** 2).mean(dim=(1, 2))
        loss = loss.sum()

        true_reward,_ = self.model.encode_reward(zs, \
                                            za,\
                                            zs_next)
        loss_true = ((true_reward - rewards_batch) ** 2).mean(dim=(1, 2))
        loss += loss_true.sum()
        return loss






    def get_inverse_action(self, src_data):

        self.model.inference()
        with torch.no_grad():

            # src_data = self.src_replay_buffer.sample_all()

            src_obss_ = src_data[0]
            src_actions_ = src_data[1]
            src_next_obss_ = src_data[2]
            src_rewards_ = src_data[3]

            from collections import defaultdict
            rollout_transitions = defaultdict(list)

            # state_batch = torch.FloatTensor(src_obss).to('cuda')
            # nextstate_batch = torch.FloatTensor(src_next_obss).to('cuda')
            i = 0
            while i < len(src_obss_):
                src_obss = src_obss_[i:i + 5000]
                src_next_obss = src_next_obss_[i:i + 5000]

                zs, _, _ = self.model.encode_state(src_obss)
                zs_next, _, _ = self.model.encode_state(src_next_obss)

                za = zs_next - zs
                trg_action_batch = self.model.decode_trg_action(za)
                trg_action_batch = trg_action_batch.mean(0)

                # print(trg_action_batch.shape, src_obss.shape)

                rewards_batch, _ = self.model.encode_reward(src_obss, trg_action_batch, src_next_obss)
                rewards_batch = rewards_batch.mean(0)

                # rollout_transitions = {}
                rollout_transitions["obss"].append(src_obss.cpu().numpy())
                rollout_transitions["next_obss"].append( src_next_obss.cpu().numpy())
                rollout_transitions["actions"].append( trg_action_batch.cpu().numpy())
                rollout_transitions["rewards"].append( rewards_batch.cpu().numpy())
                rollout_transitions["terminals"].append( torch.zeros_like(rewards_batch).cpu().numpy())
                i += 5000
            for k, v in rollout_transitions.items():
                for rollout_index in range(len(v)):
                    if len(v[rollout_index].shape) == 1:
                        v[rollout_index] = v[rollout_index].reshape(-1, len(v[rollout_index]))
                
                rollout_transitions[k] = np.concatenate(v, axis=0)
        return rollout_transitions
    

    def learn_sep_reward(self, src_train_obss, src_train_actions, src_train_next_obss, src_train_rewards,\
                         trg_train_obss, trg_train_actions, trg_train_next_obss, trg_train_rewards, batch_size):

        self.model.train()
        train_size = trg_train_obss.shape[1]
        losses = []
        transition_losses = []
        encoder_losses = []
        domain_losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            self.total_steps += 1

            src_obss_batch = src_train_obss[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            src_actions_batch = src_train_actions[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            src_next_obss_batch = src_train_next_obss[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            src_rewards_batch = src_train_rewards[:, batch_num * batch_size:(batch_num + 1) * batch_size]

            trg_obss_batch = trg_train_obss[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            trg_actions_batch = trg_train_actions[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            trg_next_obss_batch = trg_train_next_obss[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            trg_rewards_batch = trg_train_rewards[:, batch_num * batch_size:(batch_num + 1) * batch_size]

            if self.config['latent_reward']:
                src_reward_loss = self.reward_loss_with_latent(src_obss_batch, src_actions_batch, src_next_obss_batch, src_rewards_batch, False)
                trg_reward_loss = self.reward_loss_with_latent( trg_obss_batch, trg_actions_batch, trg_next_obss_batch, trg_rewards_batch, True)
            else:
                src_reward_loss = self.reward_loss( src_obss_batch, src_actions_batch, src_next_obss_batch, src_rewards_batch, False)
                trg_reward_loss = self.reward_loss( trg_obss_batch, trg_actions_batch, trg_next_obss_batch, trg_rewards_batch, True)
            loss = src_reward_loss + trg_reward_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())

        return np.mean(losses)

    def learn_src_trg(self, use_trg_data, train_obss, train_actions, train_next_obss, train_rewards, \
                      train_obss_trg, train_actions_trg, train_next_obss_trg, train_rewards_trg,
                      batch_size, logvar_loss_coef, trg_transition = None):

        self.model.train()
        train_size = train_obss_trg.shape[1]
        losses = []
        transition_losses = []
        encoder_losses = []
        domain_losses = []
        recon_losses = []
        kl_losses =[]

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            self.total_steps += 1

            obss_batch = train_obss[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            actions_batch = train_actions[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            next_obss_batch = train_next_obss[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            rewards_batch = train_rewards[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            encoder_loss, recon_loss, kl_loss = self.encoder_loss( obss_batch, actions_batch, next_obss_batch, False)
            nextstate_batch_reward = torch.cat([next_obss_batch, rewards_batch], axis=-1)
            transition_loss = self.transition_loss( obss_batch, actions_batch, next_obss_batch, False,logvar_loss_coef)
            
            # print(encoder_loss  )
            # if use_trg_data:
            #     loss = transition_loss + 5 * self.encoder_loss_coef * encoder_loss
            # else:
            loss = transition_loss + self.encoder_loss_coef * encoder_loss

            if not self.config['inverse_sep_reward_loss']:
                if self.config['latent_reward']:
                    reward_loss = self.reward_loss_with_latent(obss_batch, actions_batch, next_obss_batch, rewards_batch, False)
                else:
                    reward_loss = self.reward_loss( obss_batch, actions_batch, next_obss_batch, rewards_batch, False)
                loss += reward_loss
            
            # trg 
            obss_batch = train_obss_trg[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            actions_batch = train_actions_trg[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            next_obss_batch = train_next_obss_trg[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            rewards_batch = train_rewards_trg[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            encoder_loss, recon_loss, kl_loss= self.encoder_loss( obss_batch, actions_batch, next_obss_batch, True)
            nextstate_batch_reward = torch.cat([next_obss_batch, rewards_batch], axis=-1)
            transition_loss = self.transition_loss( obss_batch, actions_batch, next_obss_batch, True,logvar_loss_coef)
            
            # print(encoder_loss  )
            # if use_trg_data:
            loss += transition_loss + self.encoder_loss_coef * encoder_loss

            if not self.config['inverse_sep_reward_loss']:
                if self.config['latent_reward']:
                    reward_loss = self.reward_loss_with_latent(obss_batch, actions_batch, next_obss_batch, rewards_batch, True)
                else:
                    reward_loss = self.reward_loss( obss_batch, actions_batch, next_obss_batch, rewards_batch, True)
                loss += reward_loss


            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
            transition_losses.append(transition_loss.item())
            encoder_losses.append(encoder_loss.item())
            kl_losses.append(kl_loss.item())


        return np.mean(losses), np.mean(transition_losses), np.mean(encoder_losses), np.mean(recon_losses), np.mean(kl_losses)
    



    def learn(self, use_trg_data, train_obss, train_actions, train_next_obss, train_rewards, batch_size, logvar_loss_coef, trg_transition = None):

        self.model.train()
        train_size = train_obss.shape[1]
        losses = []
        transition_losses = []
        encoder_losses = []
        domain_losses = []
        kl_losses = []

        n_batch = int(np.ceil(train_size / batch_size))
        recon_losses = []
        for batch_num in range(int(np.ceil(train_size / batch_size))):
            self.total_steps += 1
            import time
            # start = time.time()
            obss_batch = train_obss[:, batch_num * batch_size:(batch_num + 1) * batch_size].to('cuda', non_blocking = True)
            actions_batch = train_actions[:, batch_num * batch_size:(batch_num + 1) * batch_size].to('cuda', non_blocking = True)
            next_obss_batch = train_next_obss[:, batch_num * batch_size:(batch_num + 1) * batch_size].to('cuda', non_blocking = True)
            rewards_batch = train_rewards[:, batch_num * batch_size:(batch_num + 1) * batch_size].to('cuda', non_blocking = True)
            # print('to gpu', time.time() - start)

            if not self.config['no_vae']:
                encoder_loss, recon_loss, kl_loss = self.encoder_loss( obss_batch, actions_batch, next_obss_batch, use_trg_data)
            nextstate_batch_reward = torch.cat([next_obss_batch, rewards_batch], axis=-1)
            transition_loss = self.transition_loss( obss_batch, actions_batch, next_obss_batch, use_trg_data,logvar_loss_coef)
            
            # print(encoder_loss  )
            if not self.config['no_vae']:
                if use_trg_data:
                    loss = transition_loss + 5 * self.encoder_loss_coef * encoder_loss
                else:
                    loss = transition_loss + 1 * self.encoder_loss_coef * encoder_loss
                encoder_losses.append(encoder_loss.item())
                recon_losses.append(recon_loss.item())
                kl_losses.append(kl_loss.item())
                
            else:
                loss = transition_loss 
                encoder_losses.append(0)
                recon_losses.append(0)
                kl_losses.append(0)

            if not self.config['inverse_sep_reward_loss']:
                if self.config['latent_reward']:
                    reward_loss = self.reward_loss_with_latent(obss_batch, actions_batch, next_obss_batch, rewards_batch, use_trg_data)
                else:
                    reward_loss = self.reward_loss( obss_batch, actions_batch, next_obss_batch, rewards_batch, use_trg_data)
                loss += reward_loss
            

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
            transition_losses.append(transition_loss.item())
            

        return np.mean(losses), np.mean(transition_losses), np.mean(encoder_losses), np.mean(recon_losses), np.mean(kl_losses)


    def shuffle_rows(self, arr):
        idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
        return arr[np.arange(arr.shape[0])[:, None], idxes]
    
    def update_classifier(self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None):
        src_state, src_action, src_next_state, _, _ = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(batch_size)

        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)

        # set labels for different domains
        label = torch.cat([torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], dim=0).long().to(self.device)

        indexs = torch.randperm(label.shape[0])
        state_batch, action_batch, nextstate_batch = state[indexs], action[indexs], next_state[indexs]
        label = label[indexs]

        sas_logits, sa_logits = self.classifier(state_batch, action_batch, nextstate_batch, with_noise=True)
        loss_sas = F.cross_entropy(sas_logits, label)
        loss_sa =  F.cross_entropy(sa_logits, label)
        classifier_loss = loss_sas + loss_sa
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

        return loss_sa, loss_sas

    def data_augmentation(self, buffer):
        self.device = 'cuda'
        src_replay_buffer, tar_replay_buffer = buffer 
        self.classifier = Classifier(self.config['state_dim'], self.config['action_dim'], self.config['hidden_sizes'], self.config['gaussian_noise_std']).to('cuda')
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.config['actor_lr'])

        for _ in range(8000):
            loss_sa, loss_sas = self.update_classifier(src_replay_buffer, tar_replay_buffer, 256, None)
            if _ % 2000 == 0:
                print(loss_sa, loss_sas)
        

        with torch.no_grad():
            src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample_all()
            sas_logits, sa_logits = self.classifier(src_state, src_action, src_next_state, with_noise=False)
            sas_probs, sa_probs = F.softmax(sas_logits, -1), F.softmax(sa_logits, -1)

            sas_probs = sas_probs[:, 1]

            print('distribution 0.5',sum(sas_probs > 0.5))
            print(sum(sas_probs > 0.6))
            print(sum(sas_probs > 0.65))
            print(sum(sas_probs > 0.7))
            print(sum(sas_probs > 0.75))

            include_in_target_training = sas_probs > self.config['train_with_src_threshold']

            print('number of added data',sum(include_in_target_training))

            import algo.utils as utils
            src_replay_buffer_sim_trg = utils.ReplayBuffer(self.config['state_dim'], self.config['action_dim'], 'cuda')
            batch = {}
            batch["obss"] = src_state[include_in_target_training].cpu()
            batch["next_obss"] = src_next_state[include_in_target_training].cpu()
            batch["actions"] = src_action[include_in_target_training].cpu()
            batch["rewards"] = src_reward[include_in_target_training].cpu()
            batch["terminals"] = src_not_done[include_in_target_training].cpu()
            src_replay_buffer_sim_trg.add_batch(batch)

            self.src_replay_buffer_sim_trg = src_replay_buffer_sim_trg


        
    


    def train(
        self,
        src_data: Dict,
        trg_data: Dict,
        # logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
        writer = None,
        buffer = None
    ) -> None:
        
        if self.config['train_with_src_threshold'] != 1:
            self.data_augmentation(buffer)


        
        self.src_replay_buffer = src_data
        
        self.total_steps = 0

        src_obss = src_data[0]
        src_actions = src_data[1]
        src_next_obss = src_data[2]
        src_rewards = src_data[3]

        src_data_size = src_obss.shape[0]

        trg_obss = trg_data[0]
        trg_actions = trg_data[1]
        trg_next_obss = trg_data[2]
        trg_rewards = trg_data[3]

        trg_data_size = trg_obss.shape[0]

        
        src_holdout_size = min(int(src_data_size * holdout_ratio), 1000)
        trg_holdout_size = min(int(trg_data_size * holdout_ratio), 500)

        src_train_size = src_data_size - src_holdout_size
        src_train_splits, src_holdout_splits = torch.utils.data.random_split(range(src_data_size), (src_train_size, src_holdout_size))
        
        trg_train_size = trg_data_size - trg_holdout_size
        trg_train_splits, trg_holdout_splits = torch.utils.data.random_split(range(trg_data_size), (trg_train_size, trg_holdout_size))

        


        
        src_train_obss, src_train_actions, src_train_next_obss, src_train_rewards = src_obss[src_train_splits.indices], \
                                                                    src_actions[src_train_splits.indices],\
                                                                    src_next_obss[src_train_splits.indices], \
                                                                    src_rewards[src_train_splits.indices]
        src_holdout_obss, src_holdout_actions, src_holdout_next_obss, src_holdout_rewards = src_obss[src_holdout_splits.indices],\
                                                                    src_actions[src_holdout_splits.indices], \
                                                                    src_next_obss[src_holdout_splits.indices],\
                                                                    src_rewards[src_holdout_splits.indices]    
        

        
        trg_train_obss, trg_train_actions, trg_train_next_obss, trg_train_rewards = trg_obss[trg_train_splits.indices],\
                                                                    trg_actions[trg_train_splits.indices],\
                                                                    trg_next_obss[trg_train_splits.indices],\
                                                                    trg_rewards[trg_train_splits.indices]
        if self.config['train_with_src_threshold'] != 1:
            sim_trg_data = self.src_replay_buffer_sim_trg.sample_all()
            sim_trg_train_obss, \
                sim_trg_train_actions, \
                    sim_trg_train_next_obss, \
                        sim_trg_train_rewards = sim_trg_data[0].cpu(), sim_trg_data[1].cpu(), sim_trg_data[2].cpu(), sim_trg_data[3].cpu()
            print(trg_train_rewards.shape)
            trg_train_obss = torch.cat([trg_train_obss, sim_trg_train_obss], axis=0)
            trg_train_actions = torch.cat([trg_train_actions, sim_trg_train_actions], axis=0)
            trg_train_next_obss = torch.cat([trg_train_next_obss, sim_trg_train_next_obss], axis=0)
            trg_train_rewards = torch.cat([trg_train_rewards, sim_trg_train_rewards], axis=0)
            print(trg_train_rewards.shape)

            trg_train_size = trg_train_obss.shape[0]

        
        
        trg_holdout_obss, trg_holdout_actions, trg_holdout_next_obss, trg_holdout_rewards = trg_obss[trg_holdout_splits.indices],\
                                                                    trg_actions[trg_holdout_splits.indices], \
                                                                    trg_next_obss[trg_holdout_splits.indices], \
                                                                    trg_rewards[trg_holdout_splits.indices]  
        
        self.obs_scaler.fit(torch.cat([src_train_obss, trg_train_obss], axis=0))
        # self.action_scaler.fit(torch.cat([src_train_actions, trg_train_actions], axis=0))

        src_train_obss = self.obs_scaler.transform(src_train_obss)
        src_holdout_obss = self.obs_scaler.transform(src_holdout_obss)
        trg_train_obss = self.obs_scaler.transform(trg_train_obss)
        trg_holdout_obss = self.obs_scaler.transform(trg_holdout_obss)

        src_train_next_obss = self.obs_scaler.transform(src_train_next_obss)
        trg_train_next_obss = self.obs_scaler.transform(trg_train_next_obss)



        src_holdout_losses = [1e10 for i in range(self.model.num_ensemble)]
        trg_holdout_losses = [1e10 for i in range(self.model.num_ensemble)]
        src_data_idxes = torch.randint(src_train_size, size=[self.model.num_ensemble, src_train_size])
        trg_data_idxes = torch.randint(trg_train_size, size=[self.model.num_ensemble, trg_train_size])

        
        epoch = 0
        cnt = 0

        print("Training dynamics:")
        import time
        start = time.time()
        
        while True:
            # break
            epoch += 1
            self.epoch = epoch
            print('train time', time.time() - start)
            start = time.time()


            if self.config['train_together']:

                src_train_loss, src_transition_loss, src_encoder_loss, domain_loss,_ = self.learn( False, \
                                                                    src_train_obss[src_data_idxes], \
                                                                    src_train_actions[src_data_idxes], \
                                                                    src_train_next_obss[src_data_idxes], \
                                                                    src_train_rewards[src_data_idxes], \
                                                                    batch_size, logvar_loss_coef)

                src_trg_train_loss, src_trg_transition_loss, src_trg_encoder_loss, src_trg_domain_loss, _ = \
                    self.learn_src_trg(False, src_train_obss[src_data_idxes], src_train_actions[src_data_idxes], src_train_next_obss[src_data_idxes], src_train_rewards[src_data_idxes], \
                                trg_train_obss[trg_data_idxes], trg_train_actions[trg_data_idxes], trg_train_next_obss[trg_data_idxes], trg_train_rewards[trg_data_idxes], \
                                batch_size, logvar_loss_coef)
                

                src_new_holdout_transition_losses, src_new_holdout_encode_losses  \
                    = self.validate(False, src_holdout_obss, src_holdout_actions, src_holdout_next_obss, src_holdout_rewards)
                src_holdout_loss = (np.sort(src_new_holdout_transition_losses)[:self.model.num_elites]).mean()

                trg_new_holdout_transition_losses, trg_new_holdout_encode_losses  \
                    = self.validate(True, trg_holdout_obss, trg_holdout_actions, trg_holdout_next_obss, trg_holdout_rewards)
                trg_holdout_loss = (np.sort(trg_new_holdout_transition_losses)[:self.model.num_elites]).mean()
                trg_holdout_reward_loss = (np.sort(trg_new_holdout_encode_losses)[:self.model.num_elites]).mean()
            
                print("src_loss/dynamics_holdout_loss", src_holdout_loss)
                print("trg_loss/dynamics_holdout_loss", trg_holdout_loss)
                print("trg_loss/dynamics_holdout_reward_loss", trg_holdout_reward_loss)
            else:

                src_train_loss, src_transition_loss, src_encoder_loss, recon_loss, kl_loss = self.learn( False, \
                                                                    src_train_obss[src_data_idxes], \
                                                                    src_train_actions[src_data_idxes], \
                                                                    src_train_next_obss[src_data_idxes], \
                                                                    src_train_rewards[src_data_idxes], \
                                                                    batch_size, logvar_loss_coef)
                src_new_holdout_transition_losses, src_new_holdout_encode_losses  \
                    = self.validate(False, src_holdout_obss, src_holdout_actions, src_holdout_next_obss, src_holdout_rewards)
                src_holdout_loss = (np.sort(src_new_holdout_transition_losses)[:self.model.num_elites]).mean()

                print(epoch)
                print("src_loss/dynamics_train_loss", src_transition_loss)
                print("src_loss/dynamics_encoder_loss", src_encoder_loss)
                print("src_loss/dynamics_recon_loss", recon_loss)
                print("src_loss/dynamics_kl_loss", kl_loss)
                print("src_loss/dynamics_holdout_loss", src_holdout_loss)
                if writer is not None:
                    writer.add_scalar("src_loss/dynamics_train_loss", src_transition_loss, global_step=epoch)
                    writer.add_scalar("src_loss/dynamics_encoder_loss", src_encoder_loss, global_step=epoch)
                    writer.add_scalar("src_loss/dynamics_domain_loss", recon_loss, global_step=epoch)
                    writer.add_scalar("src_loss/dynamics_holdout_loss", src_holdout_loss, global_step=epoch)

                for _ in range(3):
                    src_transition = [src_train_obss[src_data_idxes], \
                                    src_train_actions[src_data_idxes], \
                                    src_train_next_obss[src_data_idxes], \
                                    src_train_rewards[src_data_idxes]]
                    trg_train_loss, trg_transition_loss, trg_encoder_loss, recon_loss, kl_loss = self.learn( True, \
                                                                        trg_train_obss[trg_data_idxes], \
                                                                        trg_train_actions[trg_data_idxes], \
                                                                        trg_train_next_obss[trg_data_idxes], \
                                                                        trg_train_rewards[trg_data_idxes], \
                                                                        batch_size, logvar_loss_coef,src_transition)
        
                trg_new_holdout_transition_losses, trg_new_holdout_encode_losses  \
                    = self.validate(True, trg_holdout_obss, trg_holdout_actions, trg_holdout_next_obss, trg_holdout_rewards)
                trg_holdout_loss = (np.sort(trg_new_holdout_transition_losses)[:self.model.num_elites]).mean()
                trg_holdout_reward_loss = (np.sort(trg_new_holdout_encode_losses)[:self.model.num_elites]).mean()
                
                

                print("trg_loss/dynamics_train_loss", trg_transition_loss)
                print("trg_loss/dynamics_encoder_loss", trg_encoder_loss)
                print("trg_loss/dynamics_recon_loss", recon_loss)
                print("trg_loss/dynamics_kl_loss", kl_loss)
                print("trg_loss/dynamics_holdout_loss", trg_holdout_loss)
                print("trg_loss/dynamics_holdout_reward_loss", trg_holdout_reward_loss)
                print(' ')
                if writer is not None:
                    writer.add_scalar("trg_loss/dynamics_train_loss", trg_transition_loss, global_step=epoch)
                    writer.add_scalar("trg_loss/dynamics_encoder_loss", trg_encoder_loss, global_step=epoch)
                    writer.add_scalar("trg_loss/dynamics_holdout_loss", trg_holdout_loss, global_step=epoch)
                
                if self.config['inverse_sep_reward_loss']:

                    reward_loss = self.learn_sep_reward(src_train_obss[src_data_idxes], src_train_actions[src_data_idxes], \
                                        src_train_next_obss[src_data_idxes], src_train_rewards[src_data_idxes],
                                        trg_train_obss[trg_data_idxes], trg_train_actions[trg_data_idxes], \
                                            trg_train_next_obss[trg_data_idxes], trg_train_rewards[trg_data_idxes], batch_size)
                    print("trg_loss/dynamics_reward_loss", reward_loss)
                src_data_idxes = self.shuffle_rows(src_data_idxes)
                trg_data_idxes = self.shuffle_rows(trg_data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(trg_holdout_losses)), trg_new_holdout_transition_losses, trg_holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    trg_holdout_losses[i] = new_loss

            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                # print("src_loss/dynamics_train_loss", round(src_train_loss, 5))
                # print("src_loss/dynamics_train_transition_loss", round(src_transition_loss, 5))
                # print("src_loss/dynamics_train_encoder_loss", round(src_encoder_loss, 5))
                print("src_loss/dynamics_holdout_loss", round(src_holdout_loss,5))

                # print("trg_loss/dynamics_train_loss", round(trg_train_loss, 5))
                # print("trg_loss/dynamics_train_transition_loss", round(trg_transition_loss, 5))
                # print("trg_loss/dynamics_train_encoder_loss", round(trg_encoder_loss, 5))
                print("trg_loss/dynamics_holdout_loss", round(trg_holdout_loss,5))
                
                break
            
        
        indexes = self.select_elites(trg_holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        # self.save(logger.model_dir)
        self.model.eval()
        print("elites:{} , holdout loss: {}".format(indexes, (np.sort(trg_holdout_losses)[:self.model.num_elites]).mean()))
    
    
    
    
    

    def _train(
        self,
        src_data: Dict,
        trg_data: Dict,
        # logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:

        obss = src_data[0]
        actions = src_data[1]
        next_obss = src_data[2]
        rewards = src_data[3]

        data_size = obss.shape[0]
        
        
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))

        train_obss, train_actions, train_next_obss, train_rewards = obss[train_splits.indices], actions[train_splits.indices], next_obss[train_splits.indices], rewards[train_splits.indices]
        holdout_obss, holdout_actions, holdout_next_obss, holdout_rewards = obss[holdout_splits.indices], actions[holdout_splits.indices], next_obss[holdout_splits.indices], rewards[holdout_splits.indices]


        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = torch.randint(train_size, size=[self.model.num_ensemble, train_size])
        
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        # logger.log("Training dynamics:")

        import time
        start_time = time.time()

        print("Training target dynamics:")

        while True:
            epoch += 1
            train_loss, transition_loss, encoder_loss = self.learn( use_trg_data, \
                                                                   train_obss[data_idxes], \
                                                                   train_actions[data_idxes], \
                                                                   train_next_obss[data_idxes], \
                                                                   train_rewards[data_idxes], \
                                                                   batch_size, logvar_loss_coef)

            
            new_holdout_transition_losses, new_holdout_encode_losses  \
                = self.validate(use_trg_data, holdout_obss, holdout_actions, holdout_next_obss, holdout_rewards)
            holdout_loss = (np.sort(new_holdout_transition_losses)[:self.model.num_elites]).mean()

            print("loss/dynamics_train_loss", transition_loss)
            print("loss/dynamics_holdout_loss", holdout_loss)
            print(epoch)

            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_transition_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                print("loss/dynamics_train_loss", round(train_loss, 5))
                print("loss/dynamics_train_transition_loss", round(transition_loss, 5))
                print("loss/dynamics_train_encoder_loss", round(encoder_loss, 5))
                print("loss/dynamics_holdout_loss", round(holdout_loss,5))
                break
            
        
        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        # self.save(logger.model_dir)
        self.model.eval()
        print("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))

        print(epoch,time.time() - start_time )
        self.model.inference()
    
    # def learn(
    #     self,
    #     inputs: np.ndarray,
    #     targets: np.ndarray,
    #     batch_size: int = 256,
    #     logvar_loss_coef: float = 0.01
    # ) -> float:
    #     self.model.train()
    #     train_size = inputs.shape[1]
    #     losses = []

    #     for batch_num in range(int(np.ceil(train_size / batch_size))):
    #         inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
    #         targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
    #         # targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
    #         mean, logvar = self.model(inputs_batch)
    #         inv_var = torch.exp(-logvar)
    #         # Average over batch and dim, sum over ensembles.
    #         mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
    #         var_loss = logvar.mean(dim=(1, 2))
    #         loss = mse_loss_inv.sum() + var_loss.sum()
    #         loss = loss + self.model.get_decay_loss()
    #         loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

    #         self.optim.zero_grad()
    #         loss.backward()
    #         self.optim.step()

    #         losses.append(loss.item())
    #     return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, use_trg_data, holdout_obss, holdout_actions, holdout_next_obss, holdout_rewards) -> List[float]:
        self.model.eval()
        self.model.inference()
        
        holdout_obss = holdout_obss.to('cuda', non_blocking = True)
        holdout_actions = holdout_actions.to('cuda', non_blocking = True)
        holdout_next_obss = holdout_next_obss.to('cuda', non_blocking = True)
        holdout_rewards = holdout_rewards.to('cuda', non_blocking = True)
        # transition loss
        # encode loss
        if use_trg_data:
            mean, _, _ = self.model.forward_trg(holdout_obss, holdout_actions)
        else:
            mean, _, _ = self.model.forward_src(holdout_obss, holdout_actions)

        mean = self.obs_scaler.inverse_transform(mean)
        transition_loss = ((mean - holdout_next_obss) ** 2).mean(dim=(1, 2))
        
        # print(mean.shape, holdout_actions.shape)
        if self.config['latent_reward']:
            zs, za, zs_next_hat = self.get_latent_for_reward(holdout_obss.squeeze(0).repeat(7,1,1), holdout_actions.squeeze(0).repeat(7,1,1), True)
            pred_reward, _ = self.model.encode_reward(zs, za, zs_next_hat)
        else:
            pred_reward, _ = self.model.encode_reward(holdout_obss.squeeze(0).repeat(7,1,1), holdout_actions.squeeze(0).repeat(7,1,1), mean)
        encode_loss = ((pred_reward - holdout_rewards) ** 2).mean(dim=(1, 2))

        val_transition_loss = list(transition_loss.cpu().numpy())
        val_encode_loss = list(encode_loss.cpu().numpy())


        state_hat = self.model.encoder_decoder(holdout_obss)[0].mean(0)
        vae_recon_eval_loss = torch.sqrt(((state_hat - holdout_obss)**2).sum(dim = -1)).mean(dim = 0)
        print("vae_recon_eval_loss", vae_recon_eval_loss.item())

        self.model.uninference()

        return val_transition_loss, val_encode_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        # self.scaler.save_scaler(save_path)

        self.obs_scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.obs_scaler.load_scaler(load_path)



        
    def train_trg_only(
        self,
        src_data: Dict,
        trg_data: Dict,
        # logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
        writer = None
    ) -> None:

        trg_obss = trg_data[0]
        trg_actions = trg_data[1]
        trg_next_obss = trg_data[2]
        trg_rewards = trg_data[3]

        trg_data_size = trg_obss.shape[0]

        
        trg_holdout_size = min(int(trg_data_size * holdout_ratio), 500)


        trg_train_size = trg_data_size - trg_holdout_size
        trg_train_splits, trg_holdout_splits = torch.utils.data.random_split(range(trg_data_size), (trg_train_size, trg_holdout_size))

        
        trg_train_obss, trg_train_actions, trg_train_next_obss, trg_train_rewards = trg_obss[trg_train_splits.indices],\
                                                                    trg_actions[trg_train_splits.indices],\
                                                                    trg_next_obss[trg_train_splits.indices],\
                                                                    trg_rewards[trg_train_splits.indices]
        trg_holdout_obss, trg_holdout_actions, trg_holdout_next_obss, trg_holdout_rewards = trg_obss[trg_holdout_splits.indices],\
                                                                    trg_actions[trg_holdout_splits.indices], \
                                                                    trg_next_obss[trg_holdout_splits.indices], \
                                                                    trg_rewards[trg_holdout_splits.indices]  
        
        

        trg_holdout_losses = [1e10 for i in range(self.model.num_ensemble)]
        trg_data_idxes = torch.randint(trg_train_size, size=[self.model.num_ensemble, trg_train_size])
        
        epoch = 0
        cnt = 0

        print("Training dynamics:")
        
        while True:
            epoch += 1

            train_obss, train_actions, train_next_obss, train_rewards = trg_train_obss[trg_data_idxes], \
                                                       trg_train_actions[trg_data_idxes],\
                                                        trg_train_next_obss[trg_data_idxes],\
                                                        trg_train_rewards[trg_data_idxes]
            self.model.train()
            train_size = train_obss.shape[1]
            losses = []
    
            for batch_num in range(int(np.ceil(train_size / batch_size))):
                obss_batch = train_obss[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                actions_batch = train_actions[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                next_obss_batch = train_next_obss[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                rewards_batch = train_rewards[:, batch_num * batch_size:(batch_num + 1) * batch_size]
    
                nextstate_batch_reward = torch.cat([next_obss_batch, rewards_batch], axis=-1)
                transition_loss = self.transition_loss( obss_batch, actions_batch, nextstate_batch_reward, False,logvar_loss_coef)
                transition_loss += 0.1*self.model.get_decay_loss()
                self.optim.zero_grad()
                transition_loss.backward()
                self.optim.step()
    
                losses.append(transition_loss.item())

            
            

            trg_new_holdout_transition_losses, trg_new_holdout_encode_losses  \
                = self.validate(True, trg_holdout_obss, trg_holdout_actions, trg_holdout_next_obss, trg_holdout_rewards)
            trg_holdout_loss = (np.sort(trg_new_holdout_transition_losses)[:self.model.num_elites]).mean()
            
            
            # print(losses)
            print("trg_loss/dynamics_train_loss", np.mean(losses))
            # print("trg_loss/dynamics_encoder_loss", trg_encoder_loss)
            print("trg_loss/dynamics_holdout_loss", trg_holdout_loss)

            # src_data_idxes = self.shuffle_rows(src_data_idxes)
            trg_data_idxes = self.shuffle_rows(trg_data_idxes)


            if epoch >= 30:
                break

        # indexes = self.select_elites(trg_holdout_losses)
        # self.model.set_elites(indexes)
        # self.model.load_save()
        # # self.save(logger.model_dir)
        # self.model.eval()
        # print("elites:{} , holdout loss: {}".format(indexes, (np.sort(trg_holdout_losses)[:self.model.num_elites]).mean()))
        # print(c)
        