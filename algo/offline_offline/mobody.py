import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algo import utils
from collections import defaultdict

# domain classifier for DARC
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

class ValueFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ValueFunc, self).__init__()
        self.network = MLPNetwork(state_dim, 1, hidden_size)

    def forward(self, state):
        return self.network(state)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim, hidden_size)

    def forward(self, x):
        mu = self.network(x)
        mean = torch.tanh(mu)
        
        return mean * self.max_action

class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)
    
def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class MOBODY(object):

    def __init__(self,
                 config,
                 device,
                 target_entropy=None,
                 ):
        self.config=  config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.update_interval = config['update_interval']
        self.fake_replay_buffer = utils.ReplayBuffer(config['state_dim'], config['action_dim'], device)

        
        self.penalty_type = config['penalty_type']
        
        
        
        # IQL hyperparameter
        # self.lam = config['lam']
        # self.temp = config['temp']
        # self.eta = config['eta']
        self.total_it = 0

        # aka critic
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        self.v_func = ValueFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)


        # aka actor
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        
        self.v_optimizer = torch.optim.Adam(self.v_func.parameters(), lr=config['critic_lr'])

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])

        # aka classifier
        self.classifier = Classifier(config['state_dim'], config['action_dim'], config['hidden_sizes'], config['gaussian_noise_std']).to(self.device)
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['actor_lr'])
    

    def select_action(self, state, policy, cuda = False):
        with torch.no_grad():
            action = policy(torch.Tensor(state).view(-1,self.config['state_dim']).to(self.device))
            if cuda:
                return action.squeeze()
            else:
                return action.squeeze().cpu().numpy()
    
    def update_classifier(self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None):
        src_state, src_action, src_next_state, _, _ = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(batch_size)
        if self.config['penalize_fake'] and self.fake_replay_buffer.size > 0:
            fake_state, fake_action, fake_next_state, _, _ = self.fake_replay_buffer.sample(batch_size)
            src_state = torch.cat([src_state, fake_state], 0)
            src_action = torch.cat([src_action, fake_action], 0)
            src_next_state = torch.cat([src_next_state, fake_next_state], 0)
            tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(2 * batch_size)
            

        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)

        # set labels for different domains
        label = torch.cat([torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], dim=0).long().to(self.device, non_blocking=True)

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

        # log necessary information if the logger is not None
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/sas classifier loss', loss_sas, global_step=self.total_it)
            writer.add_scalar('train/sa classifier loss', loss_sa, global_step=self.total_it)

        return loss_sa, loss_sas

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None, wandbrun = None):
        with torch.no_grad():
            nextaction_batch = self.policy(nextstate_batch)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + not_done_batch * self.discount * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)

        # if self.total_it % 200 == 0:
            # print(self.total_it)
            # print('q_1', q_1.mean().item(), q_1.max().item(), q_1.min().item())
            # print('q_2', q_2.mean().item(), q_2.max().item(), q_2.min().item())
            #print('value_target', value_target.mean().item(), value_target.max().item(), value_target.min().item())
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
            wandbrun.log({'train/q1': q_1.mean()}, step=self.total_it, commit=False)

        loss = F.mse_loss(q_1, value_target) +  F.mse_loss(q_2, value_target)
        return loss
    
    def update_q_functions_1(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None, wandbrun = None):
        with torch.no_grad():
            # nextaction_batch = self.policy(nextstate_batch)
            # q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = self.v_func(nextstate_batch)
            value_target = reward_batch + not_done_batch * self.discount * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)

        # if self.total_it % 200 == 0:
        #     print(self.total_it)
        #     print('q_1', q_1.mean().item(), q_1.max().item(), q_1.min().item())
        #     print('q_2', q_2.mean().item(), q_2.max().item(), q_2.min().item())
        #     print('value_target', value_target.mean().item(), value_target.max().item(), value_target.min().item())
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
            wandbrun.log({'train/q1': q_1.mean()}, step=self.total_it, commit=False)

        loss = F.mse_loss(q_1, value_target) +  F.mse_loss(q_2, value_target)
        return loss
    
    def update_v_function(self, state_batch, action_batch, writer=None):
        with torch.no_grad():
            q_t1, q_t2 = self.target_q_funcs(state_batch, action_batch)
            q_t = torch.min(q_t1, q_t2)
            
        v = self.v_func(state_batch)
        adv = q_t - v
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/adv', adv.mean(), self.total_it)
            writer.add_scalar('train/value', v.mean(), self.total_it)
        v_loss = asymmetric_l2_loss(adv, 0.7)
        return v_loss, adv

    
    
    def bc_loss(self, true_state_batch, true_action_batch, true_next_state_batch, writer = None):

        # BC loss
        pred_action = self.policy(true_state_batch)
        with torch.no_grad():
            q_b1, q_b2 = self.q_funcs(true_state_batch, true_action_batch)
            qval_batch = torch.min(q_b1, q_b2)

            v = self.v_func(true_state_batch)
            if self.config['advantage']:
                adv = qval_batch - v
            else:
                adv = qval_batch 
                adv = adv/ adv.abs().mean()

        exp_adv = torch.exp(3 * adv).clamp(max=100.0)
        # exp_adv = exp_adv/ exp_adv.mean()


        bc_loss = (pred_action - true_action_batch)**2
        if not self.config['q_weighted']:
            exp_adv = 1

        if self.total_it % 1000 == 0 and self.config['q_weighted']:
            print(torch.mean(exp_adv),torch.min(exp_adv),torch.max(exp_adv))
        policy_loss = torch.mean(exp_adv * bc_loss)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/exp_adv', exp_adv.mean(), self.total_it)
            writer.add_scalar('train/bc_loss', policy_loss.mean(), self.total_it)
        
        return policy_loss
    
    def update_policy_1(self, state_batch, action_batch, true_state_batch, true_action_batch,true_next_state_batch, writer = None,wandbrun = None):
        pred_action = self.policy(state_batch)
        q_b1, q_b2 = self.q_funcs(state_batch, pred_action)
        qval_batch = torch.min(q_b1, q_b2)
        # p_w = self.config['weight'] / qval_batch.abs().mean().detach()
        p_w = 1
        pred_true_action = self.policy(true_state_batch)

        policy_loss = p_w * (- qval_batch).mean() 
        # if self.total_it % 200 == 0:
        #     print('q in current policy ', qval_batch.mean().item(),qval_batch.max().item(),qval_batch.min().item()) 
        #     with torch.no_grad():
        #         q_behavior1, q_behavior2 = self.q_funcs(state_batch, action_batch)
        #         q_behavior = torch.min(q_behavior1, q_behavior2)
        #         print('q in behavior policy', q_behavior.mean().item(),q_behavior.max().item(),q_behavior.min().item()) 
        #     print(' ')

        policy_loss += self.config['bc_coef'] * self.bc_loss(true_state_batch, true_action_batch, true_next_state_batch)
        
        if writer is not None and self.total_it % 5000 == 0:
            with torch.no_grad():
                q_behavior1, q_behavior2 = self.q_funcs(state_batch, action_batch)
                q_behavior = torch.min(q_behavior1, q_behavior2)
            writer.add_scalar('train/q_behavior', q_behavior.mean(), self.total_it)
            writer.add_scalar('train/q_policy', qval_batch.mean(), self.total_it)
            writer.add_scalar('train/policy_loss', policy_loss, self.total_it)

            wandbrun.log({
                'train/q_behavior': q_behavior.mean(), 
                'train/q_policy': qval_batch.mean(), 
                'train/policy_loss': policy_loss, 
                }, step = self.total_it)
        return policy_loss
    


    def update_policy(self, state_batch, action_batch, true_state_batch, true_action_batch,true_next_state_batch, writer = None,wandbrun = None):
        pred_action = self.policy(state_batch)
        q_b1, q_b2 = self.q_funcs(state_batch, pred_action)
        qval_batch = torch.min(q_b1, q_b2)
        p_w = self.config['weight'] / qval_batch.abs().mean().detach()
        pred_true_action = self.policy(true_state_batch)

        policy_loss = p_w * (- qval_batch).mean() 
        # if self.total_it % 200 == 0:
        #     print('q in current policy ', qval_batch.mean().item(),qval_batch.max().item(),qval_batch.min().item()) 
        #     with torch.no_grad():
        #         q_behavior1, q_behavior2 = self.q_funcs(state_batch, action_batch)
        #         q_behavior = torch.min(q_behavior1, q_behavior2)
        #         print('q in behavior policy', q_behavior.mean().item(),q_behavior.max().item(),q_behavior.min().item()) 
        #     print(' ')

        policy_loss += self.config['bc_coef'] * self.bc_loss(true_state_batch, true_action_batch, true_next_state_batch)
        
        if writer is not None and self.total_it % 5000 == 0:
            with torch.no_grad():
                q_behavior1, q_behavior2 = self.q_funcs(state_batch, action_batch)
                q_behavior = torch.min(q_behavior1, q_behavior2)
            writer.add_scalar('train/q_behavior', q_behavior.mean(), self.total_it)
            writer.add_scalar('train/q_policy', qval_batch.mean(), self.total_it)
            writer.add_scalar('train/policy_loss', policy_loss, self.total_it)

            wandbrun.log({
                'train/q_behavior': q_behavior.mean(), 
                'train/q_policy': qval_batch.mean(), 
                'train/policy_loss': policy_loss, 
                }, step = self.total_it)
        return policy_loss

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None, wandbrun  = None):
        # self.wandbrun = wandbrun
        self.total_it += 1
        self.src_replay_buffer = src_replay_buffer
        self.tar_replay_buffer = tar_replay_buffer

        # update classifier
        if self.penalty_type == 'dara':
            if self.total_it == 1:
                for _ in range(10 * 500):
                # for _ in range(10):
                    loss_sa, loss_sas = self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)
                    if _ % 2000 == 0:
                        print(loss_sa, loss_sas)

                with torch.no_grad():
                    src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample_all(False)
                    # import time
                    # start = time.time()
                    idx_penalty = 0
                    while idx_penalty < len(src_state):

                        src_state_idx = src_state[idx_penalty:idx_penalty + 1000].to(self.device, non_blocking=False)
                        src_action_idx = src_action[idx_penalty:idx_penalty + 1000].to(self.device, non_blocking=False)
                        src_next_state_idx = src_next_state[idx_penalty:idx_penalty + 1000].to(self.device, non_blocking=False)
                        
                        sas_logits, sa_logits = self.classifier(src_state_idx, src_action_idx, src_next_state_idx, with_noise=False)
                        sas_probs, sa_probs = F.softmax(sas_logits, -1), F.softmax(sa_logits, -1)
                        sas_log_probs, sa_log_probs = torch.log(sas_probs + 1e-10), torch.log(sa_probs + 1e-10)
                        reward_penalty = sas_log_probs[:, 1:] - sa_log_probs[:, 1:] - sas_log_probs[:, :1] + sa_log_probs[:,:1]
                        # clip the panlty based on the DARA paper
                        reward_penalty = reward_penalty.clamp(-10, 10)
                        src_reward[idx_penalty:idx_penalty + 1000] += self.config['penalty_coef']* reward_penalty.cpu()
                        idx_penalty += 1000
                    src_replay_buffer.reward = src_reward
                    # print('dara time'   , time.time()-start)
                    
                    # test_src_reward = src_reward

                    # src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample_all()
                    # sas_logits, sa_logits = self.classifier(src_state, src_action, src_next_state, with_noise=False)
                    # sas_probs, sa_probs = F.softmax(sas_logits, -1), F.softmax(sa_logits, -1)
                    # sas_log_probs, sa_log_probs = torch.log(sas_probs + 1e-10), torch.log(sa_probs + 1e-10)
                    # reward_penalty = sas_log_probs[:, 1:] - sa_log_probs[:, 1:] - sas_log_probs[:, :1] + sa_log_probs[:,:1]
                    # # clip the panlty based on the DARA paper
                    # reward_penalty = reward_penalty.clamp(-10, 10)
                    # src_reward += self.config['penalty_coef']* reward_penalty

                    # src_replay_buffer.reward = src_reward.cpu().numpy()


        # sample from buffer, use 'trg_ratio' to control how many data from target (avoid upsample)
        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(int(self.config['src_ratio'] * batch_size))
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(int(self.config['trg_ratio'] * batch_size))
        # self.config['trg_ratio'] 

        # reshape source reward: dara or par
        # if self.penalty_type == 'dara':
        #     with torch.no_grad():
        #         src_sa_next_obs, src_sa_reward, _, _ = self.dynamics.step(src_state, src_action)
        #         sas_logits, sa_logits = self.classifier(src_state, src_action, src_next_state, with_noise=False)
        #         sas_probs, sa_probs = F.softmax(sas_logits, -1), F.softmax(sa_logits, -1)
        #         sas_log_probs, sa_log_probs = torch.log(sas_probs + 1e-10), torch.log(sa_probs + 1e-10)
        #         reward_penalty = sas_log_probs[:, 1:] - sa_log_probs[:, 1:] - sas_log_probs[:, :1] + sa_log_probs[:,:1]
        #         # clip the panlty based on the DARA paper
        #         reward_penalty = reward_penalty.clamp(-10, 10)
        #         if writer is not None and self.total_it % 100 == 0:
        #             writer.add_scalar('train/reward_penalty_dara', reward_penalty.mean(), global_step=self.total_it)
        #             if self.total_it % 5000 == 0:
        #                 wandbrun.log({'train/reward_penalty_dara': reward_penalty.mean()}, step=self.total_it, commit=False)

        #         src_reward += self.config['penalty_coef']* reward_penalty
        #         # if self.total_it % 200 == 0:
        #         #     print('src_reward mean', src_reward.mean(),src_reward.std())
        #         #     print('dara reward penalty',reward_penalty.mean(),reward_penalty.std())

        #         with torch.no_grad():
        #             src_sa_next_obs, src_sa_reward, _, _ = self.dynamics.step(src_state, src_action)
        #             penalty = torch.mean(((src_next_state-src_sa_next_obs)**2),axis = 1, keepdims = True)
        #         # if self.total_it % 200 == 0:
        #         #     print('par reward penalty',penalty.mean())
        if self.penalty_type == 'par':
            with torch.no_grad():
                src_sa_next_obs, src_sa_reward, _, _ = self.dynamics.step(src_state, src_action)
                penalty = torch.mean(((src_next_state-src_sa_next_obs)**2),axis = 1, keepdims = True)
                if writer is not None and self.total_it % 100 == 0:
                    writer.add_scalar('train/reward_penalty_par', penalty.mean(), global_step=self.total_it)
                src_reward -= self.config['penalty_coef']* penalty

        # rollout with src dynamics
        # rollout with target dynamics
        # print('reward penalty', time.time()-start)
        # start = time.time()
        # rollout new transition from src, every 1000 training steps
        if (self.total_it-1)%5000 == 0:
            src_state_init, src_action_init, src_next_state_init, _, _ = src_replay_buffer.sample(50000)
            tar_state_init, _, _, _, _ = tar_replay_buffer.sample(2000)
            rollout_transitions, rollout_info = self.rollout(src_state_init,self.config['src_rollout_length'])
            self.fake_replay_buffer.add_batch(rollout_transitions)

            if rollout_info is not None and self.total_it % 5000 == 0:
                wandbrun.log({'train/rollout_src_state': rollout_info['reward_mean']}, step=self.total_it, commit=False)

            # rollout_transitions, rollout_info = self.rollout(src_next_state_init,self.config['src_rollout_length'])
            # self.fake_replay_buffer.add_batch(rollout_transitions)

            rollout_transitions, rollout_info = self.rollout(tar_state_init,self.config['trg_rollout_length'])
            self.fake_replay_buffer.add_batch(rollout_transitions)
            if rollout_info is not None and self.total_it % 5000 == 0:
                wandbrun.log({'train/rollout_trg_state': rollout_info['reward_mean']}, step=self.total_it, commit=False)


            # src state and src action
            if self.config['use_src_sa_to_get_target_next_state']:
                src_init_next_obs, src_init_reward, ternimal, info = self.dynamics.step(src_state_init, src_action_init)
                new_rollout_transitions = defaultdict(list)
                # new_rollout_transitions["obss"] = src_state_init.cpu().numpy()
                # new_rollout_transitions["next_obss"] = src_init_next_obs.cpu().numpy()
                # new_rollout_transitions["actions"] = src_action_init.cpu().numpy()
                # new_rollout_transitions["rewards"] = src_init_reward.cpu().numpy()
                # new_rollout_transitions["terminals"] = ternimal
                rollout_index = (info["penalty"].cpu() <  self.config['env_filter']).squeeze(1)

                new_rollout_transitions["obss"] = src_state_init.cpu()[rollout_index]
                new_rollout_transitions["next_obss"] = src_init_next_obs.cpu()[rollout_index]
                new_rollout_transitions["actions"] = src_action_init.cpu()[rollout_index]
                new_rollout_transitions["rewards"] = src_init_reward.cpu()[rollout_index]
                new_rollout_transitions["terminals"] = torch.FloatTensor(ternimal)[rollout_index]
                self.fake_replay_buffer.add_batch(new_rollout_transitions)


            # rollout from source transition.
            if self.config['rollout_from_src']:
                if self.penalty_type != 'dara':
                    self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)
                
                src_state_init, _, src_next_state_init, _, _ = src_replay_buffer.sample(50000)
                tar_state_init, _, tar_next_state_init, _, _ = tar_replay_buffer.sample(100)
                
                rollout_transitions, rollout_info = self.rollout(torch.cat([src_state_init, \
                                                                            tar_state_init],0),\
                                                                                rollout_length=self.config['rollout_from_src_length'],use_trg=False)
                
                fake_state = rollout_transitions["obss"]
                fake_next_state = rollout_transitions["next_obss"]
                fake_action = rollout_transitions["actions"]
                src_r = rollout_transitions["rewards"]

                fake_state = torch.FloatTensor(fake_state).to('cuda')
                fake_next_state = torch.FloatTensor(fake_next_state).to('cuda')
                fake_action = torch.FloatTensor(fake_action).to('cuda')

                
                with torch.no_grad():
                    sas_logits, sa_logits = self.classifier(fake_state, fake_action, fake_next_state, with_noise=False)
                    sas_probs, sa_probs = F.softmax(sas_logits, -1), F.softmax(sa_logits, -1)
                    sas_log_probs, sa_log_probs = torch.log(sas_probs + 1e-10), torch.log(sa_probs + 1e-10)
                    reward_penalty = sas_log_probs[:, 1:] - sa_log_probs[:, 1:] - sas_log_probs[:, :1] + sa_log_probs[:,:1]
                    # clip the panlty based on the DARA paper
                    reward_penalty = reward_penalty.clamp(-10, 10)
                    src_r += self.config['penalty_coef']* reward_penalty.cpu().numpy()

                rollout_transitions["rewards"] = src_r
                self.fake_replay_buffer.add_batch(rollout_transitions)

                if rollout_info is not None and self.total_it % 5000 == 0:
                    wandbrun.log({'train/rollout_from_src': rollout_info['reward_mean']}, step=self.total_it, commit=False)
        # print('rollout', time.time()-start)
        # start = time.time()
        if self.config['fake_batch_scale'] == 0:
            state = torch.cat([src_state, tar_state], 0)
            action = torch.cat([src_action, tar_action], 0)
            next_state = torch.cat([src_next_state, tar_next_state], 0)
            reward = torch.cat([src_reward, tar_reward], 0)
            not_done = torch.cat([src_not_done, tar_not_done], 0)
        else:
            fake_state, fake_action, fake_next_state, fake_reward, fake_not_done = self.fake_replay_buffer.sample(int(self.config['fake_batch_scale'] * batch_size))
            # concat data
            state = torch.cat([src_state, tar_state, fake_state], 0)
            action = torch.cat([src_action, tar_action, fake_action], 0)
            next_state = torch.cat([src_next_state, tar_next_state, fake_next_state], 0)
            reward = torch.cat([src_reward, tar_reward, fake_reward], 0)
            not_done = torch.cat([src_not_done, tar_not_done, fake_not_done], 0)
        # print('concat data', time.time()-start)
        # start = time.time()
        # update v (might not used)
        if self.config['advantage']:
            v_loss_step, adv = self.update_v_function(state, action, writer)
            self.v_optimizer.zero_grad()
            v_loss_step.backward()
            self.v_optimizer.step()
        

        # update q 
        if self.config['advantage']:
            q_loss_step = self.update_q_functions_1(state, action, reward, next_state, not_done, writer, wandbrun)
        else:
            q_loss_step = self.update_q_functions(state, action, reward, next_state, not_done, writer, wandbrun)
        
        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()
        # print('update q', time.time()-start)
        # start = time.time()

        self.update_target()

        # update policy and temperature parameter
        for p in self.q_funcs.parameters():
            p.requires_grad = False

        # update policy
        if self.config['scale_Q']:
            pi_loss_step = self.update_policy(state, action,\
                                            torch.cat([src_state, tar_state], 0), \
                                                torch.cat([src_action, tar_action], 0),\
                                                    torch.cat([src_next_state, tar_next_state], 0), writer, wandbrun)

        else:
            pi_loss_step = self.update_policy_1(state, action,\
                                            torch.cat([src_state, tar_state], 0), \
                                                torch.cat([src_action, tar_action], 0),\
                                                    torch.cat([src_next_state, tar_next_state], 0), writer, wandbrun)
        
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()
        # print('update policy', time.time()-start)
        # print(' ')

        for p in self.q_funcs.parameters():
            p.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
    
    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int,
        use_trg = True
    ):
        if rollout_length == 0:
            return None, None

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations , self.policy, cuda = True)
            # print(observations.shape, actions.shape)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions.reshape(-1, self.config['action_dim']), use_trg)

            # rewards = rewards.cpu().numpy()
            # rollout_transitions["obss"].append(observations.cpu().numpy())
            # rollout_transitions["next_obss"].append(next_observations.cpu().numpy())
            # rollout_transitions["actions"].append(actions.cpu().numpy())
            # rollout_transitions["rewards"].append(rewards.cpu().numpy())
            # rollout_transitions["terminals"].append(terminals)
            # rollout_transitions["penalty"].append(info['penalty'].cpu().numpy())

            rollout_transitions["obss"].append(observations.cpu())
            rollout_transitions["next_obss"].append(next_observations.cpu())
            rollout_transitions["actions"].append(actions.cpu())
            rollout_transitions["rewards"].append(rewards.cpu())
            rollout_transitions["terminals"].append(torch.FloatTensor(terminals))
            rollout_transitions["penalty"].append(info['penalty'].cpu())
            

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.cpu().numpy().flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            for rollout_index in range(len(v)):
                if len(v[rollout_index].shape) == 1:
                    v[rollout_index] = v[rollout_index].reshape(-1, len(v[rollout_index]))
                
            rollout_transitions[k] = torch.cat(v, axis=0)
            # rollout_transitions[k] = torch.FloatTensor(rollout_transitions[k])
        if self.config['filter_bad_rollout']:
            idx = (rollout_transitions['penalty'] <= self.config['env_filter']).squeeze(1)
            for k, v in rollout_transitions.items():
                # print(v.shape, idx.shape)
                rollout_transitions[k] = v[idx]
            print('filtered rollout', sum(idx), len(idx))



        return rollout_transitions, {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}
