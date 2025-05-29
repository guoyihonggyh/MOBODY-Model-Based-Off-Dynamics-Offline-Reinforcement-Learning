import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TD3BC(object):

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

        self.total_it = 0

        # aka critic
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])

                # aka classifier
        self.classifier = Classifier(config['state_dim'], config['action_dim'], config['hidden_sizes']).to(self.device)
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['actor_lr'])
    

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

        # log necessary information if the logger is not None
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/sas classifier loss', loss_sas, global_step=self.total_it)
            writer.add_scalar('train/sa classifier loss', loss_sa, global_step=self.total_it)
    
    def select_action(self, state, test=True):
        with torch.no_grad():
            action = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        return action.squeeze().cpu().numpy()

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            nextaction_batch = self.policy(nextstate_batch)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + not_done_batch * self.discount * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
        loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)
        return loss

    def update_policy(self, state_batch, action_batch):
        pred_action = self.policy(state_batch)
        q_b1, q_b2 = self.q_funcs(state_batch, pred_action)
        qval_batch = torch.min(q_b1, q_b2)
        p_w = self.config['weight'] / qval_batch.abs().mean().detach()

        adv = qval_batch / qval_batch.abs().mean().detach()
        adv = torch.exp(1 * adv).clamp(max = 100.0)
        if self.config['advantage'] == 1:
            bc_loss =  adv * (pred_action - action_batch)**2
            bc_loss = bc_loss.mean()
        else:
            bc_loss =  (pred_action - action_batch)**2
            bc_loss = bc_loss.mean()
        
        policy_loss = p_w * (- qval_batch).mean() + self.config['bc_coef'] * bc_loss 
        return policy_loss

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None, wandbrun = None):

        self.total_it += 1

        

        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(int(self.config['trg_ratio'] * batch_size))
        
        if self.config['penalty_type'] == 'dara':
            self.update_classifier(src_replay_buffer, tar_replay_buffer, batch_size, writer)
            with torch.no_grad():
                sas_logits, sa_logits = self.classifier(src_state, src_action, src_next_state, with_noise=False)
                sas_probs, sa_probs = F.softmax(sas_logits, -1), F.softmax(sa_logits, -1)
                sas_log_probs, sa_log_probs = torch.log(sas_probs + 1e-10), torch.log(sa_probs + 1e-10)
                reward_penalty = sas_log_probs[:, 1:] - sa_log_probs[:, 1:] - sas_log_probs[:, :1] + sa_log_probs[:,:1]
                # clip the panlty based on the DARA paper
                reward_penalty = reward_penalty.clamp(-10, 10)

                if writer is not None and self.total_it % 5000 == 0:
                    writer.add_scalar('train/reward penalty', reward_penalty.mean(), global_step=self.total_it)

            src_reward += 0.1 * reward_penalty
        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        reward = torch.cat([src_reward, tar_reward], 0)
        not_done = torch.cat([src_not_done, tar_not_done], 0)

        q_loss_step = self.update_q_functions(state, action, reward, next_state, not_done, writer)

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        # update policy and temperature parameter
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        pi_loss_step = self.update_policy(state, action)
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()

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