import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
import yaml
import json # in case the user want to modify the hyperparameters
import d4rl # used to make offline environments for source domains
import algo.utils as utils

from pathlib                              import Path
from algo.call_algo                       import call_algo
from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.infos                           import get_normalized_score
from tensorboardX                         import SummaryWriter
from algo.mb_utils.terminal_funs import get_termination_fn
from algo.vec_env import VecEnv

import os

import gym
# from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np

import warnings
from tqdm import tqdm


from algo.dynamics.mobody_dynamics import MOBODYEnsembleDynamics
from algo.dynamics.mobody_module import MOBODYModule



import wandb

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def eval_policy_batch(policy, env, policy_distribution,eval_episodes=10, eval_cnt=None, dynamics = None, eval_trg = False):
    eval_env = env
    state_list = []
    action_list = []
    next_state_list = []
    reward_list = []
    done_list = []  

    avg_reward = 0.

    state = eval_env.reset()
    mydone = np.zeros(eval_episodes)

    done_index = np.ones(eval_episodes, dtype=int) * 1000

    reward_all = np.zeros((eval_episodes,1000))

    it = 0

    while sum(mydone) < eval_episodes:
        action = policy.select_action(np.array(state), policy_distribution)
        next_state, reward, done, _ = eval_env.step(action)
        reward_all[:,it] = reward
        
        for i in range(eval_episodes):
            if done[i] and mydone[i] == 0:
                mydone[i] = 1
                done_index[i] = it
                state_list.append(state[i])
                action_list.append(action[i])
                next_state_list.append(next_state[i])
                reward_list.append(reward[i])
                done_list.append(done[i])
            elif mydone[i] == 0:
                state_list.append(state[i])
                action_list.append(action[i])
                next_state_list.append(next_state[i])
                reward_list.append(reward[i])
                done_list.append(done[i])

        state = next_state

        it += 1
    # print(done_index)
    avg_reward = np.array([np.sum(reward_all[i, :done_index[i] + 1 ]) for i in range(eval_episodes)]).sum() 
    avg_reward /= eval_episodes

    if eval_trg and dynamics is not None:
        # print(state_list)
        state_list = torch.tensor(state_list).float().to('cuda', non_blocking=True)
        action_list = torch.tensor(action_list).float().to('cuda', non_blocking=True)
        next_state_list = torch.tensor(next_state_list).float().to('cuda', non_blocking=True)
        reward_list = torch.tensor(reward_list).float().to('cuda', non_blocking=True)

        next_obs, reward, terminal, info = dynamics.step(state_list,action_list,False)

        obs_mse = torch.mean(torch.sqrt(torch.sum((next_obs - next_state_list)**2, dim=1)))

        obs_mse_indiviual = torch.sqrt(torch.sum((next_obs - next_state_list)**2, dim=1))
        reward_mse_loss = torch.mean((reward_list - reward.squeeze(1))**2)


        randix = np.random.permutation(np.arange(len(reward_list)))[:8]
        try:
            print('true reward        ',reward_list[randix])
            print('pred reward        ',reward[randix].squeeze(1))
            print('penalty            ', info['penalty'][randix].reshape(1,-1))
            print('obs mse id         ', obs_mse_indiviual[randix])
            # print('fake reward penalty', reward[randix].squeeze(1) - dynamics._penalty_coef * info['penalty'][randix])
            preward = reward[randix].squeeze(1) - dynamics._penalty_coef  * info['penalty'][randix]

            # print('reward count > 0', sum(preward > 0)/len(preward) )
            # print('reward count > 1', sum(preward > 1)/len(preward) )

            
            print('penalty std', torch.std(info['penalty']))
            print('penalty min', torch.min(info['penalty']))
        except:
            pass
        print('reward mse', reward_mse_loss.item())
        print('obs mse', obs_mse.item())

    if eval_trg:
        print("[{}] Evaluation on target over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))
    else:
        print("[{}] Evaluation on source over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward

def eval_policy(policy, env, policy_distribution,eval_episodes=10, eval_cnt=None, dynamics = None, eval_trg = False):
    eval_env = env
    state_list = []
    action_list = []
    next_state_list = []
    reward_list = []
    done_list = []  

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), policy_distribution)
            next_state, reward, done, _ = eval_env.step(action)
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            reward_list.append(reward)
            done_list.append(done)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    if eval_trg and dynamics is not None:
        # print(state_list)
        state_list = torch.tensor(state_list).float().to('cuda', non_blocking=True)
        action_list = torch.tensor(action_list).float().to('cuda', non_blocking=True)
        next_state_list = torch.tensor(next_state_list).float().to('cuda', non_blocking=True)
        reward_list = torch.tensor(reward_list).float().to('cuda', non_blocking=True)

        next_obs, reward, terminal, info = dynamics.step(state_list,action_list,False)

        obs_mse = torch.mean(torch.sqrt(torch.sum((next_obs - next_state_list)**2, dim=1)))

        obs_mse_indiviual = torch.sqrt(torch.sum((next_obs - next_state_list)**2, dim=1))
        reward_mse_loss = torch.mean((reward_list - reward.squeeze(1))**2)


        randix = np.random.permutation(np.arange(len(reward_list)))[:8]
        # try:
        #     print('true reward        ',reward_list[randix])
        #     print('pred reward        ',reward[randix].squeeze(1))
        #     print('penalty            ', info['penalty'][randix])
        #     print('obs mse id         ', obs_mse_indiviual[randix])
        #     print('fake reward penalty', reward[randix].squeeze(1) - dynamics._penalty_coef * info['penalty'][randix])
        #     preward = reward[randix].squeeze(1) - dynamics._penalty_coef  * info['penalty'][randix]

        #     print('reward count > 0', sum(preward > 0)/len(preward) )
        #     print('reward count > 1', sum(preward > 1)/len(preward) )

            
        #     print('penalty std', torch.std(info['penalty']))
        #     print('penalty min', torch.min(info['penalty']))
        # except:
        #     pass
        print('reward mse', reward_mse_loss.item())
        print('obs mse', obs_mse.item())

    if eval_trg:
        print("[{}] Evaluation on one target over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))
    else:
        print("[{}] Evaluation on one source over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="SAC", help='policy to use')
    parser.add_argument("--env", default="halfcheetah-friction")
    parser.add_argument('--srctype', default="medium", help='dataset type used in the source domain') # only useful when source domain is offline
    parser.add_argument('--tartype', default="medium", help='dataset type used in the target domain') # only useful when target domain is offline
    # support dataset type:
    # source domain: all valid datasets from D4RL
    # target domain: random, medium, medium-expert, expert
    parser.add_argument('--shift_level', default=0.1, help='the scale of the dynamics shift. Note that this value varies on different settins')
    parser.add_argument('--mode', default=3, type=int, help='the training mode, there are four types, 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    parser.add_argument('--max_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    parser.add_argument("--purely_model_based", default=0, type=int)
    parser.add_argument("--num_envs", default=32, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)

    parser.add_argument("--dara_eta", default=0, type=float)


    # model-based params
    parser.add_argument('--only_use_trg_transition', default=5e4, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--transition_update_freq', default=2500, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--transition_update_start', default=5e4, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--trg_rollout_batch_size', default=5e4, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--trg_rollout_length', default=1, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--src_rollout_batch_size', default=5e4, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--src_rollout_length', default=1, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--model_based_training_steps', default=100, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--fake_batch_scale', default=0.5, type=float)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--dynamics_lr', default=1e-3, type=float)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--encoder_loss_coef', default=1, type=float)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--domain_loss_coef', default=0.0 , type=float)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--cycle_loss_coef', default=0.3, type=float)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--bc_coef', default=1.0, type=float)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--q_weighted', default=1, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--advantage', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--scale_q', default=1, type=int)  # the maximum gradient step for off-dynamics rl learning

    
    # offline method
    parser.add_argument('--mobile', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--relu_reward', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--penalize_fake', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning

    parser.add_argument('--gaussian_dynamics', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--sep_reward_dynamics', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--vae', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--sas_reward', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--inverse', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--inverse_sep_reward_loss', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--latent_reward', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--filter_bad_rollout', default=1, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--train_together', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--encode_sa', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--vae_a', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning

    
    parser.add_argument('--train_with_src_threshold', default=1, type=float)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--env_filter', default=10, type=float)  # the maximum gradient step for off-dynamics rl learning
# use_src_sa_to_get_target_next_state
    parser.add_argument('--use_src_sa_to_get_target_next_state', default=1, type=int)  # the maximum gradient step for off-dynamics rl learning

    parser.add_argument('--env_penalty_coef', default=0.1, type=float)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--lcb_penalty_coef', default=0, type=float)  # the maximum gradient step for off-dynamics rl learning

    parser.add_argument('--rollout_length', default=1, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--rollout_from_src', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--rollout_from_src_length', default=2, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--trg_ratio', default=1, type=float)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--src_ratio', default=1, type=float)  # the maximum gradient step for off-dynamics rl learning

    # if save dynamics:
    parser.add_argument('--dynamics_path', default=None, type=str)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--train_dynamics', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning

    
    parser.add_argument('--out_dir_remark', default='', type=str)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--penalty_type', default='par', type=str)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--penalty_coef', default=0.1, type=float)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--representation_noise', default=0, type=float)  # the maximum gradient step for off-dynamics rl learning

    parser.add_argument('--group', default=None, type=str)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--wandb', default=1, type=int)  # the maximum gradient step for off-dynamics rl learning
    
    # ablations

    parser.add_argument('--cat', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning

    parser.add_argument('--no_vae', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--trg_only', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--mopo', default=0, type=int)  # the maximum gradient step for off-dynamics rl learning

    
    args = parser.parse_args()
    if not args.wandb:
        wandb.init(mode="offline")
    # we support different ways of specifying tasks, e.g., hopper-friction, hopper_friction, hopper_morph_torso_easy, hopper-morph-torso-easy
    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    elif 'pen' in args.env or 'relocate' in args.env or 'door' in args.env or 'hammer' in args.env:
        domain = 'adroit'
    elif 'antmaze' in args.env:
        domain = 'antmaze'
    else:
        raise NotImplementedError
    # print(domain)

    call_env = {
        'mujoco': call_mujoco_env,
        'adroit': call_adroit_env,
        'antmaze': call_antmaze_env,
    }

    # determine referenced environment name
    ref_env_name = args.env + '-' + str(args.shift_level)
    
    if domain == 'antmaze':
        src_env_name = args.env
        src_env_name_config = args.env
    elif domain == 'adroit':
        src_env_name = args.env
        src_env_name_config = args.env.split('-')[0]
    else:
        src_env_name = args.env.split('-')[0]
        src_env_name_config = src_env_name
    tar_env_name = args.env

    def make_src_env(src_env_name):
        return gym.make(src_env_name)

    # make environments
    if args.mode == 1 or args.mode == 3:
        if domain == 'antmaze':
            src_env_name = src_env_name.split('-')[0]
            src_env_name += '-' + args.srctype + '-play-v0'
        elif domain == 'adroit':
            src_env_name = src_env_name.split('-')[0]
            # src_env_name += '-' + args.srctype + '-v0'
            src_env_name += '-' + 'human-v0'
        else:
            src_env_name += '-' + args.srctype + '-v2'
        src_env = None
        print(src_env_name)
        src_eval_env = gym.make(src_env_name)
        # print(123213)
        src_eval_env.seed(args.seed)
    else:
        if 'antmaze' in src_env_name:
            src_env_config = {
                'env_name': src_env_name,
                'shift_level': args.shift_level,
            }
            src_env = call_env[domain](src_env_config)
            src_env.seed(args.seed)
            src_eval_env = call_env[domain](src_env_config)
            src_eval_env.seed(args.seed + 100)
        else:
            src_env_config = {
                'env_name': src_env_name,
                'shift_level': args.shift_level,
            }
            src_env = call_env[domain](src_env_config)
            # src_env = VecEnv(src_env, args.num_envs, args.seed)
            # src_env.seed(args.seed)
            src_eval_env = call_env[domain](src_env_config)
            src_eval_env.seed(args.seed + 100)

    if args.mode == 2 or args.mode == 3:
        tar_env = None
        tar_env_config = {
            'env_name': tar_env_name,
            'shift_level': args.shift_level,
        }
        tar_eval_env = call_env[domain](tar_env_config)
        tar_eval_env.seed(args.seed + 100)
    else:
        tar_env_config = {
            'env_name': tar_env_name,
            'shift_level': args.shift_level,
        }
        # tar_env = call_env[domain](tar_env_config)
        tar_env = call_env[domain](tar_env_config)
        tar_env = VecEnv(tar_env, args.num_envs, args.seed)
        # tar_env.seed(args.seed)
        tar_eval_env =  call_env[domain](tar_env_config)
        tar_eval_env.seed(args.seed + 100)
    
    if args.mode not in [0,1,2,3]:
        raise NotImplementedError # cannot support other modes
    
    policy_config_name = args.policy.lower()

    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/{domain}/{policy_config_name}/{src_env_name_config}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    # log path, we use logging with tensorboard
    if args.mode == 1:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    elif args.mode == 2:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    elif args.mode == 3:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    else:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-' + str(args.shift_level) + '/r' + str(args.seed)
    outdir = outdir + args.out_dir_remark
    writer = SummaryWriter('{}/tb'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    # seed all
    # src_env.action_space.seed(args.seed) if src_env is not None else None
    # tar_env.action_space.seed(args.seed) if tar_env is not None else None
    src_eval_env.action_space.seed(args.seed)
    tar_eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # get necessary information from both domains
    state_dim = src_eval_env.observation_space.shape[0]
    action_dim = src_eval_env.action_space.shape[0] 
    max_action = float(src_eval_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine shift_level
    if domain == 'mujoco':
        if args.shift_level in ['easy', 'medium', 'hard']:
            shift_level = args.shift_level
        else:
            shift_level = float(args.shift_level)
    else:
        shift_level = args.shift_level
    
    if args.mobile == 1:
        env_penalty_coef = 0.0
    else:
        env_penalty_coef = args.env_penalty_coef


    config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'max_step': int(args.max_step),
        'shift_level': shift_level,
        'dara_eta': args.dara_eta,
        
         # model-based
        'only_use_trg_transition':args.only_use_trg_transition,
        'transition_update_freq':args.transition_update_freq,
        'transition_update_start':args.transition_update_start,
        'trg_rollout_batch_size':int(args.trg_rollout_batch_size),
        'trg_rollout_length':int(args.trg_rollout_length),
        'src_rollout_batch_size':int(args.src_rollout_batch_size),
        'src_rollout_length':int(args.src_rollout_length),
        'model_based_training_steps':args.model_based_training_steps,
        'fake_batch_scale':args.fake_batch_scale,
        'env_penalty_coef':args.env_penalty_coef,
        'lcb_penalty_coef':args.lcb_penalty_coef,

        'encoder_loss_coef': args.encoder_loss_coef, 
        'domain_loss_coef': args.domain_loss_coef, 
        'cycle_loss_coef' : args.cycle_loss_coef,
        'use_src_sa_to_get_target_next_state': args.use_src_sa_to_get_target_next_state,
        

        #
        'penalty_type': args.penalty_type,
        'penalty_coef':args.penalty_coef,
        'rollout_length': args.rollout_length,

        'penalize_fake': args.penalize_fake,
        'representation_noise': args.representation_noise,

        'eval_freq': args.eval_freq,
        'bc_coef': args.bc_coef,

        'rollout_from_src':args.rollout_from_src,
        'rollout_from_src_length' :args.rollout_from_src_length,
        'env_filter': args.env_filter,

        'trg_ratio' : args.trg_ratio,
        'src_ratio' : args.src_ratio,
        'q_weighted': args.q_weighted,
        'advantage': args.advantage,
        'scale_Q': args.scale_q,
        'inverse_sep_reward_loss': args.inverse_sep_reward_loss,
        'latent_reward': args.latent_reward,
        'filter_bad_rollout': args.filter_bad_rollout,
        'train_together': args.train_together,

        'train_with_src_threshold': args.train_with_src_threshold,

        # ablations
        'no_vae': args.no_vae,
        'trg_only': args.trg_only,
        'mopo': args.mopo,

    })

    terminal_fn = get_termination_fn(src_env_name)
    policy = call_algo(args.policy, config, args.mode, device, terminal_fn = terminal_fn)
    
    ## write logs to record training parameters
    with open(outdir + 'log.txt','w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    tar_mb_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device,max_size=int(5e6))
    t_target_rollout_trajectories = 0

    # in case that the domain is offline, we directly load its offline data
    if args.mode == 1 or args.mode == 3:
        src_replay_buffer.convert_D4RL(d4rl.qlearning_dataset(src_eval_env))
        if 'antmaze' in args.env:
            src_replay_buffer.reward -= 1.0
    
    if args.mode == 2 or args.mode == 3:
        tar_dataset = call_tar_dataset(tar_env_name, shift_level, args.tartype)
        tar_replay_buffer.convert_D4RL(tar_dataset)
        if 'antmaze' in args.env:
            tar_replay_buffer.reward -= 1.0

    eval_cnt = 0
    
    # eval_src_return = eval_policy(policy, src_eval_env, policy.policy_darc, eval_cnt=eval_cnt)
    # eval_tar_return = eval_policy(policy, tar_eval_env, policy.policy, eval_cnt=eval_cnt)
    eval_cnt += 1

    if args.mode == 0:
        # online-online learning

        src_state, src_done = src_env.reset(), False
        tar_state, tar_done = tar_env.reset(), False
        src_episode_reward, src_episode_timesteps, src_episode_num = np.zeros(args.num_envs), np.zeros(args.num_envs), 0
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = np.zeros(args.num_envs), np.zeros(args.num_envs), 0
        t = 0
        rollout_steps = -1
        while t <=int(config['max_step']):
        # for t in range(int(config['max_step'])):
            src_episode_timesteps += 1
            rollout_steps += 1
            # select action randomly or according to policy, if the policy is deterministic, add exploration noise akin to TD3 implementation
            src_action = (
                policy.select_action(np.array(src_state), policy.policy_darc, test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            src_next_state, src_reward, src_done, _ = src_env.step(src_action) 
            # src_done_bool = float(src_done) if src_episode_timesteps < 1000 else 0

            

            src_done_bool = (src_episode_timesteps >= 1000) | src_done
            
            if 'antmaze' in args.env:
                src_reward -= 1.0
            

            src_replay_buffer.add_batch_sep(src_state, src_action, src_next_state, src_reward.reshape(len(src_reward),1),\
                                            src_done_bool.reshape(len(src_reward),1))

            src_state = src_next_state
            src_episode_reward += src_reward

            t += args.num_envs
            for i, done in enumerate(src_done_bool):
                if done:
                    # src_state[i] = src_env.reset_one(i)
                    # print(src_env.env_method("reset", indices=[i])[0])
                    # (src_state[i],_) = src_env.env_method("reset", indices=[i])[0]
                    src_state[i] = src_env.reset(i)
                    print("Total T: {} Episode Num: {} Episode T: {} Source Reward: {}".format(t+1, src_episode_num+1, \
                                                                                               src_episode_timesteps[i], src_episode_reward[i]))
                # writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

                    # src_state, src_done = src_env.reset(), False
                    src_episode_reward[i] = 0
                    src_episode_timesteps[i] = 0
                    src_episode_num += 1

            
                    
            
            # interaction with tar env
            if rollout_steps % config['tar_env_interact_interval']  == 0:
                # if t_target_rollout_trajectories % 2 == 0:
                rollout_steps += 1
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), policy.policy, test=False)

                tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
                # tar_done_bool = float(tar_done) if tar_episode_timesteps < 1000 else 0
                tar_done_bool = (tar_episode_timesteps >= 1000) | tar_done

                if 'antmaze' in args.env:
                    tar_reward -= 1.0

                tar_mb_replay_buffer.add_batch_sep(tar_state, tar_action, tar_next_state, tar_reward.reshape(len(tar_reward),1), \
                                         tar_done_bool.reshape(len(tar_reward),1))

                tar_state = tar_next_state
                tar_episode_reward += tar_reward
                t += args.num_envs
                    

                for i, done in enumerate(tar_done_bool):
                    if done:
                        t_target_rollout_trajectories += 1
                        # tar_state[i] = tar_env.reset_one(i)
                        # (tar_state[i],_) = tar_env.env_method("reset", indices=[i])[0]
                        tar_state[i] = tar_env.reset(i)
                        print("Total T: {} Episode Num: {} Episode T: {} Target Reward: {}".format(t+1, tar_episode_num+1, \
                                                                                                   tar_episode_timesteps[i], tar_episode_reward[i]))
                        
                        tar_episode_reward[i] = 0
                        tar_episode_timesteps[i] = 0
                        tar_episode_num += 1
            # from tqdm import tqdm
            # import time
            # start = time.time()
            for _ in range(args.num_envs):
                policy.train(src_replay_buffer, tar_replay_buffer,tar_mb_replay_buffer, config['batch_size'], writer)
      
                # print(policy.total_it)
            # print(time.time()-start)

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env,  policy.policy_darc, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env,  policy.policy,eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                # record normalized score
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.mode == 1:
        # offline-online learning
        tar_state, tar_done = tar_env.reset(), False
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            
            # interaction with tar env
            if t % config['tar_env_interact_interval'] == 0:
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), test=False)

                tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
                tar_done_bool = float(tar_done) if tar_episode_timesteps < src_eval_env._max_episode_steps else 0

                if 'antmaze' in args.env:
                    tar_reward -= 1.0

                tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

                tar_state = tar_next_state
                tar_episode_reward += tar_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if tar_done:
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, tar_episode_num+1, tar_episode_timesteps, tar_episode_reward))
                writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)
                train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                writer.add_scalar('train/target normalized score', train_normalized_score, global_step = t+1)

                tar_state, tar_done = tar_env.reset(), False
                tar_episode_reward = 0
                tar_episode_timesteps = 0
                tar_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, policy.policy_darc, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env,  policy.policy,eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.mode == 2:
        # online-offline learning
        src_state, src_done = src_env.reset(), False
        src_episode_reward, src_episode_timesteps, src_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            src_episode_timesteps += 1
            # print(src_state.shape)

            # select action randomly or according to policy, if the policy is deterministic, add exploration noise akin to TD3 implementation
            src_action = (
                policy.select_action(np.array(src_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            src_next_state, src_reward, src_done, _ = src_env.step(src_action) 
            src_done_bool = float(src_done) if src_episode_timesteps < src_env._max_episode_steps else 0

            if 'antmaze' in args.env:
                src_reward -= 1.0

            src_replay_buffer.add(src_state, src_action, src_next_state, src_reward, src_done_bool)

            src_state = src_next_state
            src_episode_reward += src_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if src_done: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, src_episode_num+1, src_episode_timesteps, src_episode_reward))
                writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

                src_state, src_done = src_env.reset(), False
                src_episode_reward = 0
                src_episode_timesteps = 0
                src_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env,  policy.policy_darc,eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env,  policy.policy,eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    else:
        # offline-offline learning
        name =  args.env + str(args.shift_level) + args.srctype  + args.tartype 
 
        wandbrun = wandb.init(
            entity="yihongguo-johns-hopkins-university",
            project="mbod",
            group = args.group,
            name = name,
            config=config
        )
        # print(c)

    
        # define model based dynamics
        scaler = utils.StandardScaler()
        # termination_fn = get_termination_fn(task=args.task)
        termination_fn = terminal_fn


        dynamics_model = MOBODYModule(
        obs_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dims=256,
        num_ensemble=7,
        num_elites=5,
        weight_decays=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4],
        device='cuda',
        reward_relu=args.relu_reward,config=config)
    
        dynamics_optim = torch.optim.Adam(
            dynamics_model.parameters(),
            lr=args.dynamics_lr,
        )
        dynamics = MOBODYEnsembleDynamics(
            config,
            dynamics_model,
            dynamics_optim,
            scaler,
            termination_fn,
            penalty_coef = env_penalty_coef
        )



        # train model based method
        
        src_batch = src_replay_buffer.sample_all(False)
        trg_batch = tar_replay_buffer.sample_all(False)
        if 'mb' in args.policy.lower() or 'mobody' in args.policy.lower():
            if args.dynamics_path is not None and args.train_dynamics == 0:
                # dynamics.load(args.dynamics_path)
                env_path = os.path.join(args.dynamics_path, args.env )
                save_path = env_path + '/srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) 
                
                if os.path.exists(save_path):
                    dynamics.load(save_path)
                else:
                    dynamics.train(src_batch,trg_batch, writer = writer, buffer = [src_replay_buffer, tar_replay_buffer])
                    if args.dynamics_path is not None:
                        if not os.path.exists(args.dynamics_path):
                            os.mkdir(args.dynamics_path)
                        env_path = os.path.join(args.dynamics_path, args.env + '/' )
                        if not os.path.exists(env_path):
                            os.mkdir(env_path)
                        save_path = env_path + '/srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) 

                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    dynamics.save(save_path)
            else:
                # try to load:
                env_path = 'pretrained_dynamics/' + args.env + '/' 
                save_path = env_path + 'srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) 
                if not os.path.exists(env_path):
                    os.mkdir(env_path)

                if os.path.exists(save_path) and args.train_dynamics == 0:
                    try:
                        dynamics.load(save_path)
                        print('----------pretrained dynamics loaded----------')
                    except:
                        dynamics.train(src_batch,trg_batch, writer = writer, buffer = [src_replay_buffer, tar_replay_buffer])
                        if args.dynamics_path is not None:
                            if not os.path.exists(args.dynamics_path):
                                os.mkdir(args.dynamics_path)
                            env_path = os.path.join(args.dynamics_path, args.env + '/' )
                            if not os.path.exists(env_path):
                                os.mkdir(env_path)
                            save_path = env_path + '/srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) 

                    
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        dynamics.save(save_path)
                else:
                    dynamics.train(src_batch,trg_batch, writer = writer, buffer = [src_replay_buffer, tar_replay_buffer])
                    if args.dynamics_path is not None:
                        if not os.path.exists(args.dynamics_path):
                            os.mkdir(args.dynamics_path)
                        env_path = os.path.join(args.dynamics_path, args.env + '/' )
                        if not os.path.exists(env_path):
                            os.mkdir(env_path)
                        save_path = env_path + '/srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) 

                    
  
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    dynamics.save(save_path)
        else:
            dynamics = None

            
        
        config.update({'dynamics': dynamics})

        policy.dynamics = dynamics

        # self.fake_replay_buffer = utils.ReplayBuffer(config['state_dim'], config['action_dim'], config['device'])

        # src_eval_env_vec =  VecEnv(src_eval_env, args.num_envs, args.seed)

        trg_eval_env_vec = []
        for jjj in range(10):
            envvv = call_env[domain](tar_env_config)
            envvv.seed(args.seed + 100 + jjj )
            trg_eval_env_vec.append(envvv)
        trg_eval_env_vec =  VecEnv(trg_eval_env_vec, args.num_envs, args.seed)


        src_eval_env_vec = []
        for jjj in range(10):
            # envvv = call_env[domain](src_env_config)
            envvv = gym.make(src_env_name)
            envvv.seed(args.seed + 100 + jjj )
            src_eval_env_vec.append(envvv)
        src_eval_env_vec =  VecEnv(src_eval_env_vec, args.num_envs, args.seed)

        
        # src_eval_return = eval_policy(policy, src_eval_env, policy.policy, eval_cnt=eval_cnt,eval_trg = False)
        # tar_eval_return = eval_policy(policy, tar_eval_env, policy.policy, eval_cnt=eval_cnt,dynamics = dynamics, eval_trg = True)

        src_eval_return = eval_policy_batch(policy, src_eval_env_vec, policy.policy, eval_cnt=eval_cnt,eval_trg = False)
        tar_eval_return = eval_policy_batch(policy, trg_eval_env_vec, policy.policy, eval_cnt=eval_cnt,dynamics = dynamics, eval_trg = True)

        # print(c)

        eval_target_return_list = []
        eval_target_normalized_score_list = []
        running_mean_window = 5
        
        from tqdm import tqdm
        import time
        # old_start = time.time()
        start = time.time()
        for t in range(int(config['max_step'])):
            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer, wandbrun)
            
            if (t + 1) % config['eval_freq'] == 0:
                evaluate_time = time.time() 

                # if (t + 1) % (config['eval_freq'] * 20)  == 0:
                #     src_eval_return = eval_policy(policy, src_eval_env, policy.policy, eval_cnt=eval_cnt,eval_trg = False)
                src_eval_return = eval_policy_batch(policy, src_eval_env_vec, policy.policy, eval_cnt=eval_cnt,eval_trg = False)


                # evaluate performance of transition model on target env
                if (t + 1) % (1 * config['eval_freq']) == 0:
                    eval_dynamics = dynamics
                else:
                    eval_dynamics = None
                # if (t + 1) % (config['eval_freq'] * 20)  == 0:
                #     tar_eval_return = eval_policy(policy, tar_eval_env, policy.policy, eval_cnt=eval_cnt,dynamics = eval_dynamics, eval_trg = True, )
                tar_eval_return = eval_policy_batch(policy, trg_eval_env_vec, policy.policy, eval_cnt=eval_cnt,dynamics = dynamics, eval_trg = True)

                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)

                eval_target_return_list.append(tar_eval_return)
                eval_target_normalized_score_list.append(eval_normalized_score)


                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                wandbrun.log({
                    'test/source return': src_eval_return,
                    'test/target return': tar_eval_return,
                    'test/target normalized score': eval_normalized_score, 

                    'test/target smooth return': np.mean(eval_target_return_list[-running_mean_window:]),
                    'test/target smooth normalized score': np.mean(eval_target_normalized_score_list[-running_mean_window:]), 
                    },step=t + 1)
                
                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
                print('evaluate time',time.time()-evaluate_time)
                print('train + evaluate',time.time()-start)
                start = time.time()
    writer.close()
