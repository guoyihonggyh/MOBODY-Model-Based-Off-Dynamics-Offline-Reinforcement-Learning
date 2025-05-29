from collections import defaultdict
from copy import deepcopy
import numpy as np


class VecEnv:
    def __init__(self,env,num_envs, seed = None):
        # env_list = []

        # for i in range(num_envs):
        #     print(env)
        #     env_list.append(deepcopy(env))

        # if seed is not None:
        #     for i in range(num_envs):
        #         env_list[i].seed(seed)
        #         env_list[i].action_space.seed(seed)          
        self.Env = env
    def step(self, action):
        rollout_transitions = defaultdict(list)
        for i in range(len(self.Env)):
            next_observations, rewards, terminals, _ = self.Env[i].step(action[i])

            rollout_transitions["next_obss"].append(next_observations.reshape(1,len(next_observations)))
            rollout_transitions["actions"].append(action[i])
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)
        
        rollout_transitions["next_obss"] = np.concatenate(rollout_transitions["next_obss"], axis=0)
        
        rollout_transitions["rewards"] = np.array(rollout_transitions["rewards"])
        rollout_transitions["terminals"] = np.array(rollout_transitions["terminals"])

        return rollout_transitions["next_obss"],  rollout_transitions["rewards"], rollout_transitions["terminals"], None

    def reset(self, index = None):
        states = []
        if index is None:
            for i in range(len(self.Env)):
                state=self.Env[i].reset()
                states.append(state.reshape(1,len(state)))
            return np.concatenate(states, axis=0)
        else:
            state = self.Env[index].reset()
            return state.reshape(1,len(state))
            
        