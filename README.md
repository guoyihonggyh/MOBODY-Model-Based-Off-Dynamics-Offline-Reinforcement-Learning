# MOBODY: Model Based Off-Dynamics Offline Reinforcement Learning

### Abstract

We study the off-dynamics offline reinforcement learning (RL) problem, where the goal is to
learn a policy from offline datasets collected from source and target domains with mismatched
transition dynamics. Existing off-dynamics offline RL methods typically either filter source
transitions that resemble those of the target domain or apply reward augmentation to source
data, both constrained by the limited transitions available from the target domain. As a result,
the learned policy is unable to explore target domain beyond the offline datasets. We propose
MOBODY, a Model-Based Off-Dynamics offline RL algorithm that addresses this limitation
by enabling exploration of the target domain via learned dynamics. MOBODY generates new
synthetic transitions in the target domain through model rollouts, which are used as data
augmentation during offline policy learning. Unlike existing model-based methods that learn
dynamics from a single domain, MOBODY tackles the challenge of mismatched dynamics by
leveraging both source and target datasets. Directly merging these datasets can bias the learned
model toward source dynamics. Instead, MOBODY learns target dynamics by discovering a
shared latent representation of states and transitions across domains through representation
learning. To stabilize training, MOBODY incorporates a behavior cloning loss that regularizes
the policy. Specifically, we introduce a Q-weighted behavior cloning loss that regularizes the policy
toward actions with high target-domain Q-values, rather than uniformly imitating all actions in
the dataset. These Q-values are learned from an enhanced target dataset composed of offline
target data, augmented source data, and rollout data from the learned target dynamics. We
evaluate MOBODY on standard MuJoCo benchmarks and show that it significantly outperforms
state-of-the-art baselines, with especially pronounced improvements in challenging scenarios
where existing methods struggle.

Install Package
```bash
conda create -n offdynamics python=3.8.13 && conda activate offdynamics
pip install setuptools==63.2.0
pip install wheel==0.38.4
pip install -r requirement.txt
```

Run Experiments
```bash
CUDA_VISIBLE_DEVICES=1 python -u train_mobody.py --policy MOBODY --env walker2d-friction --shift_level 2.0 --seed 1 --dir runs --train_dynamics 1 --penalty_type dara --env_penalty_coef 5 --src_rollout_length 1 --trg_rollout_length 1 --bc_coef 1 --wandb 0  &
```
Baselines
```bash
CUDA_VISIBLE_DEVICES=1 python -u train_mobody.py --policy DARA --env walker2d-friction --shift_level 2.0 --seed 1 --dir runs --train_dynamics 0 --penalty_type dara --wandb 0  &
```
