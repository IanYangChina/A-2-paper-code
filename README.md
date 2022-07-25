### Abstract Demonstrations and Adaptive Exploration for Efficient and Stable Multi-step Sparse Reward Reinforcement Learning

#### This is the official repository for the codes of the A^2 paper.

##### Paper link: https://arxiv.org/abs/2207.09243

##### Video:

##### Contact: [yangx66@cardiff.ac.uk](yangx66@cardiff.ac.uk)

#### Installation

- Ubuntu 16.04 or Windows 10
- Clone and enter the repository, install requirements
  - `git clone https://github.com/IanYangChina/TA2-paper-code`
  - `cd TA2-paper-code`
  - `python -m install -r requirements.txt`
- Install [Gym-minigrid](https://github.com/maximecb/gym-minigrid)
- Install [Pybullet-multigoal-gym](https://github.com/IanYangChina/pybullet_multigoal_gym)
- Install [DRL_Implementation](https://github.com/IanYangChina/DRL_Implementation)

#### Train

To train your own agent:
- Optionally `conda activate YourCondaEnvironment`
- `python train.py --task gridworld_15 --agent dqn --render --num-seeds 2 --TA2 --eta 0.75 --tau 0.3`
- This command means you will train a DQN agent on a gridworld task of size 15x15, 
with 75% demonstrated episodes and 0.3 adaptive exploration update speed 
for 2 random seeds in rendering mode.
- The `--TA` and `--TA2` flags should not be used at the same time.

To see full argument explanation: `python train.py -h`, which gives:
```
usage: run.py [-h]
              [--task {gridworld_15,gridworld_25,block_stack,chest_push,chest_pick_and_place}]
              [--agent {dqn,sac,ddpg}] [--train] [--render]
              [--num-seeds {1,2,3,4,5,6,7,8,9,10}] [--TA] [--TA2]
              [--eta {0.25,0.5,0.75,1.0}] [--tau {0.1,0.3,0.5,0.7,0.9,1.0}]

optional arguments:
  -h, --help            show this help message and exit
  --task {gridworld_15,gridworld_25,block_stack,chest_push,chest_pick_and_place}
                        Name of the task, default: gridworld_15
  --agent {dqn,sac,ddpg}
                        Name of the agent, default: dqn
  --train               Whether to train or evaluate, default: False
  --render              Whether to render the task, default: False
  --num-seeds {1,2,3,4,5,6,7,8,9,10}
                        Number of seeds (runs), default: 1
  --TA                  Whether to use task decomposition & abstract
                        demonstrations, default: False
  --TA2                 Whether to use task decomposition, abstract
                        demonstrations & adaptive exploration, default: False
  --eta {0.25,0.5,0.75,1.0}
                        Proportion of demonstrated episodes, default: 0.75
  --tau {0.1,0.3,0.5,0.7,0.9,1.0}
                        Adaptive exploration update speed (a value of 1.0
                        means exact estimate instead of polyak), default: 0.3
```

#### Test a pretrained agent

To evaluate a pretrained agent:
- `python test.py --task gridworld_15 --agent dqn --render --TA2`
- This command means a DQN agent pretrained using TA2 on the gridworld 15x15 task 
will be evaluated in rendering mode.
- Each of the subgoals will be evaluated for 30 episodes.
- The `--TA` and `--TA2` flags should not be used at the same time.

To see full argument explanation: `python test.py -h`, which gives:
```
usage: test.py [-h]
               [--task {gridworld_15,gridworld_25,block_stack,chest_push,chest_pick_and_place}]
               [--agent {dqn,sac,ddpg}] [--render] [--TA] [--TA2]

optional arguments:
  -h, --help            show this help message and exit
  --task {gridworld_15,gridworld_25,block_stack,chest_push,chest_pick_and_place}
                        Name of the task, default: gridworld_15
  --agent {dqn,sac,ddpg}
                        Name of the agent, default: dqn
  --render              Whether to render the task, default: False
  --TA                  Whether to use task decomposition & abstract
                        demonstrations, default: False
  --TA2                 Whether to use task decomposition, abstract
                        demonstrations & adaptive exploration, default: False

```
#### Citation

#### Acknowledgement
