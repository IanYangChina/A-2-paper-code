import os
import argparse
import pybullet_multigoal_gym as pmg
from config_params import *
from agents.dqn_her_ta2 import DqnHerTA2
from agents.ddpg_her_ta2 import DdpgHerTA2
from agents.sac_her_ta2 import SacHerTA2
from gridworld.multigoal_doorkey import make_grid_world
import matplotlib as mpl

parser = argparse.ArgumentParser()
parser.add_argument('--task', dest='task', type=str,
                    help='Name of the task, default: gridworld_15',
                    default='gridworld_25', choices=['gridworld_15', 'gridworld_25',
                                                     'block_stack', 'chest_push', 'chest_pick_and_place'])
parser.add_argument('--agent', dest='agent', type=str,
                    help='Name of the agent, default: dqn', default='dqn', choices=['dqn', 'sac', 'ddpg'])
parser.add_argument('--render', dest='render',
                    help='Whether to render the task, default: False', default=False, action='store_true')

parser.add_argument('--TA', dest='ta',
                    help='Whether to use task decomposition & abstract demonstrations, default: False',
                    default=False, action='store_true')
parser.add_argument('--TA2', dest='ta2',
                    help='Whether to use task decomposition, abstract demonstrations & adaptive exploration, default: False',
                    default=False, action='store_true')

if __name__ == '__main__':
    args = vars(parser.parse_args())
    print("Current task: %s" % (args['task']))

    # setup task params
    if 'gridworld' in args['task']:
        assert args['agent'] == 'dqn', "Please use DQN for the gridworld tasks"
        if args['render']:
            mpl.use('TkAgg')
        gridworld_params['size'] = int(args['task'][-2:])
        load_network_ep = 50
    else:
        assert args['agent'] != 'dqn', "Please use SAC or DDPG for the manipulation tasks"
        manipulation_params['task'] = args['task']

        if manipulation_params['task'] == 'chest_push':
            load_network_ep = 30
            manipulation_params['distance_threshold'] = 0.05
            manipulation_params['num_block'] = 1

        elif manipulation_params['task'] == 'chest_pick_and_place':
            load_network_ep = 50
            manipulation_params['distance_threshold'] = 0.03
            manipulation_params['num_block'] = 1

        elif manipulation_params['task'] == 'block_stack':
            load_network_ep = 100
            manipulation_params['distance_threshold'] = 0.03
            manipulation_params['num_block'] = 2
        else:
            raise ValueError("Please make sure the task is in "
                             "['gridworld_15', 'gridworld_25', 'block_stack', 'chest_push', 'chest_pick_and_place'],"
                             "not {}.".format(manipulation_params['task']))

    # setup ta2 params
    if args['ta']:
        assert not args['ta2'], 'please pass only the --TA or the --TA2 flag to the script'
        case_dir = 'ta_0.75eta'
    elif args['ta2']:
        case_dir = 'ta2_0.75eta_0.3tau'
    else:
        case_dir = 'vanilla'

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'pretrained_agents',
                        args['task'] + '_' + args['agent'],
                        case_dir)

    if 'gridworld' in args['task']:
        sleep = 0
        env = make_grid_world(size=gridworld_params['size'],
                              max_episode_steps=gridworld_params['max_episode_steps'])
    else:
        sleep = 0.05
        env = pmg.make_env(task=manipulation_params['task'],
                           gripper=manipulation_params['gripper'],
                           grip_informed_goal=True,
                           num_block=manipulation_params['num_block'],
                           distance_threshold=manipulation_params['distance_threshold'],
                           max_episode_steps=manipulation_params['max_episode_steps'],
                           task_decomposition=True,
                           render=args['render'],
                           binary_reward=manipulation_params['binary_reward'])

    if args['agent'] == 'dqn':
        agent = DqnHerTA2(algo_params=agent_params, env=env, path=path)
    elif args['agent'] == 'sac':
        agent = SacHerTA2(algo_params=agent_params, env=env, path=path)
    else:
        assert args['agent'] == 'ddpg'
        agent = DdpgHerTA2(algo_params=agent_params, env=env, path=path)

    agent.run(test=True, render=args['render'], load_network_ep=load_network_ep, sleep=sleep)