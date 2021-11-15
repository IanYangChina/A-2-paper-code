import os
import argparse
import agents.plot as plot
import matplotlib as mpl
import pybullet_multigoal_gym as pmg
from config_params import *
from agents.dqn_her_ta2 import DqnHerTA2
from agents.ddpg_her_ta2 import DdpgHerTA2
from agents.sac_her_ta2 import SacHerTA2
from gridworld.multigoal_doorkey import make_grid_world

parser = argparse.ArgumentParser()
parser.add_argument('--task', dest='task', type=str,
                    help='Name of the task, default: gridworld_15',
                    default='gridworld_15', choices=['gridworld_15', 'gridworld_25',
                                                     'block_stack', 'chest_push', 'chest_pick_and_place'])
parser.add_argument('--agent', dest='agent', type=str,
                    help='Name of the agent, default: dqn', default='dqn', choices=['dqn', 'sac', 'ddpg'])
parser.add_argument('--render', dest='render',
                    help='Whether to render the task, default: False', default=False, action='store_true')
parser.add_argument('--num-seeds', dest='num_seeds',
                    help='Number of seeds (runs), default: 1', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

parser.add_argument('--TA', dest='ta',
                    help='Whether to use task decomposition & abstract demonstrations, default: False',
                    default=False, action='store_true')
parser.add_argument('--TA2', dest='ta2',
                    help='Whether to use task decomposition, abstract demonstrations & adaptive exploration, default: False',
                    default=False, action='store_true')
parser.add_argument('--eta', dest='eta',
                    help='Proportion of demonstrated episodes, default: 0.75',
                    type=float, default=0.75, choices=[0.25, 0.50, 0.75, 1.00])
parser.add_argument('--tau', dest='tau',
                    help='Adaptive exploration update speed (a value of 1.0 means exact estimate instead of polyak), default: 0.3',
                    type=float, default=0.3, choices=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

if __name__ == '__main__':
    args = vars(parser.parse_args())
    print("Current task: %s" % (args['task']))

    # setup task params
    if 'gridworld' in args['task']:
        assert args['agent'] == 'dqn', "Please use DQN for the gridworld tasks"
        gridworld_params['size'] = int(args['task'][-2:])
        agent_params['training_epochs'] = 51
        agent_params['saving_gap'] = 50
    else:
        assert args['agent'] != 'dqn', "Please use SAC or DDPG for the manipulation tasks"
        manipulation_params['task'] = args['task']

        if manipulation_params['task'] == 'chest_pick_and_place':
            agent_params['training_epochs'] = 51
            agent_params['saving_gap'] = 50
            manipulation_params['distance_threshold'] = 0.03
            manipulation_params['num_block'] = 1

        elif manipulation_params['task'] == 'chest_push':
            agent_params['training_epochs'] = 31
            agent_params['saving_gap'] = 30
            manipulation_params['distance_threshold'] = 0.05
            manipulation_params['num_block'] = 1

        elif manipulation_params['task'] == 'block_stack':
            agent_params['training_epochs'] = 101
            agent_params['saving_gap'] = 50
            manipulation_params['distance_threshold'] = 0.03
            manipulation_params['num_block'] = 2
        else:
            raise ValueError("something goes wrong")

    # setup ta2 params
    if args['ta']:
        assert not args['ta2'], 'please pass only the --TA or the --TA2 flag to the script'
        manipulation_params['task_decomposition'] = True
        agent_params['abstract_demonstration'] = True
        agent_params['abstract_demonstration_eta'] = args['eta']
        agent_params['adaptive_exploration'] = False
        case_dir = 'ta_' + str(args['eta']) + 'eta'
    elif args['ta2']:
        manipulation_params['task_decomposition'] = True
        agent_params['abstract_demonstration'] = True
        agent_params['abstract_demonstration_eta'] = args['eta']
        agent_params['adaptive_exploration'] = True
        agent_params['adaptive_exploration_tau'] = args['tau']
        case_dir = 'ta2_' + str(args['eta']) + 'eta_' + str(args['tau']) + 'tau'
    else:
        manipulation_params['task_decomposition'] = False
        agent_params['abstract_demonstration'] = False
        agent_params['adaptive_exploration'] = False
        case_dir = 'vanilla'

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'result',
                        args['task']+'_'+args['agent'],
                        case_dir)

    # dqn and sac should work for all seeds
    # while ddpg is expected to learn nothing for some of the seeds
    seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
    seed_returns = []
    seed_success_rates = []

    for i in range(args['num_seeds']):

        if 'gridworld' in args['task']:
            if args['render']:
                mpl.use('TkAgg')
            env = make_grid_world(size=gridworld_params['size'],
                                  max_episode_steps=gridworld_params['max_episode_steps'])
        else:
            env = pmg.make_env(task=manipulation_params['task'],
                               gripper=manipulation_params['gripper'],
                               grip_informed_goal=True,
                               num_block=manipulation_params['num_block'],
                               distance_threshold=manipulation_params['distance_threshold'],
                               max_episode_steps=manipulation_params['max_episode_steps'],
                               task_decomposition=manipulation_params['task_decomposition'],
                               render=args['render'],
                               binary_reward=manipulation_params['binary_reward'])

        seed = seeds[i]
        seed_path = path + '/seed' + str(seed)

        if args['agent'] == 'dqn':
            agent = DqnHerTA2(algo_params=agent_params, env=env, path=seed_path, seed=seed)
        elif args['agent'] == 'sac':
            agent = SacHerTA2(algo_params=algo_params, env=env, path=seed_path, seed=seed)
        else:
            assert args['agent'] == 'ddpg'
            agent = DdpgHerTA2(algo_params=algo_params, env=env, path=seed_path, seed=seed)

        agent.run(test=False, render=args['render'])
        seed_returns.append(agent.statistic_dict['epoch_test_return'])
        seed_success_rates.append(agent.statistic_dict['epoch_test_success_rate'])

        del env, agent

    # only plot mean-deviation figures when 2 or more seeds have been run
    if args['num_seeds'] >= 2:
        return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                                       file_name=os.path.join(path, 'return_statistic.json'))
        plot.smoothed_plot_mean_deviation(path + '/returns', return_statistic, x_label='Epoch', y_label='Average returns')

        success_rate_statistic = plot.get_mean_and_deviation(seed_success_rates, save_data=True,
                                                             file_name=os.path.join(path, 'success_rate_statistic.json'))
        plot.smoothed_plot_mean_deviation(path + '/success_rates', success_rate_statistic,
                                          x_label='Epoch', y_label='Success rates')
