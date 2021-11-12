import time
import os
from copy import deepcopy as dcp
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from drl_implementation.agent.utils.networks_mlp import Critic
from drl_implementation.agent.agent_base import Agent
from drl_implementation.agent.utils.plot import smoothed_plot_multi_line
from agents.adaptive_exploration import ExpDecayEpsilonGreedy, AdaptiveEpsilonGreedy


class DqnHerTA2(Agent):
    def __init__(self, algo_params, env, transition_tuple=None, path=None, seed=-1):
        # environment
        self.env = env
        self.env.seed(seed)
        obs = self.env.reset()
        algo_params.update({'state_dim': obs['observation'].shape[0],
                            'goal_dim': obs['desired_goal'].shape[0],
                            'action_dim': self.env.action_space.n,
                            'init_input_means': None,
                            'init_input_vars': None
                            })

        # training args
        self.training_epochs = algo_params['training_epochs']
        self.training_cycles = algo_params['training_cycles']
        self.training_episodes = algo_params['training_episodes']
        self.testing_gap = algo_params['testing_gap']
        self.testing_episodes = algo_params['testing_episodes']
        self.saving_gap = algo_params['saving_gap']

        super(DqnHerTA2, self).__init__(algo_params,
                                        action_type='discrete',
                                        transition_tuple=transition_tuple,
                                        goal_conditioned=True,
                                        path=path,
                                        seed=seed)

        # torch
        self.network_dict.update({
            'q': Critic(self.state_dim + self.goal_dim, self.action_dim, fc1_size=64, fc2_size=128, fc3_size=64).to(self.device),
            'q_target': Critic(self.state_dim + self.goal_dim, self.action_dim, fc1_size=64, fc2_size=128, fc3_size=64).to(self.device),
        })
        self.network_keys_to_save = ['q', 'q_target']
        self.q_optimizer = Adam(self.network_dict['q'].parameters(), lr=self.critic_learning_rate)
        self._soft_update(self.network_dict['q'], self.network_dict['q_target'], tau=1)

        self.adaptive_exploration = algo_params['adaptive_exploration']
        if not self.adaptive_exploration:
            self.exploration_strategy = ExpDecayEpsilonGreedy(decay=algo_params['e_greed_decay'], rng=self.rng)
        else:
            self.exploration_strategy = AdaptiveEpsilonGreedy(goal_num=self.env.num_steps,
                                                              tau=algo_params['adaptive_exploration_tau'],
                                                              rng=self.rng)
        # training args
        self.clip_value = algo_params['clip_value']
        self.abstract_demonstration = algo_params['abstract_demonstration']
        self.activate_demonstration = False
        self.num_demo_episodes = int(algo_params['abstract_demonstration_eta'] * self.training_episodes)
        # statistic dict
        self.statistic_dict.update({
            'cycle_return': [],
            'cycle_success_rate': [],
            'epoch_test_return': [],
            'epoch_test_success_rate': [],
            'epoch_test_sub_goal_success_rate': [],
            'epoch_test_return_demo': [],
            'epoch_test_success_rate_demo': [],
            'epoch_test_sub_goal_success_rate_demo': [],
            'epoch_adaptive_epsilon': []
        })

    def run(self, test=False, render=False, load_network_ep=None, sleep=0):
        # training setup uses a hierarchy of Epoch, Cycle and Episode
        #   following the HER paper: https://papers.nips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html
        if test:
            if load_network_ep is not None:
                print("Loading network parameters...")
                self._load_network(ep=load_network_ep)
            print("Start testing...")
            self.activate_demonstration = False
            test_return = 0
            test_success = 0
            test_sub_goal_success = np.zeros(self.env.num_steps)
            for goal_ind in self.env.step_demonstrator.demonstrations[0]:
                for test_ep in range(self.testing_episodes):
                    ep_test_return = self._interact(render, goal_ind=goal_ind, test=True, sleep=sleep)
                    test_return += ep_test_return
                    if ep_test_return > -self.env._max_episode_steps:
                        test_sub_goal_success[goal_ind] += 1
                        test_success += 1
            print("Finished testing")
            print("Return: {}, Avg success: {}".format(test_return / (self.testing_episodes * self.env.num_steps),
                                                       test_success / self.testing_episodes * self.env.num_steps))
            print("Subgoal avg success: ", test_sub_goal_success / self.testing_episodes)
        else:
            print("Start training...")

            for epo in range(self.training_epochs):
                for cyc in range(self.training_cycles):
                    cycle_return = 0
                    cycle_success = 0
                    self.activate_demonstration = False
                    for ep in range(self.training_episodes):
                        if ep >= (self.training_episodes - self.num_demo_episodes):
                            self.activate_demonstration = True
                        ep_return = self._interact(render, test=False, sleep=sleep)
                        cycle_return += ep_return
                        if ep_return > -self.env._max_episode_steps:
                            cycle_success += 1

                    self.statistic_dict['cycle_return'].append(cycle_return / self.training_episodes)
                    self.statistic_dict['cycle_success_rate'].append(cycle_success / self.training_episodes)
                    print("Epoch %i" % epo, "Cycle %i" % cyc,
                          "avg. return %0.1f" % (cycle_return / self.training_episodes),
                          "success rate %0.1f" % (cycle_success / self.training_episodes))

                if epo % self.testing_gap == 0:
                    if self.abstract_demonstration:
                        self.activate_demonstration = True
                        # testing during training
                        test_return = 0
                        test_success = 0
                        test_sub_goal_success = np.zeros(self.env.num_steps)
                        for goal_ind in self.env.step_demonstrator.demonstrations[-1]:
                            for test_ep in range(self.testing_episodes):
                                ep_test_return = self._interact(render, goal_ind=goal_ind, test=True, sleep=sleep)
                                test_return += ep_test_return
                                if ep_test_return > -self.env._max_episode_steps:
                                    test_sub_goal_success[goal_ind] += 1
                                    test_success += 1
                        self.statistic_dict['epoch_test_return_demo'].append(test_return / (self.testing_episodes*self.env.num_steps))
                        self.statistic_dict['epoch_test_success_rate_demo'].append(test_success / (self.testing_episodes*self.env.num_steps))
                        self.statistic_dict['epoch_test_sub_goal_success_rate_demo'].append(
                            (test_sub_goal_success / self.testing_episodes).tolist())
                        if self.adaptive_exploration:
                            self.exploration_strategy.update_success_rates(
                                test_sub_goal_success / self.testing_episodes
                            )
                            self.statistic_dict['epoch_adaptive_epsilon'].append(self.exploration_strategy.epsilon.tolist())
                            print("Epoch %i" % epo, "subgoal exploration epsilons {}".format(self.exploration_strategy.epsilon))

                        print("Epoch %i" % epo, "test subgoal success demo {}".format(self.statistic_dict['epoch_test_sub_goal_success_rate_demo'][-1]))

                    self.activate_demonstration = False
                    # testing during training
                    test_return = 0
                    test_success = 0
                    test_sub_goal_success = np.zeros(self.env.num_steps)
                    for goal_ind in self.env.step_demonstrator.demonstrations[-1]:
                        for test_ep in range(self.testing_episodes):
                            ep_test_return = self._interact(render, goal_ind=goal_ind, test=True, sleep=sleep)
                            test_return += ep_test_return
                            if ep_test_return > -self.env._max_episode_steps:
                                test_sub_goal_success[goal_ind] += 1
                                test_success += 1
                    self.statistic_dict['epoch_test_return'].append(test_return / (self.testing_episodes*self.env.num_steps))
                    self.statistic_dict['epoch_test_success_rate'].append(test_success / (self.testing_episodes*self.env.num_steps))
                    self.statistic_dict['epoch_test_sub_goal_success_rate'].append(
                        (test_sub_goal_success / self.testing_episodes).tolist())

                    print("Epoch %i" % epo, "test subgoal success {}".format(self.statistic_dict['epoch_test_sub_goal_success_rate'][-1]))

                if (epo % self.saving_gap == 0) and (epo != 0):
                    self._save_network(ep=epo)

            print("Finished training")

            print("Saving statistics...")
            self._save_statistics()
            if self.abstract_demonstration:
                self.statistic_dict['epoch_test_sub_goal_success_rate_demo'] = np.transpose(self.statistic_dict['epoch_test_sub_goal_success_rate_demo'])
                smoothed_plot_multi_line(os.path.join(self.path, 'epoch_test_sub_goal_success_rate_demo.png'),
                                         self.statistic_dict['epoch_test_sub_goal_success_rate_demo'],
                                         x_label='Epoch', y_label='Success rate', window=5)

            self.statistic_dict['epoch_test_sub_goal_success_rate'] = np.transpose(self.statistic_dict['epoch_test_sub_goal_success_rate'])
            smoothed_plot_multi_line(os.path.join(self.path, 'epoch_test_sub_goal_success_rate.png'),
                                     self.statistic_dict['epoch_test_sub_goal_success_rate'],
                                     x_label='Epoch', y_label='Success rate', window=5)
            if self.adaptive_exploration:
                self.statistic_dict['epoch_adaptive_epsilon'] = np.transpose(self.statistic_dict['epoch_adaptive_epsilon'])
                smoothed_plot_multi_line(os.path.join(self.path, 'epoch_adaptive_epsilon.png'),
                                         self.statistic_dict['epoch_adaptive_epsilon'],
                                         x_label='Epoch', y_label='Epsilon', window=5)

            self._plot_statistics(
                keys=['critic_loss',
                      'cycle_return', 'cycle_success_rate',
                      'epoch_test_return', 'epoch_test_success_rate',
                      'epoch_test_return_demo', 'epoch_test_success_rate_demo'],
                x_labels={
                    'critic_loss': 'Optimization epoch (per ' + str(self.optimizer_steps) + ' steps)',
                },
                save_to_file=False)

    def _interact(self, render=False, test=False, goal_ind=-1, sleep=0):
        done = False
        obs = self.env.reset()
        if test:
            obs['desired_goal'] = self.env.set_sub_goal(goal_ind)

        if self.activate_demonstration:
            self.env.step_demonstrator.manual_reset(goal_ind)
            goal_ind = self.env.step_demonstrator.get_next_goal()
            obs['desired_goal'] = self.env.set_sub_goal(goal_ind)
        ep_return = 0
        trajectory_list = []
        trajectory = []
        # start a new episode
        while not done:
            if render:
                self.env.render()
            action = self._select_action(obs, goal_ind=goal_ind, test=test)
            new_obs, reward, done, info = self.env.step(action)
            ep_return += reward
            if not test:
                trajectory.append(
                    [obs['observation'], obs['desired_goal'], action,
                     new_obs['observation'], new_obs['achieved_goal'], reward, 1 - int(done)]
                )
            obs = new_obs
            if not test:
                self.env_step_count += 1
                if self.activate_demonstration and info['goal_achieved'] and (not self.env.step_demonstrator.final):
                    ep_return -= 1
                    goal_ind = self.env.step_demonstrator.get_next_goal()
                    obs['desired_goal'] = self.env.set_sub_goal(goal_ind)
                    # store the previous trajectory
                    trajectory_list.append(dcp(trajectory))
                    # replace goal & recompute reward
                    for transition in trajectory:
                        transition[1] = obs['desired_goal'].copy()
                        transition[5], _ = self.env.env._compute_reward(transition[4], obs['desired_goal'])

        if not test:
            trajectory_list.append(trajectory)
            # store all trajectories
            for trj in trajectory_list:
                new_trj = True
                for trs in trj:
                    self._remember(*tuple(trs), new_episode=new_trj)
                    new_trj = False

            self._learn()

        return ep_return

    def _select_action(self, obs, goal_ind=-1, test=False):
        inputs = np.concatenate((obs['observation'], obs['desired_goal']), axis=0)
        if self.adaptive_exploration:
            if self.exploration_strategy(goal_ind):
                action = self.env.action_space.sample()
            else:
                with T.no_grad():
                    inputs = T.as_tensor(inputs, dtype=T.float, device=self.device)
                    values = self.network_dict['q_target'](inputs).cpu().detach()
                    action = T.argmax(values).item()
        else:
            if self.exploration_strategy(self.env_step_count):
                action = self.env.action_space.sample()
            else:
                with T.no_grad():
                    inputs = T.as_tensor(inputs, dtype=T.float, device=self.device)
                    values = self.network_dict['q_target'](inputs).cpu().detach()
                    action = T.argmax(values).item()
        return action

    def _learn(self, steps=None):
        if self.hindsight:
            self.buffer.modify_episodes()
        self.buffer.store_episodes()
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        critic_losses = T.zeros(1, device=self.device)
        for i in range(steps):
            if self.prioritised:
                batch, weights, inds = self.buffer.sample(self.batch_size)
                weights = T.as_tensor(weights).view(self.batch_size, 1).to(self.device)
            else:
                batch = self.buffer.sample(self.batch_size)
                weights = T.ones(size=(self.batch_size, 1)).to(self.device)
                inds = None

            inputs = np.concatenate((batch.state, batch.desired_goal), axis=1)
            inputs = T.as_tensor(inputs, dtype=T.float32, device=self.device)
            actions = T.as_tensor(batch.action, dtype=T.long, device=self.device).unsqueeze(1)
            inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
            inputs_ = T.as_tensor(inputs_, dtype=T.float32, device=self.device)
            rewards = T.as_tensor(batch.reward, dtype=T.float32, device=self.device).unsqueeze(1)
            done = T.as_tensor(batch.done, dtype=T.float32, device=self.device).unsqueeze(1)

            if self.discard_time_limit:
                done = done * 0 + 1

            with T.no_grad():
                maximal_next_values = self.network_dict['q_target'](inputs_).max(1)[0].view(self.batch_size, 1)
                value_target = rewards + done*self.gamma*maximal_next_values
                value_target = T.clamp(value_target, -self.clip_value, 0.0)

            self.q_optimizer.zero_grad()
            value_estimate = self.network_dict['q'](inputs).gather(1, actions)
            loss = F.smooth_l1_loss(value_estimate, value_target.detach(), reduction='none')
            (loss * weights).mean().backward()
            self.q_optimizer.step()

            if self.prioritised:
                assert inds is not None
                self.buffer.update_priority(inds, np.abs(loss.cpu().detach().numpy()))

            self._soft_update(self.network_dict['q'], self.network_dict['q_target'])

            self.optim_step_count += 1
            critic_losses += loss.detach().mean()

        self.statistic_dict['critic_loss'].append(critic_losses / steps)
