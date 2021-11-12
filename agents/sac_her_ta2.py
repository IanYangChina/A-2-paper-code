import time
import os
from copy import deepcopy as dcp
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from drl_implementation.agent.utils.networks_mlp import StochasticActor, Critic
from drl_implementation.agent.agent_base import Agent
from drl_implementation.agent.utils.plot import smoothed_plot_multi_line
from agents.adaptive_exploration import PolyakAvgSuccessRate
from collections import namedtuple


class SacHerTA2(Agent):
    def __init__(self, algo_params, env, transition_tuple=None, path=None, seed=-1):
        # environment
        self.env = env
        self.env.seed(seed)
        obs = self.env.reset()
        self.observation_type = algo_params['observation_type']
        algo_params.update({'state_dim': obs['observation'].shape[0],
                            'goal_dim': obs['desired_goal'].shape[0],
                            'action_dim': self.env.action_space.shape[0],
                            'action_max': self.env.action_space.high,
                            'action_scaling': self.env.action_space.high[0],
                            'init_input_means': None,
                            'init_input_vars': None
                            })
        self.curriculum = False
        if 'curriculum' in algo_params.keys():
            self.curriculum = algo_params['curriculum']
        # training args
        self.training_epochs = algo_params['training_epochs']
        self.training_cycles = algo_params['training_cycles']
        self.training_episodes = algo_params['training_episodes']
        self.testing_gap = algo_params['testing_gap']
        self.testing_episodes = algo_params['testing_episodes']
        self.saving_gap = algo_params['saving_gap']

        super(SacHerTA2, self).__init__(algo_params,
                                        transition_tuple=namedtuple("transition",
                                                                    ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done')),
                                        goal_conditioned=True,
                                        path=path,
                                        seed=seed)
        # torch
        self.network_dict.update({
            'actor': StochasticActor(self.state_dim + self.goal_dim, self.action_dim, log_std_min=-6, log_std_max=1,
                                     action_scaling=self.action_scaling).to(self.device),
            'critic_1': Critic(self.state_dim + self.goal_dim + self.action_dim, 1).to(self.device),
            'critic_1_target': Critic(self.state_dim + self.goal_dim + self.action_dim, 1).to(self.device),
            'critic_2': Critic(self.state_dim + self.goal_dim + self.action_dim, 1).to(self.device),
            'critic_2_target': Critic(self.state_dim + self.goal_dim + self.action_dim, 1).to(self.device),
            'alpha': algo_params['alpha'],
            'alpha_train': T.tensor(algo_params['alpha'], requires_grad=True, device=self.device),
            'log_alpha': T.tensor(np.log(algo_params['alpha']), requires_grad=True, device=self.device),
        })
        self.network_keys_to_save = ['actor', 'critic_1_target']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.actor_learning_rate)
        self.critic_1_optimizer = Adam(self.network_dict['critic_1'].parameters(), lr=self.critic_learning_rate)
        self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'], tau=1)
        self.critic_2_optimizer = Adam(self.network_dict['critic_2'].parameters(), lr=self.critic_learning_rate)
        self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'], tau=1)
        self.target_entropy = -self.action_dim
        self.alpha_optimizer = Adam([self.network_dict['alpha_train']], lr=self.actor_learning_rate)
        self.adaptive_exploration = algo_params['adaptive_exploration']
        self.success_recorder = PolyakAvgSuccessRate(self.env.num_steps, tau=algo_params['adaptive_exploration_tau'])
        # training args
        self.clip_value = algo_params['clip_value']
        self.actor_update_interval = algo_params['actor_update_interval']
        self.critic_target_update_interval = algo_params['critic_target_update_interval']
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
            'epoch_std_scale': [],
            'alpha': [],
            'policy_entropy': [],
            'policy_log_probs': []
        })

    def run(self, test=False, render=False, load_network_ep=None, sleep=0, user_ctrl=False):
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
            for test_ep in range(self.testing_episodes):
                for goal_ind in self.env.step_demonstrator.demonstrations[-1]:
                    ep_test_return = self._interact(render, goal_ind=goal_ind, test=True, sleep=sleep, user_ctrl=user_ctrl)
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
                if self.curriculum:
                    self.env.activate_curriculum_update()
                for cyc in range(self.training_cycles):
                    cycle_return = 0
                    cycle_success = 0
                    self.activate_demonstration = False
                    for ep in range(self.training_episodes):
                        if (ep >= (self.training_episodes - self.num_demo_episodes)) and self.abstract_demonstration:
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
                    self.activate_demonstration = True
                    if self.curriculum:
                        self.env.deactivate_curriculum_update()
                    # testing during training
                    test_return = 0
                    test_success = 0
                    test_sub_goal_success = np.zeros(self.env.num_steps)
                    for goal_ind in self.env.step_demonstrator.demonstrations[-1]:
                        for test_ep in range(self.testing_episodes):
                            ep_test_return = self._interact(render, goal_ind=goal_ind, test=True, sleep=sleep)
                            test_return += ep_test_return
                            if (ep_test_return + self.env._max_episode_steps) > 2:
                                test_sub_goal_success[goal_ind] += 1
                                test_success += 1
                    self.statistic_dict['epoch_test_return_demo'].append(test_return / (self.testing_episodes*self.env.num_steps))
                    self.statistic_dict['epoch_test_success_rate_demo'].append(test_success / (self.testing_episodes*self.env.num_steps))
                    self.statistic_dict['epoch_test_sub_goal_success_rate_demo'].append(
                        (test_sub_goal_success / self.testing_episodes).tolist())

                    if self.adaptive_exploration:
                        self.success_recorder.update_success_rates(
                            test_sub_goal_success / self.testing_episodes
                        )
                        self.statistic_dict['epoch_std_scale'].append((1-self.success_recorder.success_rates).tolist())
                        print("Epoch %i" % epo,
                              "subgoal std scale {}".format(self.statistic_dict['epoch_std_scale'][-1]))

                    print("Epoch %i" % epo, "test subgoal success demo {}".format(self.statistic_dict['epoch_test_sub_goal_success_rate_demo'][-1]))

                    self.activate_demonstration = False
                    if self.curriculum:
                        self.env.deactivate_curriculum_update()
                    # testing during training
                    test_return = 0
                    test_success = 0
                    test_sub_goal_success = np.zeros(self.env.num_steps)
                    for goal_ind in self.env.step_demonstrator.demonstrations[-1]:
                        for test_ep in range(self.testing_episodes):
                            ep_test_return = self._interact(render, goal_ind=goal_ind, test=True, sleep=sleep)
                            test_return += ep_test_return
                            if (ep_test_return + self.env._max_episode_steps) > 2:
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
                self.statistic_dict['epoch_std_scale'] = np.transpose(self.statistic_dict['epoch_std_scale'])
                smoothed_plot_multi_line(os.path.join(self.path, 'epoch_std_scale.png'),
                                         self.statistic_dict['epoch_std_scale'],
                                         x_label='Epoch', y_label='Action standard deviation', window=5)

            self._plot_statistics(
                keys=['critic_loss', 'actor_loss',
                      'cycle_return', 'cycle_success_rate',
                      'epoch_test_return_demo', 'epoch_test_success_rate_demo',
                      'epoch_test_return', 'epoch_test_success_rate',
                      'alpha', 'policy_entropy', 'policy_log_probs'],
                x_labels={
                    'critic_loss': 'Optimization epoch (per ' + str(self.optimizer_steps) + ' steps)',
                    'actor_loss': 'Optimization epoch (per ' + str(self.optimizer_steps) + ' steps)',
                    'alpha': 'Optimization epoch (per ' + str(self.optimizer_steps) + ' steps)',
                    'policy_entropy': 'Optimization epoch (per ' + str(self.optimizer_steps) + ' steps)',
                    'policy_log_probs': 'Optimization epoch (per ' + str(self.optimizer_steps) + ' steps)'
                },
                save_to_file=False)

    def _interact(self, render=False, test=False, goal_ind=-1, sleep=0, user_ctrl=False):
        if goal_ind == -1:
            goal_ind = len(self.env.step_demonstrator.demonstrations[-1])-1
        time_done = False
        obs = self.env.reset(test=test)
        if self.curriculum:
            self.env._max_episode_steps = self.env.env.curriculum_goal_step
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
        while not time_done:
            if render:
                self.env.render()
            action = self._select_action(obs, goal_ind=goal_ind, test=test)
            if user_ctrl:
                user_action = input("Type one of the following actions: "
                                    "rl:use RL actions, "
                                    "x+, x-, y+, y-, z+, z-, "
                                    "g:close finger, f:open finger")
                if user_action != 'rl':
                    action = np.zeros((self.action_dim,))
                    if user_action == 'x+':
                        action[0] += 0.2
                    if user_action == 'x-':
                        action[0] -= 0.2
                    if user_action == 'y+':
                        action[1] += 0.2
                    if user_action == 'y-':
                        action[1] -= 0.2
                    if user_action == 'z+':
                        action[2] += 0.2
                    if user_action == 'z-':
                        action[2] -= 0.2
                    if user_action == 'f':
                        action[-1] = -1.0
                    if user_action == 'g':
                        action[-1] = 1.0

            new_obs, reward, time_done, info = self.env.step(action)
            time.sleep(sleep)
            ep_return += reward
            if not test:
                trajectory.append(
                    [obs[self.observation_type], obs['desired_goal'], action,
                     new_obs[self.observation_type], new_obs['achieved_goal'], reward, 1 - int(info['goal_achieved'])]
                )
                if self.observation_normalization:
                    self.normalizer.store_history(np.concatenate((new_obs[self.observation_type],
                                                                  new_obs['achieved_goal']), axis=0))
            obs = new_obs

            if self.activate_demonstration and info['goal_achieved'] and (not self.env.step_demonstrator.final):
                ep_return -= 1
                goal_ind = self.env.step_demonstrator.get_next_goal()
                obs['desired_goal'] = self.env.set_sub_goal(goal_ind)
                if not test:
                    # store the previous trajectory
                    trajectory_list.append(dcp(trajectory))
                    # replace goal & recompute reward
                    for transition in trajectory:
                        transition[1] = obs['desired_goal'].copy()
                        transition[5], goal_achieved = self.env.env._compute_reward(transition[4], obs['desired_goal'])
                        transition[6] = 1 - int(goal_achieved)

        if not test:
            trajectory_list.append(trajectory)
            # store all trajectories
            for trj in trajectory_list:
                new_trj = True
                for trs in trj:
                    self._remember(*tuple(trs), new_episode=new_trj)
                    new_trj = False

            self.normalizer.update_mean()
            self._learn()

        return ep_return

    def _select_action(self, obs, goal_ind=-1, test=False):
        inputs = np.concatenate((obs['observation'], obs['desired_goal']), axis=0)
        inputs = self.normalizer(inputs)
        inputs = T.as_tensor(inputs, dtype=T.float).to(self.device)
        return self.network_dict['actor'].get_action(inputs, std_scale=(1-self.success_recorder(goal_ind)),
                                                     mean_pi=test).detach().cpu().numpy()

    def _learn(self, steps=None):
        if self.hindsight:
            self.buffer.modify_episodes()
        self.buffer.store_episodes()
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        critic_losses = T.zeros(1, device=self.device)
        actor_losses = T.zeros(1, device=self.device)
        alphas = T.zeros(1, device=self.device)
        policy_entropies = T.zeros(1, device=self.device)
        policy_log_probs = T.zeros(1, device=self.device)
        for i in range(steps):
            if self.prioritised:
                batch, weights, inds = self.buffer.sample(self.batch_size)
                weights = T.as_tensor(weights).view(self.batch_size, 1).to(self.device)
            else:
                batch = self.buffer.sample(self.batch_size)
                weights = T.ones(size=(self.batch_size, 1)).to(self.device)
                inds = None

            actor_inputs = np.concatenate((batch.state, batch.desired_goal), axis=1)
            actor_inputs = self.normalizer(actor_inputs)
            actor_inputs = T.as_tensor(actor_inputs, dtype=T.float32, device=self.device)
            actions = T.as_tensor(batch.action, dtype=T.float32, device=self.device)
            critic_inputs = T.cat((actor_inputs, actions), dim=1)
            actor_inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
            actor_inputs_ = self.normalizer(actor_inputs_)
            actor_inputs_ = T.as_tensor(actor_inputs_, dtype=T.float32, device=self.device)
            rewards = T.as_tensor(batch.reward, dtype=T.float32, device=self.device).unsqueeze(1)
            done = T.as_tensor(batch.done, dtype=T.float32, device=self.device).unsqueeze(1)

            with T.no_grad():
                actions_, log_probs_ = self.network_dict['actor'].get_action(actor_inputs_, probs=True)
                critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1)
                value_1_ = self.network_dict['critic_1_target'](critic_inputs_)
                value_2_ = self.network_dict['critic_2_target'](critic_inputs_)
                value_ = T.min(value_1_, value_2_) - (self.network_dict['alpha'] * log_probs_)
                value_target = rewards + done * self.gamma * value_
                value_target = T.clamp(value_target, -self.clip_value, 0.0)

            self.critic_1_optimizer.zero_grad()
            value_estimate_1 = self.network_dict['critic_1'](critic_inputs)
            critic_loss_1 = F.mse_loss(value_estimate_1, value_target.detach(), reduction='none')
            (critic_loss_1 * weights).mean().backward()
            self.critic_1_optimizer.step()

            if self.prioritised:
                assert inds is not None
                self.buffer.update_priority(inds, np.abs(critic_loss_1.cpu().detach().numpy()))

            self.critic_2_optimizer.zero_grad()
            value_estimate_2 = self.network_dict['critic_2'](critic_inputs)
            critic_loss_2 = F.mse_loss(value_estimate_2, value_target.detach(), reduction='none')
            (critic_loss_2 * weights).mean().backward()
            self.critic_2_optimizer.step()

            critic_losses += critic_loss_1.detach().mean()

            if self.optim_step_count % self.critic_target_update_interval == 0:
                self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'])
                self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'])

            if self.optim_step_count % self.actor_update_interval == 0:
                self.actor_optimizer.zero_grad()
                new_actions, new_log_probs, entropy = self.network_dict['actor'].get_action(actor_inputs, probs=True,
                                                                                            entropy=True)
                critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
                new_values = T.min(self.network_dict['critic_1'](critic_eval_inputs),
                                   self.network_dict['critic_2'](critic_eval_inputs))
                actor_loss = (self.network_dict['alpha'] * new_log_probs - new_values).mean()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.alpha_optimizer.zero_grad()
                alpha_loss = (self.network_dict['alpha_train'] * (-new_log_probs - self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.network_dict['alpha'] = self.network_dict['alpha_train'].detach()

                alphas += self.network_dict['alpha'].detach()
                actor_losses += actor_loss.detach()
                policy_entropies += entropy.detach().mean()
                policy_log_probs += new_log_probs.detach().mean()

            self.optim_step_count += 1

        self.statistic_dict['critic_loss'].append(critic_losses / steps)
        self.statistic_dict['actor_loss'].append(actor_losses / steps)
        self.statistic_dict['alpha'].append(alphas / steps)
        self.statistic_dict['policy_entropy'].append(policy_entropies / steps)
        self.statistic_dict['policy_log_probs'].append(policy_log_probs / steps)
