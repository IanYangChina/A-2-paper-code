import os

gridworld_params = {
    'size': 15,
    'max_episode_steps': 60,
}

manipulation_params = {
    'task': 'chest_pick_and_place',
    'gripper': 'parallel_jaw',
    'num_block': 1,
    'distance_threshold': 0.03,
    'max_episode_steps': 50,
    'task_decomposition': False,
    'binary_reward': True,
}

agent_params = {
    'hindsight': True,
    'her_sampling_strategy': 'future',
    'prioritised': False,
    'memory_capacity': int(1e6),
    'actor_learning_rate': 0.001,
    'critic_learning_rate': 0.001,
    'update_interval': 1,
    'batch_size': 128,
    'optimization_steps': 40,
    'tau': 0.05,
    'discount_factor': 0.98,
    'clip_value': 50,
    'observation_type': 'observation',
    'discard_time_limit': False,
    'terminate_on_achieve': True,
    'observation_normalization': True,

    # TA2 params
    'abstract_demonstration': False,
    'abstract_demonstration_eta': 0.75,
    'adaptive_exploration': False,
    'adaptive_exploration_tau': 0.3,

    # DQN params
    'e_greed_decay': 5e5,

    # DDPG params
    'Q_weight_decay': 0.0,
    'exploration_epsilon': 0.2,
    'exploration_sigma': 0.05,

    # SAC params
    'alpha': 1.0,
    'actor_update_interval': 2,
    'critic_target_update_interval': 2,

    'curriculum': False,

    'training_epochs': 51,
    'training_cycles': 50,
    'training_episodes': 16,
    'testing_gap': 1,
    'testing_episodes': 30,
    'saving_gap': 50,

    'cuda_device_id': 0
}