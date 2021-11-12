from gym.envs.registration import register, make, registry
from gym_minigrid.minigrid import *
from pybullet_multigoal_gym.utils.demonstrator import StepDemonstrator


class MGDoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8, max_steps=50):
        self.size = size
        self.num_steps = 3
        self.step_demonstrator = StepDemonstrator([
            [0],
            [0, 1],
            [0, 1, 2]
        ])
        self.sub_goals = {
            'key': None,
            'door': None,
            'goal': None
        }
        self.desired_sub_goal_name = 'goal'
        self.sub_goal_names = ['key', 'door', 'goal']
        super().__init__(
            grid_size=size,
            max_steps=max_steps
        )
        self.action_space = spaces.Discrete(3)
        self.distance_threshold = 0.0001

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal at a random position in the right room
        self.sub_goals['goal'] = np.array([self._rand_int(width//2+1, width - 2), self._rand_int(2, height - 2)])
        self.put_obj(Goal(), self.sub_goals['goal'][0], self.sub_goals['goal'][1])

        # Create a vertical splitting wall
        splitIdx = width//2
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        key_pos = self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.sub_goals['key'] = key_pos
        self.sub_goals['door'] = np.array([splitIdx, doorIdx])
        self.desired_sub_goal_name = 'goal'
        self.mission = "use the key to open the door and then get to the goal"

    def reset(self):
        obs = super().reset()
        obs = self._get_state()
        return obs

    def step(self, action):
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell_obj = self.grid.get(*fwd_pos)
        # pick up key and open door automatically
        if fwd_cell_obj is not None:
            if fwd_cell_obj.type == 'door' and action == 2:
                fwd_cell_obj.toggle(self, fwd_pos)
            elif fwd_cell_obj.type == 'key' and action == 2:
                self.carrying = fwd_cell_obj
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(*fwd_pos, None)

        _, reward, done, info = super().step(action)
        obs = self._get_state()
        reward, goal_achieved = self._compute_reward(obs['achieved_goal'], obs['desired_goal'])
        info.update({
            'goal_achieved': goal_achieved
        })
        return obs, reward, done, info

    def _compute_reward(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        not_achieved = (d > self.distance_threshold)
        return -not_achieved.astype(np.float32), ~not_achieved

    def set_sub_goal(self, ind):
        self.desired_sub_goal_name = self.sub_goal_names[ind]
        return self.sub_goals[self.desired_sub_goal_name].copy() / self.size

    def _get_state(self):
        # observation consists of:
        #   agent's xy and heading direction
        #   whether the agent is carrying the key (0 = False, 1 = True)
        #   whether the door is opened (0 = False, 1 = True)
        #   door's xy and key's xy
        # desired goals and achieved goals are coordinates
        # all coordinates are normalized using the size of the grid world
        return {
            'observation': np.concatenate([self.agent_pos.copy() / self.size,
                                           [self.agent_dir],
                                           [np.asarray(isinstance(self.carrying, Key), dtype=float)],
                                           [np.asarray(self.grid.get(*self.sub_goals['door'].copy()).can_overlap(), dtype=float)],
                                           self.sub_goals['door'].copy() / self.size,
                                           self.sub_goals['key'].copy() / self.size]),
            'desired_goal': self.sub_goals[self.desired_sub_goal_name].copy() / self.size,
            'achieved_goal': self.agent_pos.copy() / self.size
        }


def make_grid_world(size, max_episode_steps):
    env_id = 'MGDoorKeyEnv'+str(size)+'x'+str(size)+'-v0'
    print(env_id)
    if env_id not in registry.env_specs:
        register(
            id=env_id,
            entry_point='gridworld.multigoal_doorkey:MGDoorKeyEnv',
            kwargs={
                'size': size,
                'max_steps': max_episode_steps
            },
            max_episode_steps=max_episode_steps
        )
    return make(env_id)