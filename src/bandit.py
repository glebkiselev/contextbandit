import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from copy import copy, deepcopy

class BanditEnv(gym.Env):

    def __init__(self, done_reward = 10, ref_reward = -5, full_tree_reward = -3, middle_tree_reward = -2, wrong_act_reward = -5):

        self.done = False
        self.state = None
        self.goal = None
        self.step_reward = -1
        self.done_reward = done_reward
        self.rwd = ref_reward
        self.ftr = full_tree_reward
        self.mtr = middle_tree_reward
        self.actions = 12
        self.state_actions = {0:0, 1:0, 2:0, 3:0.1, 4:0.1, 5:0.1, 6:0.2, 7: 0.2, 8:0.2, 9: 0.3, 10: 0.3, 11:0.3}
        self.action_space = spaces.Discrete(self.actions)
        self.observation_space = spaces.Discrete(1)
        self.war = wrong_act_reward
        self.last_acts = [2, 5, 8]
        self._seed()
        self.used_states = set()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        assert self.action_space.contains(action)
        # # goal check
        # if self.state == self.goal:
        #     self.done = True
        #     return self.state, self.done_reward, self.done, None
        #actions
        reward = -1
        if {0.1, 0.2, 0.3} <= self.used_states:
            reward+=self.done_reward
            self.done = True
            return self.state, self.done_reward, self.done, None
        new_state = None
        act = int(action)
        if self.state_actions[act] != self.state:
            reward = self.war
            new_state = self.state
        else:
            if self.state == 0:
                if act == 0:
                    reward += self.step_reward
                    new_state = 0.1
                elif act == 1:
                    reward += self.step_reward
                    new_state = 0.2
                elif act == 2:
                    reward += self.step_reward
                    new_state = 0.3
            elif self.state == 0.1:
                if act == 3:
                    reward += self.step_reward
                elif act == 4:
                    reward += self.mtr
                elif act == 5:
                    reward += self.ftr
                new_state = 0
            elif self.state == 0.2:
                if act == 6:
                    reward += self.step_reward
                elif act == 7:
                    reward += self.mtr
                elif act == 8:
                    reward += self.ftr
                new_state = 0
            elif self.state == 0.3:
                if act == 9:
                    reward += self.step_reward
                elif act == 10:
                    reward += self.mtr
                elif act == 11:
                    reward += self.ftr
                new_state = 0
        # randomize ref reward:
        # todo change for real value and teach on network

        if len(self.used_states)>1:
            if np.random.randint(10) < 6 and action not in self.last_acts:
                reward+=self.rwd

        self.used_states.add(copy(self.state))
        self.state = new_state

        return self.state, reward, self.done, None

    def _get_reward(self, action):
        pass

    def reset(self):
        self.used_states = set()
        self.state = 0
        return 0

    def render(self, mode='human', close=False):
        pass


# class NineArmed(BanditEnv):
#
#     def __init__(self, bandits=9):
#         # make random distribution of action choose
#         p_dist = np.random.uniform(size=bandits)
#         # make random distribution of env rewards
#         r_dist = np.full(bandits, 1)
#         BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)
#         print("in bandit")

