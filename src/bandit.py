import numpy as np
import re
import gym

from gym import spaces
from gym.utils import seeding
from copy import copy
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import linear_model

class BanditEnv(gym.Env):

    def __init__(self, done_reward = 100, ref_reward = -5, full_tree_reward = -3, middle_tree_reward = -2, wrong_act_reward = -5):
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
        self.unrefinable_acts = [0,1,2, 5, 8, 11]
        self._seed()
        self.used_states = set()
        self.model = self.fit_model()

    def fit_model(self):
        self.X = []
        self.y = []
        with open("carusers_with_actions.txt", 'r') as f:
            for line in f:
                l = line.split(':')
                self.y.append(int(l[0]))
                self.X.append([float(s.strip()) for s in re.findall(r"[-+]?\d*\.\d+|\d+", l[1])])
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.X, self.y = shuffle(self.X, self.y, random_state=0)
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        return regr

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action, user):
        assert self.action_space.contains(action)
        # # goal check
        # if self.state == self.goal:
        #     self.done = True
        #     return self.state, self.done_reward, self.done, None
        #actions
        reward = -1
        illigal = False
        if {0.1, 0.2, 0.3} <= self.used_states:
            reward+=self.done_reward
            self.done = True
            return self.state, self.done_reward, self.done, None
        new_state = None
        act = int(action)
        if self.state_actions[act] != self.state:
            reward = self.war
            new_state = self.state
            illigal = True
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
        # todo change for real value and update network
        reward += self.network_reward(act, user)
        # if len(self.used_states)>1:
        #     if np.random.randint(10) < 8 and action not in self.unrefinable_acts and not illigal:
        #         reward+=self.rwd
        if new_state == 0:
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


    def network_reward(self, act, user):
        reward = 0
        if act not in [0, 1, 2]:
            predicted_act = round(self.model.predict([user])[0])
            if act == predicted_act:
                reward+=5
        return reward





