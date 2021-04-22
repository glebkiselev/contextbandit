import gym
from gym import spaces
from gym.utils import seeding
from copy import copy
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


class BanditEnv(gym.Env):

    def __init__(self, done_reward=100, ref_reward=-5, full_tree_reward=-3, middle_tree_reward=-2, wrong_act_reward=-5):
        self.done = False
        self.state = None
        self.goal = None
        self.step_reward = -1
        self.done_reward = done_reward
        self.rwd = ref_reward
        self.ftr = full_tree_reward
        self.mtr = middle_tree_reward
        self.actions = 12
        self.state_actions = {0: 0, 1: 0, 2: 0, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.2, 7: 0.2, 8: 0.2, 9: 0.3, 10: 0.3,
                              11: 0.3}
        self.classes = {0: [0, 1, 2, 3, 6, 9], 1: [0, 1, 2, 4, 7, 10], 2: [0, 1, 2, 5, 8, 11]}
        self.action_space = spaces.Discrete(self.actions)
        self.observation_space = spaces.Discrete(1)
        self.war = wrong_act_reward
        self.unrefinable_acts = [0, 1, 2, 5, 8, 11]
        self._seed()
        self.used_states = {cl: dict() for cl, acts in self.classes.items()}
        self.model = self.load_model()
        self.df = self.load_dataset()

    def load_model(self, model='model.car'):
        import pickle
        try:
            loaded_model = pickle.load(open(model, 'rb'))
        except FileNotFoundError:
            raise Exception('Wrong path to model file')
        return loaded_model

    def load_dataset(self, dataset='test_batch.csv'):
        import pandas as pd
        df = pd.read_csv(dataset)
        return df

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, user):
        assert self.action_space.contains(action)
        # # goal check
        # if self.state == self.goal:
        #     self.done = True
        #     return self.state, self.done_reward, self.done, None
        # actions
        reward = -1
        illigal = False
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
                # if act == 3:
                #     reward += self.step_reward
                # elif act == 4:
                #     reward += self.mtr
                # elif act == 5:
                #     reward += self.ftr
                reward += self.step_reward
                new_state = 0
            elif self.state == 0.2:
                # if act == 6:
                #     reward += self.step_reward
                # elif act == 7:
                #     reward += self.mtr
                # elif act == 8:
                #     reward += self.ftr
                reward += self.step_reward
                new_state = 0
            elif self.state == 0.3:
                # if act == 9:
                #     reward += self.step_reward
                # elif act == 10:
                #     reward += self.mtr
                # elif act == 11:
                #     reward += self.ftr
                reward += self.step_reward
                new_state = 0
        # randomize ref reward:
        # todo change for real value and update network
        predicted_group = None

        if not illigal:
            predicted_group = round(self.model.predict([user])[0])
            if action in self.classes[predicted_group]:
                reward += 5
            if min([len(val) for el, val in self.used_states.items()]) == 3:
                # print(min(min([list(val.values()) for el, val in self.used_states.items()])))
                min_comes = 100
                for group, states in self.used_states.items():
                    for state, comes in states.items():
                        if comes < min_comes:
                            min_comes = comes
                if min_comes >= 20:
                # if min(min([list(val.values()) for el, val in self.used_states.items()])) >= 10:
                    reward += self.done_reward
                    self.done = True
                    print(f'\nGoal achieved: {self.used_states}\n')
            if new_state == 0 and not illigal:
                for gr, sts in self.used_states.items():
                    if gr == predicted_group:
                        if self.state in sts:
                            sts[self.state] +=1
                        else:
                            sts[self.state] = 1
                        break
                # self.used_states.setdefault(predicted_group, set().add(copy(self.state)))
            actions = self.df[self.df['State'] == user[0]][self.df['Gender'] == user[1]] \
                [self.df['Interest'] == user[2]][self.df['Key-words in query'] == user[3]][
                self.df['Prev Interest'] == user[4]][
                'Action'].tolist()
            if action in actions:
                reward += 20

        self.state = new_state

        return self.state, reward, self.done, predicted_group

    def _get_reward(self, action):
        pass

    def reset(self):
        self.used_states = {cl: dict() for cl, acts in self.classes.items()}
        self.state = 0
        self.done = False
        return 0

    def render(self, mode='human', close=False):
        pass

    # def network_reward(self, act, user):
    #     reward = 0
    #     predicted_group = round(self.model.predict([user])[0])
    #     return reward
