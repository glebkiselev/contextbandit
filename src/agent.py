import sys
from collections import defaultdict
import numpy as np
import re
from sklearn.utils import shuffle

class Agent:
    def act(self, state):
        raise NotImplementedError()

    def update(self, state, action, reward, next_state):
        raise NotImplementedError()

class QLearningAgent(Agent):
    """
    An implementation of the Q Learning agent.
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1, beta=0.2):
        self.environment = env
        self.number_of_action = env.action_space.n
        self.q = defaultdict(lambda: np.zeros(self.number_of_action))
        self.r_avg = 0
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.policy = self._make_epsilon_greedy_policy()
        self.X = []
        self.y = []
        with open("carusers_with_actions.txt", 'r') as f:
            for line in f:
                l = line.split(':')
                self.y.append(int(l[0]))
                self.X.append([float(s.strip()) for s in re.findall(r"[-+]?\d*\.\d+|\d+", l[1])])
        self.X, self.y = shuffle(self.X, self.y, random_state=0)

    def _make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        """
        def policy_fn(state):
            A = np.ones(self.number_of_action, dtype=float) * self.epsilon / self.number_of_action
            best_action = np.argmax(self.q[state])
            A[best_action] += (1.0 - self.epsilon)
            return A
        return policy_fn

    def act(self, state):
        action_probs = self.policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q[next_state])
        td_target = reward + self.gamma * self.q[next_state][best_next_action]
        td_delta = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_delta

    def train(self, num_episodes=500, verbose=False):
        total_total_reward = 0.0
        rewards = []
        # for each episod while we train. Total - 1000 episodes
        for i_episode in range(num_episodes):
            # Print out which episode we're on.
            if verbose:
                if (i_episode + 1) % 1 == 0:
                    print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                    sys.stdout.flush()
            # set env to start values
            state = self.environment.reset()
            state = str(state)
            total_reward = 0.0
            for _ in range(1000):
                # choose action by eps greedy policy
                action = self.act(state)
                user = self.uspred(action)
                # if illigal act -5 if nice act +5 reward else rew -1
                next_state, reward, done, _ = self.environment.step(action, user)
                next_state = str(next_state)
                total_reward += reward
                self.update(state, action, reward, next_state)
                if done:
                    total_total_reward += total_reward
                    rewards.append(total_reward)
                    break

                state = next_state
        # return total_total_reward / num_episodes, rewards  # return average eps reward
        return rewards, total_total_reward / num_episodes

    def user_classification_on_batch(self, n = 4):
        """
        we create n groups of people using small batch and
        linear regression. After that this batch is sent to
        Q-agent to create Q-table. In this table every state has
        a classification of actions that are applicable in it
        for each group and a cost of each action.
        In real life application this table can change cost values.
        :return:
        """
        pass

    def uspred(self, action):
        """
        Here we create a user vector for get additional
        reward from the environment prediction function
        todo - connect to real vk-bot
        """
        if action in [0, 1, 2]:
            user = [0, 0, 0, 0, 0]
        else:
            users = [us for us, ac in zip(self.X, self.y) if ac == action]
            user = users[np.random.randint(len(users))]
        return user
