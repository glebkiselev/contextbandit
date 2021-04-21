import sys
from collections import defaultdict
import numpy as np
from sklearn.utils import shuffle
import pandas as pd

class Agent:
    def __init__(self):
        # todo automate classes search and save somewhere in common with enviroment place
        self.classes = {0: [3, 6, 9], 1: [4, 7, 10], 2: [5, 8, 11]}

    def act(self, state):
        raise NotImplementedError()

    def update(self, state, action, reward, next_state, parametr):
        raise NotImplementedError()

    def load_model(self, model = 'model.car'):
        import pickle
        try:
            loaded_model = pickle.load(open(model, 'rb'))
        except FileNotFoundError:
            raise Exception('Wrong path to model file')
        return loaded_model

    def load_dataset(self, dataset = 'carusers_with_actions.csv', batch_size = 100):
        import pandas as pd
        chunk = pd.read_csv(dataset, nrows=batch_size)
        return chunk


    def update_model(self, data = pd.DataFrame(), model = 'model.car', datasetfile = 'carusers_with_actions.csv', batch_size = 100):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.naive_bayes import GaussianNB
        import pickle

        if data.empty:
            data = pd.read_csv(datasetfile)
            data.to_csv('test_batch.csv')
        #shuffle data
        data = data.sample(frac=1).reset_index(drop=True)
        # ".iloc"  - row_indexer, column_indexer
        X = data.iloc[:batch_size, 1:].values
        y = data.iloc[:batch_size, 0].values

        yy = []
        for el in y:
            for key, value in self.classes.items():
                if el in value: yy.append(key)

        X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.20, random_state=27)

        g_model = GaussianNB()
        g_model.fit(X_train, y_train)
        g_prediction = g_model.predict(X_test)
        print(f"accuracy of classification by Gauss: {accuracy_score(g_prediction, y_test)}")
        pickle.dump(g_model, open(model, 'wb'))
        return g_model

class QLearningAgent(Agent):
    """
    An implementation of the Q Learning agent.
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1, beta=0.2, zeroupdate = False):
        super(QLearningAgent, self).__init__()
        self.environment = env
        self.number_of_action = env.action_space.n
        self.q = defaultdict(lambda: [np.zeros(self.number_of_action) for _ in range(len(self.classes.keys()))])
        self.r_avg = 0
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.policy = self._make_epsilon_greedy_policy()

        # todo - change to real user choosing
        data = self.load_dataset(batch_size = 50)
        self.X = data.iloc[:,1:].values
        self.y = data.iloc[:,0].values
        self.X, self.y = shuffle(self.X, self.y, random_state=0)
        if not zeroupdate:
            self.model = self.load_model()
        else:
            self.model = self.update_model()


    def _make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        """
        def policy_fn(state, group):
            A = np.ones(self.number_of_action, dtype=float) * self.epsilon / self.number_of_action
            best_action = np.argmax(self.q[state][group])
            A[best_action] += (1.0 - self.epsilon)
            return A
        return policy_fn

    def act(self, state):
        group = np.random.choice(len(self.classes.keys()))
        action_probs = self.policy(state, group)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, state, action, reward, next_state, gr):
        if gr is None:
            groups = self.classes.keys()
        else:
            groups = [gr]
        for group in groups:
            best_next_action = np.argmax(self.q[next_state][group])
            td_target = reward + self.gamma * self.q[next_state][group][best_next_action]
            td_delta = td_target - self.q[state][group][action]
            self.q[state][group][action] += round(self.alpha * td_delta, 1)

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
                next_state, reward, done, group = self.environment.step(action, user)
                next_state = str(next_state)
                total_reward += reward
                self.update(state, action, reward, next_state, group)
                if done:
                    total_total_reward += total_reward
                    rewards.append(total_reward)
                    break

                state = next_state
        # return total_total_reward / num_episodes, rewards  # return average eps reward
        return rewards, total_total_reward / num_episodes

    def classification_on_batch(self, n = 4):
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

    def classify(self, user):
        return self.model.predict([user])[0]


    def uspred(self, action):
        user = self.X[np.random.randint(len(self.X))]
        return user
