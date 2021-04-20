import gym, Bandits
env = gym.make("BanditEnv-v0")
from Bandits.agent import QLearningAgent
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import configparser

gamma = 0.1
alpha=0.5
epsilon=0.5


def q_to_policy(q, offset=0):
    optimal_policy_groups = {}
    for state in q:
        i = 0
        for group in q[state]:
            optimal_policy_groups.setdefault(state, {}).update({i: np.argmax(group)})
            i += 1
    return optimal_policy_groups


def pseudo_user_gen(users_num=100, groups=150, car_groups=10, key_words_num=20, all_words_num=30, save = False):
    users = []
    pbuttons = np.linspace(0.1, 0.3, 3)
    previnterest = np.linspace(0.1, 0.3, 3)
    gender = [0, 1]
    for _ in range(0, users_num):
        u_gr = np.random.randint(car_groups)
        interest = round(u_gr / np.random.randint(u_gr + 1, groups), 1)
        u_kw = np.random.randint(key_words_num)
        query = round(u_kw / np.random.randint(u_kw + 1, all_words_num), 1)
        user = [np.random.choice(pbuttons), np.random.choice(gender), interest, query, np.random.choice(previnterest)]
        users.append(user)
    if save:
        df1 = pd.DataFrame({
            #     "Id":np.arange(len(y)),
            "State": [p[0] for p in users],
            "Gender": [p[1] for p in users],
            "Interest": [p[2] for p in users],
            "Key-words in query": [p[3] for p in users],
            "Prev Interest": [p[4] for p in users]
        })
        df1.to_csv("test_batch.csv", index=False)
    return users

def make_bias(actions):
    """
    here we make synthetic bias to actions
    :param actions: predicted actions
    :return: new list of actions with bias
    """
    state_acts = {0.1: [3, 4, 5], 0.2: [6, 7, 8], 0.3: [9, 10, 11]}
    new_actions = []
    biased = 0
    for action in actions:
        if np.random.randint(10) < 2:
            for _, acts in state_acts.items():
                if action in acts:
                    np.random.shuffle(acts)
                    new_actions.append([act for act in acts if act != action][0])
                    biased+=1
        else:
            new_actions.append(action)
    print(f"\nBiased actions from dataset {len(actions)} was {biased}")
    return new_actions


def syntetic_batch(agent, confile, save = True):
    config = configparser.ConfigParser()
    config.read(confile)
    kwargs = config["SECOND"]
    kwargs = {el:eval(val) for el, val in kwargs.items()}
    test_batch = pseudo_user_gen(**kwargs)
    actions = predict_actions(agent, test_batch)
    actions = make_bias(actions)

    df = pd.DataFrame({
        "Action": actions,
        "State": [p[0] for p in test_batch],
        "Gender": [p[1] for p in test_batch],
        "Interest": [p[2] for p in test_batch],
        "Key-words in query": [p[3] for p in test_batch],
        "Prev Interest": [p[4] for p in test_batch]
    })
    if save:
        df.to_csv("test_batch.csv", index=False)
    return df

def predict_actions(agent, batch):
    policy = q_to_policy(agent.q)
    actions = []
    for user in batch:
        group = agent.classify(user)
        action = policy[str(user[0])][group]
        actions.append(action)
    return actions

def print_accuracy(agent,data):
    y = data['Action']
    X = data.iloc[:,1:].values
    predicted = predict_actions(agent,X)
    print(f"accuracy of retrained agent is {accuracy_score(y, predicted)}")
    print()

if __name__ == "__main__":
    agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    all_rewards, average_rewards = agent.train(100, True)
    # print(all_rewards)
    # print(average_rewards)
    df = syntetic_batch(agent, 'config.ini')
    olddf =  pd.read_csv('carusers_with_actions.csv')
    df.append(olddf)
    agent.update_model(data = df)
    agent.train(100, True)
    print_accuracy(agent, olddf)




    # test_batch = pseudo_user_gen(users_num=10, groups=200, car_groups=15, key_words_num=5, all_words_num=15, save = True)
    # test_agent(agent, test_batch)



# todo test agent for new user and give a new batch for retrain

