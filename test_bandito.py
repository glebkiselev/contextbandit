import gym, Bandits
env = gym.make("BanditEnv-v0")
from Bandits.agent import QLearningAgent
import numpy as np

gamma = 1.0
alpha=0.5
epsilon=0.1


def q_to_policy(q, offset=0):
    optimal_policy_groups = {}
    for state in q:
        i = 0
        for group in q[state]:
            optimal_policy_groups.setdefault(state, {}).update({i: np.argmax(group)})
            i += 1
    return optimal_policy_groups

if __name__ == "__main__":
    agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    all_rewards, average_rewards = agent.train(100, True)
    print(all_rewards)
    print(average_rewards)
    policy = q_to_policy(agent.q)
    print(policy)
    #print(agent.q)

