import gym, Bandits
env = gym.make("BanditEnv-v0") # Replace with relevant env
from Bandits.agent import QLearningAgent

gamma = 1.0
alpha=0.5
epsilon=0.1

agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

all_rewards, average_rewards = agent.train(100, True)
print(all_rewards)

