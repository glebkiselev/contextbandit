from gym.envs.registration import register

from .bandit import BanditEnv

environments = [['BanditEnv', 'v0']
                ]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='Bandits:{}'.format(environment[0]),
    )