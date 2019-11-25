# Script that runs the experiment
import gym
import sys

from mdp import MDP

sys.path.append(' .')
gym.envs.register('Gambler-v0', entry_point='gambler:GamblerEnv', max_episode_steps=1000)

if __name__ == '__main__':

    frozen_lake = MDP(environment='FrozenLake-v0', convergence_threshold=0.0001, grid=True)
    print('Frozen Lake :)))')

    # frozen_lake.value_iteration(iterations_to_save=[1, 5, 10, 15, 20, 25, 30, 35], visualize=True)
    # frozen_lake.policy_iteration(iterations_to_save=[0, 1, 2], visualize=True)
    # frozen_lake.Q_learning(num_episodes=10000, learning_rate_decay=0.995, epsilon_decay=0.995, visualize=True)

    gambler = MDP(environment='Gambler-v0', convergence_threshold=0.00001, grid=False)
    print('Gambler :)))')

    gambler.value_iteration(iterations_to_save=[1, 2, 4, 6, 8, 10, 12], visualize=False)
    gambler.policy_iteration(iterations_to_save=[1, 8, 16], visualize=False)
    gambler.Q_learning(num_episodes=10000, learning_rate_decay=0.995, epsilon_decay=0.995, visualize=True)
