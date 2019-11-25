# Markov Decision Process (MDP) script

import gym
import matplotlib.pyplot as plt
import numpy as np
import time
import utils

np.set_printoptions(precision=10, suppress=True)


class MDP:

    def __init__(self, environment, convergence_threshold=0.00001, gamma=0.9, max_iterations=1e6, grid=False):
        self.env_name = environment
        self.env = gym.make(environment)
        self.convergence_threshold = convergence_threshold
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.optimal_policy = None
        self.optimal_policy_actions = None
        self.optimal_V = None
        self.grid = grid

    def __del__(self):
        self.env.close()

    def epsilon_greedy_policy(self, Q_current_state, epsilon):
        """ Epsilon-greedy policy

            Args:
                state  (np array):  State to act from

            Returns:
                action (int): action to take, random or action that maximizes Q(s,a)

            """
        # If acting randomly, take random action, otherwise take action that maximizes Q(s,a)
        if np.random.random() < epsilon:
            return np.random.randint(self.env.nA)
        else:
            return np.argmax(Q_current_state.round(decimals=2))

    def one_step_lookahead(self, state, V):

        # Initialize state-action value function at current state
        Q_current_state = np.zeros(self.env.nA)

        # For each possible action in the action space
        for action in range(self.env.nA):
            # For all next state we can end up in, starting from current state and taking current action
            for probability, next_state, reward, _ in self.env.P[state][action]:
                # Update state-action function at current state and action
                Q_current_state[action] += probability * (reward + self.gamma * V[next_state])

        return Q_current_state

    def play_optimal_policy(self):
        self.play_policy(self.optimal_policy)

    def play_policy(self, policy):

        print('\nPlay policy')

        state = self.env.reset()  # reset environment and get initial state
        accumulated_reward = 0.  # initialize accumulated reward
        t = 0  # initialize time

        # Play the episode until it is terminated
        while True:
            t += 1  # increase time
            self.env.render()  # render the environment

            if policy is not None:
                action = np.argmax(policy[state])  # choose an action following the policy
            else:
                action = self.env.action_space.sample()

            new_state, reward, done, _ = self.env.step(action)  # step to the new state
            accumulated_reward = reward + self.gamma * accumulated_reward  # update the accumulated reward
            print('\nt = {} | s, a, r, s1 = {}, {}, {}, {} | done ?= {}'.
                  format(t, state, action, reward, new_state, done))

            # If the episode is terminated
            if done:
                print('Episode terminated after {} timesteps'.format(t))
                break
            else:
                state = new_state

        self.env.render()  # render the environment last state

    def policy_evaluation(self, V, policy=None, iterations_to_save=[1], visualize=False):

        # Initialize iteration counter
        iteration, saving_index = 0, 0
        delta = 1000.
        deltas = []

        # Start iterating until the maximum number of iterations is reached or we have converged
        while iteration <= self.max_iterations and delta >= self.convergence_threshold:

            iteration += 1  # increase iteration
            delta = 0.  # reset delta of convergence

            # For all possible states in the state space
            for state in range(self.env.nS):
                # Save previous state value
                previous_V = V[state]

                if policy is not None:
                    # Iterative Policy Evaluation: update state value using the Bellman expectation equation
                    Q_current_state = self.one_step_lookahead(state, V)
                    V[state] = 0.
                    for action in range(self.env.nA):
                        V[state] += policy[state, action] * Q_current_state[action]
                else:
                    # Value Iteration: update state value using the Bellman optimality equation
                    V[state] = np.max(self.one_step_lookahead(state, V))

                # Update the delta of convergence to the maximum delta found so far
                delta = max(delta, np.abs(V[state] - previous_V))

                if visualize and policy is None and \
                   iteration == iterations_to_save[saving_index % len(iterations_to_save)]:
                    utils.plot_value_function(V, iteration, show_label=True)
                    saving_index += 1

            deltas.append(delta)

        return iteration, deltas

    def policy_improvement(self, V, policy, policy_actions):

        improve_policy = False  # True if we need to continue improving the policy, False if we have converged

        # For each state
        for state in range(self.env.nS):

            # Save previous best action and compute new best action
            previous_best_action = np.argmax(policy[state])
            Q_current_state = self.one_step_lookahead(state, V)
            new_best_action = np.argmax(Q_current_state.round(decimals=4))

            # If policy has improved
            if previous_best_action != new_best_action:
                improve_policy = True  # we need to continue improving the policy

            # Set probability of new best action to 1 both we have improved or not
            # (if we have improved we change the policy accordingly, otherwise if
            # previous_best_action == new_best_action we reset previous_best_action
            # in the case it was originally a random policy and needed to be updated anyway)
            policy[state] = np.eye(self.env.nA)[new_best_action]
            policy_actions[state] = new_best_action

        return improve_policy

    def policy_iteration(self, iterations_to_save=[0], visualize=False):

        print('\nPolicy-iteration')

        # Initialize start time
        start = time.time()

        # Initialize iteration and improve policy flag
        iteration, saving_index = 0, 0
        improve_policy = True

        # Initialize state value function and random policy
        self.optimal_V = np.zeros(self.env.nS)
        self.optimal_policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
        self.optimal_policy_actions = np.zeros(self.env.nS, dtype=int)

        if visualize and iteration == iterations_to_save[saving_index % len(iterations_to_save)]:
            plt.figure(figsize=(10, 10))
            if self.grid:
                utils.plot_heatmap_policy(self.optimal_policy_actions, self.optimal_V, self.env.nrow, self.env.ncol)
            else:
                utils.plot_optimal_policy(self.optimal_policy_actions)

            utils.set_plot_title_labels(title='PI - Policy at iteration = {}'.format(0))
            utils.save_figure('{}_PI_policy_iter{}'.format(self.env_name, 0))
            saving_index += 1

        # Start iterating until the maximum number of iterations is reached or we have converged
        while iteration <= self.max_iterations and improve_policy:

            iteration += 1  # increase iteration
            print('PI iteration {}'.format(iteration))

            # Evaluate current policy
            self.policy_evaluation(self.optimal_V, self.optimal_policy)

            # Improve current policy
            improve_policy = self.policy_improvement(self.optimal_V, self.optimal_policy, self.optimal_policy_actions)

            if visualize and iteration == iterations_to_save[saving_index % len(iterations_to_save)]:
                plt.figure(figsize=(10, 10))

                if self.grid:
                    utils.plot_heatmap_policy(self.optimal_policy_actions, self.optimal_V, self.env.nrow, self.env.ncol)
                else:
                    utils.plot_optimal_policy(self.optimal_policy_actions)

                utils.set_plot_title_labels(title='PI - Policy at iteration = {}'.format(iteration))
                utils.save_figure('{}_PI_policy_iter{}'.format(self.env_name, iteration))
                saving_index += 1

        if iteration < self.max_iterations:
            print('\n---> converged at iteration {} in {:.4f} seconds'.format(iteration, time.time() - start))
            print('\n---> optimal V = \n{}'.format(self.optimal_V))
            print('\n---> optimal policy = \n{}'.format(self.optimal_policy))
            print('\n---> optimal policy actions = \n{}'.format(self.optimal_policy_actions))
        else:
            print('---> not converged in {} iterations'.format(self.max_iterations))

        if visualize:
            plt.figure(figsize=(10, 10))

            if self.grid:
                utils.plot_heatmap_value_function(self.optimal_V, self.env.nrow, self.env.ncol)
            else:
                utils.plot_value_function(self.optimal_V, 0)

            utils.set_plot_title_labels(title='PI - Optimal Value-function')
            utils.save_figure('{}_PI_optimal_V'.format(self.env_name))

    def Q_learning(self, num_episodes=1000, learning_rate_decay=0.99, epsilon_decay=0.99, visualize=False):

        print('\nQ-Learning')

        # Initialize start time
        start = time.time()

        # Initialize list of accumulated rewards, the episode counter, the epsilon for exploration and the learning rate
        scores, average_scores = [], []
        episode = 0
        epsilon, learning_rate = 1.0, 1.0

        # Initialize state-action value function Q
        Q = np.zeros((self.env.nS, self.env.nA))

        # Play episodes until we have converged
        while True:

            score = 0.  # initialize the accumulated reward
            t = 0  # initialize the time
            episode += 1  # increase episode
            state = self.env.reset()  # reset environment and get initial state

            # Play the episode until it is terminated
            while True:
                t += 1  # increase time
                # self.env.render()  # render the environment

                action = self.epsilon_greedy_policy(Q[state], epsilon)  # choose an action with an epsilon-greedy policy
                new_state, reward, done, _ = self.env.step(action)  # step to the new state
                score += reward  # update the accumulated reward

                # print('\nt = {} | s, a, r, s1 = {}, {}, {}, {} | done ?= {}'.
                #       format(t, state, action, reward, new_state, done))

                # If the episode is terminated
                if done:
                    # Update Q at current state and action
                    Q[state, action] += learning_rate * (reward - Q[state, action])
                    # print('Episode terminated after {} timesteps'.format(t))
                    break
                else:
                    # Update Q at current state and action by bootstrapping next state
                    Q[state, action] += learning_rate * (reward + self.gamma * np.max(Q[new_state]) - Q[state, action])
                    state = new_state

            scores.append(score)  # save the accumulated reward
            average_scores.append(np.mean(scores[-100:]))
            # self.env.render()  # render the environment last state

            if episode % 10 == 0:
                print('\nEpisode {}'.format(episode))
                print('average score = {:.4f}'.format(average_scores[-1]))

            # If we have converged, break
            if episode > num_episodes:
                print('\n---> converged at episode {} in {:.4f} seconds'.format(episode, time.time() - start))
                break

            epsilon = max(0.01, epsilon_decay * epsilon)  # exponentially decay epsilon
            learning_rate = max(0.01, learning_rate_decay * learning_rate)  # exponentially decay learning rate

            # if episode % 10 == 0:
            #     print('epsilon = {:.4f}, lr = {:.4f}'.format(epsilon, learning_rate))

        # Initialize optimal policy and state value function
        self.optimal_V = np.zeros(self.env.nS)
        self.optimal_policy = np.zeros((self.env.nS, self.env.nA))
        self.optimal_policy_actions = np.zeros(self.env.nS, dtype=int)

        # For all possible states in the state space
        for state in range(self.env.nS):
            # Select best action and best value based on the highest state-action value at current state
            best_action = np.argmax(Q[state])
            best_value = np.max(Q[state])

            # Update the optimal state value and policy to perform the best action at current state
            self.optimal_policy[state, best_action] = 1.0
            self.optimal_V[state] = best_value
            self.optimal_policy_actions[state] = best_action

        print('\n---> optimal Q = \n{}'.format(Q))
        print('\n---> optimal V = \n{}'.format(self.optimal_V))
        print('\n---> optimal policy = \n{}'.format(self.optimal_policy))

        if visualize:

            plt.figure(figsize=(10, 10))
            utils.plot_value_function(average_scores, 0)
            utils.set_plot_title_labels(title='Q-Learning - Moving Average Score',
                                        x_label='Episodes', y_label='Moving Average Score')
            utils.save_figure('{}_QL_accumulated_reward'.format(self.env_name))

            plt.figure(figsize=(10, 10))

            if self.grid:
                utils.plot_heatmap_value_function(self.optimal_V, self.env.nrow, self.env.ncol)
            else:
                utils.plot_value_function(self.optimal_V, 0)

            utils.set_plot_title_labels(title='Q-Learning - Optimal State-Value-function')
            utils.save_figure('{}_QL_optimal_V'.format(self.env_name))

            plt.figure(figsize=(10, 10))

            if self.grid:
                utils.plot_heatmap_policy(self.optimal_policy_actions, self.optimal_V, self.env.nrow, self.env.ncol)
            else:
                utils.plot_optimal_policy(self.optimal_policy_actions)

            utils.set_plot_title_labels(title='Q-Learning - Optimal Policy')
            utils.save_figure('{}_QL_optimal_policy'.format(self.env_name))

    def value_iteration(self, iterations_to_save=[1], visualize=False):

        print('\nValue-iteration')

        # Initialize start time
        start = time.time()

        # Initialize state value function
        self.optimal_V = np.zeros(self.env.nS)

        # Compute the optimal state value function using policy evaluation, but with the Bellman optimality equation
        if visualize:
            plt.figure(figsize=(10, 10))

        iteration, deltas = self.policy_evaluation(self.optimal_V,
                                                   iterations_to_save=iterations_to_save,
                                                   visualize=visualize)

        if visualize:
            utils.set_plot_title_labels(title='VI - Value-function through iterations',
                                        x_label='State', y_label='V estimate', legend=True)
            utils.save_figure('{}_VI_iterations'.format(self.env_name))

            plt.figure(figsize=(10, 10))
            utils.plot_value_convergence(deltas, iteration)
            utils.set_plot_title_labels(title='VI - Value-function convergence',
                                        x_label='Iterations', y_label='V Delta')
            utils.save_figure('{}_VI_convergence'.format(self.env_name))

        # Initialize optimal policy
        self.optimal_policy = np.zeros((self.env.nS, self.env.nA))
        self.optimal_policy_actions = np.zeros(self.env.nS, dtype=int)

        # For all possible states in the state space
        for state in range(self.env.nS):

            # Select best action based on the highest state-action value at current state
            Q_current_state = self.one_step_lookahead(state, self.optimal_V)
            best_action = np.argmax(Q_current_state.round(decimals=4))

            # Update the optimal policy to perform the best action at current state
            self.optimal_policy[state, best_action] = 1.0
            self.optimal_policy_actions[state] = best_action

        if iteration < self.max_iterations:
            print('\n---> converged at iteration {} in {:.4f} seconds'.format(iteration, time.time() - start))
            print('\n---> optimal V = \n{}'.format(self.optimal_V))
            print('\n---> optimal policy = \n{}'.format(self.optimal_policy))
            print('\n---> optimal policy actions = \n{}'.format(self.optimal_policy_actions))
        else:
            print('---> not converged in {} iterations'.format(self.max_iterations))

        if visualize:

            plt.figure(figsize=(10, 10))

            if self.grid:
                utils.plot_heatmap_value_function(self.optimal_V, self.env.nrow, self.env.ncol)
            else:
                utils.plot_value_function(self.optimal_V, 0)

            utils.set_plot_title_labels(title='VI - Optimal Value-function')
            utils.save_figure('{}_VI_optimal_V'.format(self.env_name))

            plt.figure(figsize=(10, 10))

            if self.grid:
                utils.plot_heatmap_policy(self.optimal_policy_actions, self.optimal_V, self.env.nrow, self.env.ncol)
            else:
                utils.plot_optimal_policy(self.optimal_policy_actions)

            utils.set_plot_title_labels(title='VI - Optimal Policy')
            utils.save_figure('{}_VI_optimal_policy'.format(self.env_name))
