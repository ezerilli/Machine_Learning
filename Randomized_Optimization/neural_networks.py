# Neural Networks utils

import numpy as np
import matplotlib.pyplot as plt
import utils

from mlrose.algorithms.decay import ExpDecay
from mlrose.neural import NeuralNetwork

from sklearn.metrics import log_loss, classification_report
from sklearn.model_selection import train_test_split

IMAGE_DIR = 'images/'


def plot_nn_performances(x_train, y_train, random_seeds, **kwargs):
    """Plot Neural Networks performances on the training set.

        Use different optimizations algorithms (RHC, SA, GA and GD) and compare results on the training set using
        k-fold cross-validation.

        Args:
        x_train (ndarray): training data.
        y_train (ndarray): training labels.
        random_seeds (list or array): random seeds for multiple random runs to use for k-fold cross-validation.
        kwargs (dict): additional arguments to pass for curves plotting:
                  - rhc_max_iters (list or ndarray): RHC list or array of maximum number of iterations to plot vs.
                  - sa_max_iters (list or ndarray): SA list or array of maximum number of iterations to plot vs.
                  - ga_max_iters (list or ndarray): GA list or array of maximum number of iterations to plot vs.
                  - gd_max_iters (list or ndarray): GD list or array of maximum number of iterations to plot vs.
                  - init_temp (float): SA initial temperature.
                  - exp_decay_rate (float): SA temperature exponential decay rate.
                  - min_temp (float): SA minimum temperature.
                  - pop_size (int): GA population size.
                  - mutation_prob (float): GA mutation probability.

        Returns:
        None.
           """

    # Initialize algorithms, corresponding acronyms and max number of iterations
    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']
    acronyms = ['RHC', 'SA', 'GA', 'GD']
    max_iters = ['rhc_max_iters', 'sa_max_iters', 'ga_max_iters', 'gd_max_iters']

    # Initialize lists of training curves, validation curves and training times curves
    train_curves, val_curves, train_time_curves = [], [], []

    # Define SA exponential decay schedule
    exp_decay = ExpDecay(init_temp=kwargs['init_temp'],
                         exp_const=kwargs['exp_decay_rate'],
                         min_temp=kwargs['min_temp'])

    # Create one figure for training and validation losses, the second for training time
    plt.figure()
    train_val_figure = plt.gcf().number
    plt.figure()
    train_times_figure = plt.gcf().number

    # For each of the optimization algorithms to test the Neural Network with
    for i, algorithm in enumerate(algorithms):
        print('\nAlgorithm = {}'.format(algorithm))

        # For multiple random runs
        for random_seed in random_seeds:

            # Initialize training losses, validation losses and training time lists for current random run
            train_losses, val_losses, train_times = [], [], []

            # Compute stratified k-fold
            x_train_fold, x_val_fold, y_train_fold, y_val_fold = train_test_split(x_train, y_train,
                                                                                  test_size=0.2, shuffle=True,
                                                                                  random_state=random_seed,
                                                                                  stratify=y_train)
            # For each max iterations to run for
            for max_iter in kwargs[max_iters[i]]:

                print('Iteration = {}'.format(max_iter))

                # Define Neural Network using current algorithm
                nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                                   algorithm=algorithm, max_iters=int(max_iter),
                                   bias=True, is_classifier=True, learning_rate=0.001,
                                   early_stopping=False, clip_max=1e10, schedule=exp_decay,
                                   pop_size=kwargs['pop_size'], mutation_prob=kwargs['mutation_prob'],
                                   max_attempts=int(max_iter), random_state=random_seed, curve=False)

                # Train on current training fold and append training time
                global start_time
                start_time = time.time()
                nn.fit(x_train_fold, y_train_fold)
                train_times.append(time.time() - start_time)

                # Compute and append training and validation log losses
                train_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
                val_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print('\ntrain loss = {}, val loss = {}'.format(train_loss, val_loss))

            # Append curves for current random seed to corresponding lists of curves
            train_curves.append(train_losses)
            val_curves.append(val_losses)
            train_time_curves.append(train_times)

        # Plot training and validation figure for current algorithm
        plt.figure(train_val_figure)
        utils.plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(train_curves), label='{} train'.format(acronyms[i]))
        utils.plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(val_curves), label='{} val'.format(acronyms[i]))

        # Plot training time figure for current algorithm
        plt.figure(train_times_figure)
        utils.plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(train_time_curves), label=acronyms[i])

    # Set title and labels to training and validation figure
    plt.figure(train_val_figure)
    utils.set_plot_title_labels(title='Neural Network - Loss vs. iterations',
                                x_label='Iterations',
                                y_label='Loss')

    # Save figure
    plt.savefig(IMAGE_DIR + 'nn_objective_vs_iterations')

    # Set title and labels to training time figure
    plt.figure(train_times_figure)
    utils.set_plot_title_labels(title='Neural Network - Time vs. iterations',
                                x_label='Iterations',
                                y_label='Time (seconds)')

    # Save figure
    plt.savefig(IMAGE_DIR + 'nn_time_vs_iterations')


def test_nn_performances(x_train, x_test, y_train, y_test, random_seed, **kwargs):
    """Test Neural Networks performances on the test set using different optimizations algorithms: RHC, SA, GA and GD.

        Args:
        x_train (ndarray): training data.
        x_test (ndarray): test data.
        y_train (ndarray): training labels.
        y_test (ndarray): test labels.
        random_seed (int): random seed.
        kwargs (dict): additional arguments to pass for curves plotting:
                   - max_iters (int): maximum number of iterations.
                   - init_temp (float): SA initial temperature.
                   - exp_decay_rate (float): SA temperature exponential decay rate.
                   - min_temp (float): SA minimum temperature.
                   - pop_size (int): GA population size.
                   - mutation_prob (float): GA mutation probability.

        Returns:
        None.
        """

    # Define SA exponential decay schedule
    exp_decay = ExpDecay(init_temp=kwargs['init_temp'],
                         exp_const=kwargs['exp_decay_rate'],
                         min_temp=kwargs['min_temp'])

    # Define Neural Network using RHC for weights optimization
    rhc_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                           algorithm='random_hill_climb', max_iters=kwargs['max_iters'],
                           bias=True, is_classifier=True, learning_rate=0.001,
                           early_stopping=False, clip_max=1e10,
                           max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)

    # Define Neural Network using SA for weights optimization
    sa_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                          algorithm='simulated_annealing', max_iters=kwargs['max_iters'],
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10, schedule=exp_decay,
                          max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)

    # Define Neural Network using GA for weights optimization
    ga_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                          algorithm='genetic_alg', max_iters=kwargs['max_iters'],
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10,
                          pop_size=kwargs['pop_size'], mutation_prob=kwargs['mutation_prob'],
                          max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)

    # Define Neural Network using GD for weights optimization
    gd_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                          algorithm='gradient_descent', max_iters=kwargs['max_iters'],
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10,
                          max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)

    # Fit each of the Neural Networks using the different optimization algorithms
    rhc_nn.fit(x_train, y_train)
    sa_nn.fit(x_train, y_train)
    ga_nn.fit(x_train, y_train)
    gd_nn.fit(x_train, y_train)

    # Print classification reports for all of the optimization algorithms
    print('RHC test classification report = \n {}'.format(classification_report(y_test, rhc_nn.predict(x_test))))
    print('SA test classification report = \n {}'.format(classification_report(y_test, sa_nn.predict(x_test))))
    print('GA test classification report = \n {}'.format(classification_report(y_test, ga_nn.predict(x_test))))
    print('GD test classification report = \n {}'.format(classification_report(y_test, gd_nn.predict(x_test))))
