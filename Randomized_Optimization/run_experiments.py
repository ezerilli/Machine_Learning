# Script to run experiments

import neural_networks as nn
import numpy as np
import matplotlib.pyplot as plt
import randomized_optimization as ro


from mlrose.opt_probs import TSPOpt, DiscreteOpt
from mlrose.fitness import TravellingSales, FlipFlop, FourPeaks

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(split_percentage=0.2):
    """Load WDBC dataset.

        Args:
           split_percentage (float): validation split.

        Returns:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           y_test (ndarray): test labels.
        """

    # Loadt dataset and split in training and validation sets, preserving classes representation
    data = load_breast_cancer()
    x, y, labels, features = data.data, data.target, data.target_names, data.feature_names
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_percentage, shuffle=True,
                                                        random_state=42, stratify=y)

    # Normalize feature data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print('\nTotal dataset size:')
    print('Number of instances: {}'.format(x.shape[0]))
    print('Number of features: {}'.format(x.shape[1]))
    print('Number of classes: {}'.format(len(labels)))
    print('Training Set : {}'.format(x_train.shape))
    print('Testing Set : {}'.format(x_test.shape))

    return x_train, x_test, y_train, y_test


def flip_plop(length, random_seeds):
    """Define and experiment Flip Flop optimization problem.

        Flip Flop is an optimization problem whose purpose is to optimize
        the number of alternations in as vector of bits.

        Args:
           length (int): problem length.
           random_seeds (list or ndarray): random seeds for get performances over multiple random runs.

        Returns:
           None.
        """

    # Define Flip Flop objective function and problem
    flip_flop_objective = FlipFlop()
    problem = DiscreteOpt(length=length, fitness_fn=flip_flop_objective, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)  # set fast MIMIC

    # Plot optimizations for RHC, SA, GA and MIMIC
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=1000, sa_max_iters=1000, ga_max_iters=250, mimic_max_iters=50,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.05, 2.01, 0.05), sa_min_temp=0.001,
                          ga_pop_size=300, mimic_pop_size=1500, ga_keep_pct=0.2, mimic_keep_pct=0.4,
                          pop_sizes=np.arange(100, 2001, 200), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='Flip Flop', plot_ylabel='Fitness')

    # Plot performances for RHC, SA, GA and MIMIC
    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=1000, sa_max_iters=1000, ga_max_iters=500, mimic_max_iters=100,
                         sa_init_temp=100, sa_exp_decay_rate=0.3, sa_min_temp=0.001,
                         ga_pop_size=300, ga_keep_pct=0.2,
                         mimic_pop_size=1500, mimic_keep_pct=0.4,
                         plot_name='Flip Flop', plot_ylabel='Fitness')


def four_peaks(length, random_seeds):
    """Define and experiment Four Peaks optimization problem.

        Four Peaks is an optimization problem presenting 2 global optima
        and 2 local optima.

        Args:
           length (int): problem length.
           random_seeds (list or ndarray): random seeds for get performances over multiple random runs.

        Returns:
           None.
        """

    # Define Four Peaks objective function and problem
    four_fitness = FourPeaks(t_pct=0.1)
    problem = DiscreteOpt(length=length, fitness_fn=four_fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)  # set fast MIMIC

    # Plot optimizations for RHC, SA, GA and MIMIC
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=10000, sa_max_iters=10000, ga_max_iters=250, mimic_max_iters=250,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.002, 0.1, 0.002), sa_min_temp=0.001,
                          ga_pop_size=1000, mimic_pop_size=1000, ga_keep_pct=0.1, mimic_keep_pct=0.2,
                          pop_sizes=np.arange(100, 1001, 100), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='Four Peaks', plot_ylabel='Fitness')

    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=7000, sa_max_iters=7000, ga_max_iters=250, mimic_max_iters=250,
                         sa_init_temp=100, sa_exp_decay_rate=0.02, sa_min_temp=0.001,
                         ga_pop_size=1000, ga_keep_pct=0.1,
                         mimic_pop_size=1000, mimic_keep_pct=0.2,
                         plot_name='Four Peaks', plot_ylabel='Fitness')


def neural_network(x_train, x_test, y_train, y_test, random_seeds):
    """Define and experiment the Neural Network weights optimization problem.

        Training Neural Networks weights can be done using GD and backpropation, but also another RO
        optimization algorithm, like RHC, SA or GA, can be used.

        Args:
          x_train (ndarray): training data.
          x_test (ndarray): test data.
          y_train (ndarray): training labels.
          y_test (ndarray): test labels.
          random_seeds (list or ndarray): random seeds for get performances over multiple random runs.

        Returns:
          None.
        """
    # Maximum iterations to run the Neural Network for
    iterations = np.array([i for i in range(1, 10)] + [10 * i for i in range(1, 20, 2)])

    # Plot performances for RHC, SA, GA and GD with Neural Networks
    nn.plot_nn_performances(x_train, y_train,
                            random_seeds=random_seeds,
                            rhc_max_iters=iterations, sa_max_iters=iterations,
                            ga_max_iters=iterations, gd_max_iters=iterations,
                            init_temp=100, exp_decay_rate=0.1, min_temp=0.001,
                            pop_size=100, mutation_prob=0.2)

    # Test performances for RHC, SA, GA and GD with Neural Networks
    nn.test_nn_performances(x_train, x_test, y_train, y_test,
                            random_seeds=random_seeds[0], max_iters=200,
                            init_temp=100, exp_decay_rate=0.1, min_temp=0.001,
                            pop_size=100, mutation_prob=0.2)


def travel_salesman(length, distances, random_seeds):
    """Define and experiment the Travel Salesman optimization problem.

        The Travel Salesman is a famous optimization problem presenting whose
        purpose is to find the shortest tout of N cities and visit each exactly once.

        Args:
           length (int): problem length.
           distances(list of tuples): list of inter-distances between each pair of cities.
           random_seeds (list or ndarray): random seeds for get performances over multiple random runs.

        Returns:
           None.
        """

    # Define Travel Salesman objective function and problem
    tsp_objective = TravellingSales(distances=distances)
    problem = TSPOpt(length=length, fitness_fn=tsp_objective, maximize=False)
    problem.set_mimic_fast_mode(True)  # set fast MIMIC

    # Plot optimizations for RHC, SA, GA and MIMIC
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=500, sa_max_iters=500, ga_max_iters=50, mimic_max_iters=10,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.005, 0.05, 0.005), sa_min_temp=0.001,
                          ga_pop_size=100, mimic_pop_size=700, ga_keep_pct=0.2, mimic_keep_pct=0.2,
                          pop_sizes=np.arange(100, 1001, 100), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='TSP', plot_ylabel='Cost')

    # Plot performances for RHC, SA, GA and MIMIC
    nn.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=500, sa_max_iters=500, ga_max_iters=50, mimic_max_iters=10,
                         sa_init_temp=100, sa_exp_decay_rate=0.03, sa_min_temp=0.001,
                         ga_pop_size=100, ga_keep_pct=0.2,
                         mimic_pop_size=700, mimic_keep_pct=0.2,
                         plot_name='TSP', plot_ylabel='Cost')


if __name__ == "__main__":

    random_seeds = [5 + 5 * i for i in range(5)]  # random seeds for get performances over multiple random runs

    # Define list of inter-distances between each pair of the following cities (in order from 0 to 9):
    # Rome, Florence, Barcelona, Paris, London, Amsterdam, Berlin, Prague, Budapest, Venice
    cities_distances = [(0, 1, 0.274), (0, 2, 1.367), (1, 2, 1.091), (0, 3, 1.422), (1, 3, 1.153), (2, 3, 1.038),
                        (0, 4, 1.870), (1, 4, 1.602), (2, 4, 1.495), (3, 4, 0.475), (0, 5, 1.652), (1, 5, 1.381),
                        (2, 5, 1.537), (3, 5, 0.515), (4, 5, 0.539), (0, 6, 1.504), (1, 6, 1.324), (2, 6, 1.862),
                        (3, 6, 1.060), (4, 6, 1.097), (5, 6, 0.664), (0, 7, 1.301), (1, 7, 1.031), (2, 7, 1.712),
                        (3, 7, 1.031), (4, 7, 1.261), (5, 7, 0.893), (6, 7, 0.350), (0, 8, 1.219), (1, 8, 0.948),
                        (2, 8, 1.923), (3, 8, 1.484), (4, 8, 1.723), (5, 8, 1.396), (6, 8, 0.872), (7, 8, 0.526),
                        (0, 9, 0.529), (1, 9, 0.258), (2, 9, 1.233), (3, 9, 1.137), (4, 9, 1.560), (5, 9, 1.343),
                        (6, 9, 1.131), (7, 9, 0.816), (8, 9, 0.704)]

    # Experiment the Travel Salesman Problem, Flip Flop and Four Peaks with RHC, SA, GA and MIMIC
    travel_salesman(length=10, distances=cities_distances, random_seeds=random_seeds)
    flip_plop(length=100, random_seeds=random_seeds)
    four_peaks(length=100, random_seeds=random_seeds)

    # Experiment Neural Networks optimization with RHC, SA, GA and GD on the WDBC dataset
    x_train, x_test, y_train, y_test = load_dataset(split_percentage=0.2)
    neural_network(x_train, x_test, y_train, y_test, random_seeds=random_seeds)
