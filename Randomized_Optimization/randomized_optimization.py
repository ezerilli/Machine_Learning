# Randomized Optimization utils

import numpy as np
import matplotlib.pyplot as plt
import utils

from mlrose.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose.algorithms.decay import ExpDecay


IMAGE_DIR = 'images/'


def plot_ga_mimic_optimization(problem, param_name, random_seeds, **kwargs):
    """Plot objective function vs parameter for GA and MIMIC.

        Args:
         problem (DiscreteOpt): mlrose Discrete Optimization problem to run.
         param_name (string): GA and MIMIC param to plot objective vs, 'pop_size' or 'keep_pct'.
         random_seeds (list): random seeds to use for averaging results over multiple random runs.
         kwargs (dict): additional arguments to pass for curves plotting:
                    - pop_sizes (list or ndarray): lits or array of population sizes to objective plot vs.
                    - keep_pcts (list or ndarray): lits or array of keep percentages to objective plot vs.
                    - ga_pop_size (int): GA population size.
                    - ga_keep_pct (float): GA keep percentage.
                    - ga_max_iters (int): GA maximum number of iterations.
                    - mimic_pop_size (int): MIMIC population size.
                    - mimic_keep_pct (float): MIMIC keep percentage.
                    - mimic_max_iters (int): MIMIC maximum number of iterations.
                    - plot_name (string): name of the plot.
                    - plot_ylabel (string): y axis label.

        Returns:
         None.
        """

    ga_curve, mimic_curve = [], []  # curves to plot for GA and MIMIC

    # Initialize, for GA and MIMIC, the parameter we don't have to loop through and label for plotting
    if param_name == 'keep_pct':
        ga_pop_size = kwargs['ga_pop_size']
        mimic_pop_size = kwargs['mimic_pop_size']
        label = 'Percentage to keep'
    elif param_name == 'pop_size':
        ga_keep_pct = kwargs['ga_keep_pct']
        mimic_keep_pct = kwargs['mimic_keep_pct']
        label = 'Population size'
    else:
        raise Exception('Param name has to be either pop_size or keep_pct')  # raise exception if invalid entry

    # For all parameters
    for param in kwargs[param_name + 's']:

        print('\nGA & MIMIC: {} = {:.3f}'.format(param_name, param))
        ga_objectives, mimic_objectives = [], []  # list of results from multiple random runs for current parameter

        # Initialize, for GA and MIMIC, the parameter we have to loop through
        if param_name == 'keep_pct':
            ga_keep_pct = param
            mimic_keep_pct = param
        elif param_name == 'pop_size':
            ga_pop_size = int(param)
            mimic_pop_size = int(param)

        # For multiple random runs
        for random_seed in random_seeds:

            # Run GA and get best state and objective found
            best_state, best_objective, _ = genetic_alg(problem,
                                                        pop_size=ga_pop_size,  # population size
                                                        pop_breed_percent=1.0 - ga_keep_pct,  # percentage to breed
                                                        max_attempts=kwargs['ga_max_iters'],  # unlimited attempts
                                                        max_iters=kwargs['ga_max_iters'],
                                                        curve=False, random_state=random_seed)
            ga_objectives.append(best_objective)
            print('GA: best_objective = {:.3f}'.format(best_objective))

            # Run MIMIC and get best state and objective found
            best_state, best_objective, _ = mimic(problem,
                                                  pop_size=mimic_pop_size,  # population size
                                                  keep_pct=mimic_keep_pct,  # percentage to keep
                                                  max_attempts=kwargs['mimic_max_iters'],  # unlimited attempts
                                                  max_iters=kwargs['mimic_max_iters'],
                                                  curve=False, random_state=random_seed)
            mimic_objectives.append(best_objective)
            print('MIMIC: best_objective = {:.3f}'.format(best_objective))

        # Append random run to GA and MIMIC curves
        ga_curve.append(ga_objectives)
        mimic_curve.append(mimic_objectives)

    # Plot, set title and labels
    plt.figure()
    utils.plot_helper(x_axis=kwargs[param_name + 's'], y_axis=np.array(ga_curve).T, label='GA')
    utils.plot_helper(x_axis=kwargs[param_name + 's'], y_axis=np.array(mimic_curve).T, label='MIMIC')
    utils.set_plot_title_labels(title='{} - Objective vs. {}}'.format(kwargs['plot_name'], label),
                                x_label=label,
                                y_label=kwargs['plot_ylabel'])

    # Save figure
    plt.savefig(IMAGE_DIR + '{}_objective_vs_{}'.format(kwargs['plot_name'], param_name))


def plot_sa_optimization(problem, random_seeds, **kwargs):
    """Plot objective function vs temperature for SA.

        Args:
         problem (DiscreteOpt): mlrose Discrete Optimization problem to run.
         random_seeds (list): random seeds to use for averaging results over multiple random runs.
         kwargs (dict): additional arguments to pass for curves plotting:
                    - sa_decay_rates (list or ndarray): lits or array of exponential decay rates to objective plot vs.
                    - sa_init_temp (float): SA initial temperature.
                    - sa_min_temp (float): SA minimum temperature.
                    - sa_max_iters (int): SA maximum number of iterations.
                    - plot_name (string): name of the plot.
                    - plot_ylabel (string): y axis label.

        Returns:
         None.
        """

    sa_curve = []  # curve to plot for SA

    # For all temperature exponential decay rate
    for exp_decay_rate in kwargs['sa_decay_rates']:

        # Define exponential decay schedule
        print('\nSA: exp decay rate = {:.3f}'.format(exp_decay_rate))
        exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                             exp_const=exp_decay_rate,
                             min_temp=kwargs['sa_min_temp'])

        sa_objectives = []  # list of results from multiple random runs for current rate

        # For multiple random runs
        for random_seed in random_seeds:

            # Run SA and get best state and objective found
            best_state, best_objective, _ = simulated_annealing(problem,
                                                                schedule=exp_decay,
                                                                max_attempts=kwargs['sa_max_iters'],
                                                                max_iters=kwargs['sa_max_iters'],
                                                                curve=False, random_state=random_seed)
            sa_objectives.append(best_objective)
            print('SA: best_fitness = {:.3f}'.format(best_objective))

        sa_curve.append(sa_objectives)  # append random run to SA curve

    # Plot, set title and labels
    plt.figure()
    utils.plot_helper(x_axis=kwargs['sa_exp_consts'], y_axis=np.array(sa_curve).T, label='SA')
    utils.set_plot_title_labels(title='{} - Objective vs. temperature exponential decay rate'.format(kwargs['plot_name']),
                                x_label='Exponential decay rate',
                                y_label=kwargs['plot_ylabel'])

    # Save figure
    plt.savefig(IMAGE_DIR + '{}_objective_vs_temp'.format(kwargs['plot_name']))


def plot_optimizations(problem, random_seeds, **kwargs):
    """Plot optimizations for SA temperature and GA and MIMIC population sizes and fractions to keep.

       Args:
        problem (DiscreteOpt): mlrose Discrete Optimization problem to run.
        random_seeds (list): random seeds to use for averaging results over multiple random runs.
        kwargs (dict): additional arguments to pass for curves plotting:
                   - sa_decay_rates (list or ndarray): lits or array of exponential decay rates to objective plot vs.
                   - sa_init_temp (float): SA initial temperature.
                   - sa_min_temp (float): SA minimum temperature.
                   - sa_max_iters (int): SA maximum number of iterations.
                   - pop_sizes (list or ndarray): lits or array of population sizes to objective plot vs.
                   - keep_pcts (list or ndarray): lits or array of keep percentages to objective plot vs.
                   - ga_pop_size (int): GA population size.
                   - ga_keep_pct (float): GA keep percentage.
                   - ga_max_iters (int): GA maximum number of iterations.
                   - mimic_pop_size (int): MIMIC population size.
                   - mimic_keep_pct (float): MIMIC keep percentage.
                   - mimic_max_iters (int): MIMIC maximum number of iterations.
                   - plot_name (string): name of the plot.
                   - plot_ylabel (string): y axis label.

       Returns:
        None.
       """
    plot_sa_optimization(problem, random_seeds, **kwargs)
    plot_ga_mimic_optimization(problem, 'pop_size', random_seeds, **kwargs)
    plot_ga_mimic_optimization(problem, 'keep_pct', random_seeds, **kwargs)


def plot_performances(problem, random_seeds, **kwargs):
    """Plot performances for RHC, SA, GA and MIMIC.

       Args:
        problem (DiscreteOpt): mlrose Discrete Optimization problem to run.
        random_seeds (list): random seeds to use for averaging results over multiple random runs.
        kwargs (dict): additional arguments to pass for curves plotting:
                   - sa_init_temp (float): SA initial temperature.
                   - sa_min_temp (float): SA minimum temperature.
                   - sa_exp_decay_rate (float): SA temperature exponential decay rate.
                   - sa_max_iters (int): SA maximum number of iterations.
                   - pop_sizes (list or ndarray): lits or array of population sizes to objective plot vs.
                   - keep_pcts (list or ndarray): lits or array of keep percentages to objective plot vs.
                   - ga_pop_size (int): GA population size.
                   - ga_keep_pct (float): GA keep percentage.
                   - ga_max_iters (int): GA maximum number of iterations.
                   - mimic_pop_size (int): MIMIC population size.
                   - mimic_keep_pct (float): MIMIC keep percentage.
                   - mimic_max_iters (int): MIMIC maximum number of iterations.
                   - plot_name (string): name of the plot.
                   - plot_ylabel (string): y axis label.

       Returns:
        None.
       """

    # Initialize lists of objectives curves and time curves
    rhc_objectives, sa_objectives, ga_objectives, mimic_objectives = [], [], [], []
    rhc_times, sa_times, ga_times, mimic_times = [], [], [], []

    # Set an exponential decay schedule for SA
    exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                         exp_const=kwargs['sa_exp_decay_rate'],
                         min_temp=kwargs['sa_min_temp'])

    # For multiple random runs
    for random_seed in random_seeds:

        # Run RHC and get best state and objective found
        _, best_objective, objective_curve = random_hill_climb(problem,
                                                               max_attempts=kwargs['rhc_max_iters'],
                                                               max_iters=kwargs['rhc_max_iters'],
                                                               curve=True, random_state=random_seed,
                                                               state_fitness_callback=utils.time_callback,
                                                               callback_user_info=[])

        rhc_objectives.append(objective_curve)
        rhc_times.append(times)
        print('RHC: best_objective = {:.3f}'.format(best_objective))

        # Run SA and get best state and objective found
        _, best_objective, objective_curve = simulated_annealing(problem,
                                                                 schedule=exp_decay,
                                                                 max_attempts=kwargs['sa_max_iters'],
                                                                 max_iters=kwargs['sa_max_iters'],
                                                                 curve=True, random_state=random_seed,
                                                                 state_fitness_callback=utils.time_callback,
                                                                 callback_user_info=[])

        sa_objectives.append(objective_curve)
        sa_times.append(times)
        print('SA: best_objective = {:.3f}'.format(best_objective))

        # Run GA and get best state and objective found
        _, best_objective, objective_curve = genetic_alg(problem,
                                                         pop_size=kwargs['ga_pop_size'],
                                                         pop_breed_percent=1.0 - kwargs['ga_keep_pct'],
                                                         max_attempts=kwargs['ga_max_iters'],
                                                         max_iters=kwargs['ga_max_iters'],
                                                         curve=True, random_state=random_seed,
                                                         state_fitness_callback=utils.time_callback,
                                                         callback_user_info=[])

        ga_objectives.append(objective_curve)
        ga_times.append(times)
        print('GA: best_objective = {:.3f}'.format(best_objective))

        # Run MIMIC and get best state and objective found
        _, best_objective, objective_curve = mimic(problem,
                                                   pop_size=kwargs['mimic_pop_size'],
                                                   keep_pct=kwargs['mimic_keep_pct'],
                                                   max_attempts=kwargs['mimic_max_iters'],
                                                   max_iters=kwargs['mimic_max_iters'],
                                                   curve=True, random_state=random_seed,
                                                   state_fitness_callback=utils.time_callback,
                                                   callback_user_info=[])

        mimic_objectives.append(objective_curve)
        mimic_times.append(times)
        print('MIMIC: best_objective = {:.3f}'.format(best_objective))

    # Array of iterations to plot objectives vs. for RHC, SA, GA and MIMIC
    rhc_iterations = np.arange(1, kwargs['rhc_max_iters']+1)
    sa_iterations = np.arange(1, kwargs['sa_max_iters']+1)
    ga_iterations = np.arange(1, kwargs['ga_max_iters']+1)
    mimic_iterations = np.arange(1, kwargs['mimic_max_iters']+1)

    # Plot objective curves, set title and labels
    plt.figure()
    utils.plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_objectives), label='RHC')
    utils.plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_objectives), label='SA')
    utils.plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_objectives), label='GA')
    utils.plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_objectives), label='MIMIC')
    utils.set_plot_title_labels(title='{} - Objective versus iterations'.format(kwargs['plot_name']),
                                x_label='Iterations',
                                y_label=kwargs['plot_ylabel'])

    # Save figure
    plt.savefig(IMAGE_DIR + '{}_objective_vs_iterations'.format(kwargs['plot_name']))

    # Plot times, set title and labels
    plt.figure()
    utils.plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_times), label='RHC')
    utils.plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_times), label='SA')
    utils.plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_times), label='GA')
    utils.plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_times), label='MIMIC')
    utils.set_plot_title_labels(title='{} - Time versus iterations'.format(kwargs['plot_name']),
                                x_label='Iterations',
                                y_label='Time (seconds)')

    # Save figure
    plt.savefig(IMAGE_DIR + '{}_time_vs_iterations'.format(kwargs['plot_name']))
