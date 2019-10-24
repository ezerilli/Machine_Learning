
import numpy as np
import matplotlib.pyplot as plt
import time

from mlrose.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose.algorithms.decay import ExpDecay
from mlrose.neural import NeuralNetwork
from mlrose.opt_probs import TSPOpt, DiscreteOpt
from mlrose.fitness import TravellingSales, FlipFlop, FourPeaks

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'
start_time = 0.
times = []


def load_dataset(split_percentage=0.2):

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


def flip_plot(length, random_seeds):

    flip_flop_objective = FlipFlop()
    problem = DiscreteOpt(length=length, fitness_fn=flip_flop_objective, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)

    plot_optimizations(problem=problem,
                       random_seeds=random_seeds,
                       rhc_max_iters=1000, sa_max_iters=1000, ga_max_iters=250, mimic_max_iters=50,
                       sa_init_temp=100, sa_exp_consts=np.arange(0.05, 2.01, 0.05), sa_min_temp=0.001,
                       ga_pop_size=300, mimic_pop_size=1500, ga_keep_pct=0.2, mimic_keep_pct=0.4,
                       pop_sizes=np.arange(100, 2001, 200), keep_pcts=np.arange(0.1, 0.81, 0.1),
                       plot_name='Flip Flop', plot_ylabel='Fitness')

    plot_performances(problem=problem,
                      random_seeds=random_seeds,
                      rhc_max_iters=1000, sa_max_iters=1000, ga_max_iters=500, mimic_max_iters=100,
                      sa_init_temp=100, sa_exp_const=0.3, sa_min_temp=0.001,
                      ga_pop_size=300, ga_keep_pct=0.2,
                      mimic_pop_size=1500, mimic_keep_pct=0.4,
                      plot_name='Flip Flop', plot_ylabel='Fitness')


def four_peaks(length, random_seeds):

    four_fitness = FourPeaks(t_pct=0.1)
    problem = DiscreteOpt(length=length, fitness_fn=four_fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)

    plot_optimizations(problem=problem,
                       random_seeds=random_seeds,
                       rhc_max_iters=10000, sa_max_iters=10000, ga_max_iters=250, mimic_max_iters=250,
                       sa_init_temp=100, sa_exp_consts=np.arange(0.002, 0.1, 0.002), sa_min_temp=0.001,
                       ga_pop_size=1000, mimic_pop_size=1000, ga_keep_pct=0.1, mimic_keep_pct=0.2,
                       pop_sizes=np.arange(100, 1001, 100), keep_pcts=np.arange(0.1, 0.81, 0.1),
                       plot_name='Four Peaks', plot_ylabel='Fitness')

    plot_performances(problem=problem,
                      random_seeds=random_seeds,
                      rhc_max_iters=7000, sa_max_iters=7000, ga_max_iters=250, mimic_max_iters=250,
                      sa_init_temp=100, sa_exp_const=0.02, sa_min_temp=0.001,
                      ga_pop_size=1000, ga_keep_pct=0.1,
                      mimic_pop_size=1000, mimic_keep_pct=0.2,
                      plot_name='Four Peaks', plot_ylabel='Fitness')


def plot_helper(x_axis, y_axis, label):

    y_mean, y_std = np.mean(y_axis, axis=0), np.std(y_axis, axis=0)
    plot = plt.plot(x_axis, y_mean, label=label)
    plt.fill_between(x_axis, y_mean - y_std, y_mean + y_std, alpha=0.1, color=plot[0].get_color())


def plot_nn_performances(x_train, x_test, y_train, y_test, random_seeds, **kwargs):

    rhc_train_curves, sa_train_curves, ga_train_curves, gd_train_curves = [], [], [], []
    rhc_val_curves, sa_val_curves, ga_val_curves, gd_val_curves = [], [], [], []
    rhc_times, sa_times, ga_times, gd_times = [], [], [], []

    exp_decay = ExpDecay(init_temp=kwargs['init_temp'],
                         exp_const=kwargs['exp_const'],
                         min_temp=kwargs['min_temp'])

    for random_seed in random_seeds:

        global start_time

        rhc_train, sa_train, ga_train, gd_train = [], [], [], []
        rhc_val, sa_val, ga_val, gd_val = [], [], [], []
        rhc_time, sa_time, ga_time, gd_time = [], [], [], []

        x_train_fold, x_val_fold, y_train_fold, y_val_fold = train_test_split(x_train, y_train,
                                                                              test_size=0.2, shuffle=True,
                                                                              random_state=random_seed,
                                                                              stratify=y_train)

        for max_iter in kwargs['rhc_max_iters']:
            print(max_iter)

            nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                               algorithm='random_hill_climb', max_iters=int(max_iter),
                               bias=True, is_classifier=True, learning_rate=0.001,
                               early_stopping=False, clip_max=1e10,
                               max_attempts=int(max_iter), random_state=random_seed, curve=False)

            start_time = time.time()
            nn.fit(x_train_fold, y_train_fold)
            rhc_time.append(time.time() - start_time)

            train_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
            rhc_train.append(train_loss)
            val_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
            rhc_val.append(val_loss)

        rhc_train_curves.append(rhc_train)
        rhc_val_curves.append(rhc_val)
        rhc_times.append(rhc_time)
        print('\nRHC - train loss = {}, val loss = {}'.format(train_loss, val_loss))

        for max_iter in kwargs['sa_max_iters']:
            print(max_iter)
            nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                               algorithm='simulated_annealing', max_iters=int(max_iter),
                               bias=True, is_classifier=True, learning_rate=0.001,
                               early_stopping=False, clip_max=1e10, schedule=exp_decay,
                               max_attempts=int(max_iter), random_state=random_seed, curve=False)

            start_time = time.time()
            nn.fit(x_train_fold, y_train_fold)
            sa_time.append(time.time() - start_time)

            train_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
            sa_train.append(train_loss)
            val_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
            sa_val.append(val_loss)

        sa_train_curves.append(sa_train)
        sa_val_curves.append(sa_val)
        sa_times.append(sa_time)
        print('\nSA - train loss = {}, val loss = {}'.format(train_loss, val_loss))

        for max_iter in kwargs['ga_max_iters']:
            print(max_iter)
            nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                               algorithm='genetic_alg', max_iters=int(max_iter),
                               bias=True, is_classifier=True, learning_rate=0.001,
                               early_stopping=False, clip_max=1e10,
                               pop_size=kwargs['pop_size'], mutation_prob=kwargs['mutation_prob'],
                               max_attempts=int(max_iter), random_state=random_seed, curve=False)

            start_time = time.time()
            nn.fit(x_train_fold, y_train_fold)
            ga_time.append(time.time() - start_time)

            train_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
            ga_train.append(train_loss)
            val_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
            ga_val.append(val_loss)

        ga_train_curves.append(ga_train)
        ga_val_curves.append(ga_val)
        ga_times.append(ga_time)
        print('\nGA - train loss = {}, val loss = {}'.format(train_loss, val_loss))

        for max_iter in kwargs['gd_max_iters']:
            print(max_iter)
            nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                               algorithm='gradient_descent', max_iters=int(max_iter),
                               bias=True, is_classifier=True, learning_rate=0.001,
                               early_stopping=False, clip_max=1e10,
                               max_attempts=int(max_iter), random_state=random_seed, curve=False)

            start_time = time.time()
            nn.fit(x_train_fold, y_train_fold)
            gd_time.append(time.time() - start_time)

            train_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
            gd_train.append(train_loss)
            val_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
            gd_val.append(val_loss)

        gd_train_curves.append(gd_train)
        gd_val_curves.append(gd_val)
        gd_times.append(gd_time)
        print('\nGD - train loss = {}, val loss = {}'.format(train_loss, val_loss))

    plt.figure()
    plot_helper(x_axis=kwargs['rhc_max_iters'], y_axis=np.array(rhc_train_curves), label='RHC train')
    plot_helper(x_axis=kwargs['rhc_max_iters'], y_axis=np.array(rhc_val_curves), label='RHC val')
    plot_helper(x_axis=kwargs['sa_max_iters'], y_axis=np.array(sa_train_curves), label='SA train')
    plot_helper(x_axis=kwargs['sa_max_iters'], y_axis=np.array(sa_val_curves), label='SA val')
    plot_helper(x_axis=kwargs['ga_max_iters'], y_axis=np.array(ga_train_curves), label='GA train')
    plot_helper(x_axis=kwargs['ga_max_iters'], y_axis=np.array(ga_val_curves), label='GA val')
    plot_helper(x_axis=kwargs['gd_max_iters'], y_axis=np.array(gd_train_curves), label='GD train')
    plot_helper(x_axis=kwargs['gd_max_iters'], y_axis=np.array(gd_val_curves), label='GD val')
    plt.title('Neural Network - Loss versus iterations')
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(IMAGE_DIR + 'nn_objective_vs_iterations')

    plt.figure()
    plot_helper(x_axis=kwargs['rhc_max_iters'], y_axis=np.array(rhc_times), label='RHC')
    plot_helper(x_axis=kwargs['sa_max_iters'], y_axis=np.array(sa_times), label='SA')
    plot_helper(x_axis=kwargs['ga_max_iters'], y_axis=np.array(ga_times), label='GA')
    plot_helper(x_axis=kwargs['gd_max_iters'], y_axis=np.array(gd_times), label='GD')
    plt.title('Neural Network - Time versus iterations')
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('Time (seconds)')
    plt.savefig(IMAGE_DIR + 'nn_time_vs_iterations')


def plot_objective_vs_keep_pct(problem, random_seeds, **kwargs):

    ga_curves, mimic_curves = [], []
    for keep_pct in kwargs['keep_pcts']:
        print('\nGA & MIMIC: keep_pct = {:.3f}'.format(keep_pct))

        ga_objectives, mimic_objectives = [], []
        for random_seed in random_seeds:
            best_state, best_objective, _ = genetic_alg(problem,
                                                        pop_size=kwargs['ga_pop_size'],
                                                        pop_breed_percent=1.0 - keep_pct,
                                                        max_attempts=kwargs['ga_max_iters'],
                                                        max_iters=kwargs['ga_max_iters'],
                                                        curve=False, random_state=random_seed)
            ga_objectives.append(best_objective)
            print('GA: best_objective = {:.3f}'.format(best_objective))

            best_state, best_objective, _ = mimic(problem,
                                                  pop_size=kwargs['mimic_pop_size'],
                                                  keep_pct=keep_pct,
                                                  max_attempts=kwargs['mimic_max_iters'],
                                                  max_iters=kwargs['mimic_max_iters'],
                                                  curve=False, random_state=random_seed)
            mimic_objectives.append(best_objective)
            print('MIMIC: best_objective = {:.3f}'.format(best_objective))

        ga_curves.append(ga_objectives)
        mimic_curves.append(mimic_objectives)

    plt.figure()
    plot_helper(x_axis=kwargs['keep_pcts'], y_axis=np.array(ga_curves).T, label='GA')
    plot_helper(x_axis=kwargs['keep_pcts'], y_axis=np.array(mimic_curves).T, label='MIMIC')
    plt.title('{} - Objective vs. Fraction to Keep'.format(kwargs['plot_name']))
    plt.legend(loc='best')
    plt.xlabel('Fraction to Keep')
    plt.ylabel(kwargs['plot_ylabel'])
    plt.savefig(IMAGE_DIR + '{}_objective_vs_keep'.format(kwargs['plot_name']))


def plot_objective_vs_pop_size(problem, random_seeds, **kwargs):

    ga_curves, mimic_curves = [], []
    for pop_size in kwargs['pop_sizes']:
        print('\nGA & MIMIC: pop size = {}'.format(pop_size))

        ga_objectives, mimic_objectives = [], []
        for random_seed in random_seeds:
            best_state, best_objective, _ = genetic_alg(problem,
                                                        pop_size=int(pop_size),
                                                        pop_breed_percent=1.0 - kwargs['ga_keep_pct'],
                                                        max_attempts=kwargs['ga_max_iters'],
                                                        max_iters=kwargs['ga_max_iters'],
                                                        curve=False, random_state=random_seed)
            ga_objectives.append(best_objective)
            print('GA: best_objective = {:.3f}'.format(best_objective))

            best_state, best_objective, _ = mimic(problem,
                                                  pop_size=int(pop_size),
                                                  keep_pct=kwargs['mimic_keep_pct'],
                                                  max_attempts=kwargs['mimic_max_iters'],
                                                  max_iters=kwargs['mimic_max_iters'],
                                                  curve=False, random_state=random_seed)
            mimic_objectives.append(best_objective)
            print('MIMIC: best_objective = {:.3f}'.format(best_objective))

        ga_curves.append(ga_objectives)
        mimic_curves.append(mimic_objectives)

    plt.figure()
    plot_helper(x_axis=kwargs['pop_sizes'], y_axis=np.array(ga_curves).T, label='GA')
    plot_helper(x_axis=kwargs['pop_sizes'], y_axis=np.array(mimic_curves).T, label='MIMIC')
    plt.title('{} - Objective vs. population size'.format(kwargs['plot_name']))
    plt.legend(loc='best')
    plt.xlabel('Population Size')
    plt.ylabel(kwargs['plot_ylabel'])
    plt.savefig(IMAGE_DIR + '{}_objective_vs_pop'.format(kwargs['plot_name']))


def plot_objective_vs_temperature(problem, random_seeds, **kwargs):

    sa_curves = []
    for exp_const in kwargs['sa_exp_consts']:
        print('\nSA: exp decay rate = {:.3f}'.format(exp_const))

        exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                             exp_const=exp_const,
                             min_temp=kwargs['sa_min_temp'])

        sa_objectives = []
        for random_seed in random_seeds:
            best_state, best_objective, _ = simulated_annealing(problem,
                                                                schedule=exp_decay,
                                                                max_attempts=kwargs['sa_max_iters'],
                                                                max_iters=kwargs['sa_max_iters'],
                                                                curve=False, random_state=random_seed)
            sa_objectives.append(best_objective)
            print('SA: best_fitness = {:.3f}'.format(best_objective))

        sa_curves.append(sa_objectives)

    plt.figure()
    plot_helper(x_axis=kwargs['sa_exp_consts'], y_axis=np.array(sa_curves).T, label='SA')
    plt.title('{} - Objective vs. temperature exponential decay rate'.format(kwargs['plot_name']))
    plt.legend(loc='best')
    plt.xlabel('Exponential decay rate')
    plt.ylabel(kwargs['plot_ylabel'])
    plt.savefig(IMAGE_DIR + '{}_objective_vs_temp'.format(kwargs['plot_name']))


def plot_optimizations(problem, random_seeds, **kwargs):

    plot_objective_vs_temperature(problem, random_seeds, **kwargs)
    plot_objective_vs_pop_size(problem, random_seeds, **kwargs)
    plot_objective_vs_keep_pct(problem, random_seeds, **kwargs)


def plot_performances(problem, random_seeds, **kwargs):

    rhc_curves, sa_curves, ga_curves, mimic_curves = [], [], [], []
    rhc_times, sa_times, ga_times, mimic_times = [], [], [], []
    rhc_times2, sa_times2, ga_times2, mimic_times2 = [], [], [], []

    exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                         exp_const=kwargs['sa_exp_const'],
                         min_temp=kwargs['sa_min_temp'])

    for random_seed in random_seeds:

        best_state, best_objective, objective_curve = random_hill_climb(problem,
                                                                        max_attempts=kwargs['rhc_max_iters'],
                                                                        max_iters=kwargs['rhc_max_iters'],
                                                                        curve=True, random_state=random_seed,
                                                                        state_fitness_callback=time_callback,
                                                                        callback_user_info=rhc_times2)

        rhc_curves.append(objective_curve)
        rhc_times.append(times)
        print('RHC: best_objective = {:.3f}'.format(best_objective))

        best_state, best_objective, objective_curve = simulated_annealing(problem,
                                                                          schedule=exp_decay,
                                                                          max_attempts=kwargs['sa_max_iters'],
                                                                          max_iters=kwargs['sa_max_iters'],
                                                                          curve=True, random_state=random_seed,
                                                                          state_fitness_callback=time_callback,
                                                                          callback_user_info=sa_times2)

        sa_curves.append(objective_curve)
        sa_times.append(times)
        print('SA: best_objective = {:.3f}'.format(best_objective))

        best_state, best_objective, objective_curve = genetic_alg(problem,
                                                                  pop_size=kwargs['ga_pop_size'],
                                                                  pop_breed_percent=1.0 - kwargs['ga_keep_pct'],
                                                                  max_attempts=kwargs['ga_max_iters'],
                                                                  max_iters=kwargs['ga_max_iters'],
                                                                  curve=True, random_state=random_seed,
                                                                  state_fitness_callback=time_callback,
                                                                  callback_user_info=ga_times2)

        ga_curves.append(objective_curve)
        ga_times.append(times)
        print('GA: best_objective = {:.3f}'.format(best_objective))

        best_state, best_objective, objective_curve = mimic(problem,
                                                            pop_size=kwargs['mimic_pop_size'],
                                                            keep_pct=kwargs['mimic_keep_pct'],
                                                            max_attempts=kwargs['mimic_max_iters'],
                                                            max_iters=kwargs['mimic_max_iters'],
                                                            curve=True, random_state=random_seed,
                                                            state_fitness_callback=time_callback,
                                                            callback_user_info=mimic_times2)

        mimic_curves.append(objective_curve)
        mimic_times.append(times)
        print('MIMIC: best_objective = {:.3f}'.format(best_objective))

    rhc_iterations = np.arange(1, kwargs['rhc_max_iters']+1)
    sa_iterations = np.arange(1, kwargs['sa_max_iters']+1)
    ga_iterations = np.arange(1, kwargs['ga_max_iters']+1)
    mimic_iterations = np.arange(1, kwargs['mimic_max_iters']+1)

    plt.figure()
    plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_curves), label='RHC')
    plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_curves), label='SA')
    plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_curves), label='GA')
    plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_curves), label='MIMIC')
    plt.title('{} - Objective versus iterations'.format(kwargs['plot_name']))
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel(kwargs['plot_ylabel'])
    plt.savefig(IMAGE_DIR + '{}_objective_vs_iterations'.format(kwargs['plot_name']))

    plt.figure()
    plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_times), label='RHC')
    plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_times), label='SA')
    plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_times), label='GA')
    plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_times), label='MIMIC')
    plt.title('{} - Time versus iterations'.format(kwargs['plot_name']))
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('Time (seconds)')
    plt.savefig(IMAGE_DIR + '{}_time_vs_iterations'.format(kwargs['plot_name']))


def neural_network(x_train, x_test, y_train, y_test, random_seeds):

    iterations = np.array([i for i in range(1, 10)] + [10 * i for i in range(1, 20, 2)])

    # plot_nn_performances(x_train, x_test, y_train, y_test,
    #                      random_seeds=random_seeds,
    #                      rhc_max_iters=iterations, sa_max_iters=iterations,
    #                      ga_max_iters=iterations, gd_max_iters=iterations,
    #                      init_temp=100, exp_const=0.1, min_temp=0.001,
    #                      pop_size=100, mutation_prob=0.2)

    test_nn_performances(x_train, x_test, y_train, y_test,
                         random_seeds=random_seeds, max_iters=200,
                         init_temp=100, exp_const=0.1, min_temp=0.001,
                         pop_size=100, mutation_prob=0.2)


def test_nn_performances(x_train, x_test, y_train, y_test, random_seeds, **kwargs):

    exp_decay = ExpDecay(init_temp=kwargs['init_temp'],
                         exp_const=kwargs['exp_const'],
                         min_temp=kwargs['min_temp'])

    rhc_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                           algorithm='random_hill_climb', max_iters=kwargs['max_iters'],
                           bias=True, is_classifier=True, learning_rate=0.001,
                           early_stopping=False, clip_max=1e10,
                           max_attempts=kwargs['max_iters'], random_state=random_seeds[0], curve=False)

    sa_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                          algorithm='simulated_annealing', max_iters=kwargs['max_iters'],
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10, schedule=exp_decay,
                          max_attempts=kwargs['max_iters'], random_state=random_seeds[0], curve=False)

    ga_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                          algorithm='genetic_alg', max_iters=kwargs['max_iters'],
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10,
                          pop_size=kwargs['pop_size'], mutation_prob=kwargs['mutation_prob'],
                          max_attempts=kwargs['max_iters'], random_state=random_seeds[0], curve=False)

    gd_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                          algorithm='gradient_descent', max_iters=kwargs['max_iters'],
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10,
                          max_attempts=kwargs['max_iters'], random_state=random_seeds[0], curve=False)

    rhc_nn.fit(x_train, y_train)
    sa_nn.fit(x_train, y_train)
    ga_nn.fit(x_train, y_train)
    gd_nn.fit(x_train, y_train)

    print('RHC test classification report = \n {}'.format(classification_report(y_test, rhc_nn.predict(x_test))))
    print('SA test classification report = \n {}'.format(classification_report(y_test, sa_nn.predict(x_test))))
    print('GA test classification report = \n {}'.format(classification_report(y_test, ga_nn.predict(x_test))))
    print('GD test classification report = \n {}'.format(classification_report(y_test, gd_nn.predict(x_test))))


def time_callback(iteration, attempt=None, done=None, state=None, fitness=None, curve=None, user_data=None):
    global start_time, times
    if iteration == 0:
        start_time = time.time()
        times = []
    else:
        times.append(time.time() - start_time)
    return True


def travel_salesman(length, distances, random_seeds):

    tsp_objective = TravellingSales(distances=distances)
    problem = TSPOpt(length=length, fitness_fn=tsp_objective, maximize=False)
    problem.set_mimic_fast_mode(True)

    plot_optimizations(problem=problem,
                       random_seeds=random_seeds,
                       rhc_max_iters=500, sa_max_iters=500, ga_max_iters=50, mimic_max_iters=10,
                       sa_init_temp=100, sa_exp_consts=np.arange(0.005, 0.05, 0.005), sa_min_temp=0.001,
                       ga_pop_size=100, mimic_pop_size=700, ga_keep_pct=0.2, mimic_keep_pct=0.2,
                       pop_sizes=np.arange(100, 1001, 100), keep_pcts=np.arange(0.1, 0.81, 0.1),
                       plot_name='TSP', plot_ylabel='Cost')

    plot_performances(problem=problem,
                      random_seeds=random_seeds,
                      rhc_max_iters=500, sa_max_iters=500, ga_max_iters=50, mimic_max_iters=10,
                      sa_init_temp=100, sa_exp_const=0.03, sa_min_temp=0.001,
                      ga_pop_size=100, ga_keep_pct=0.2,
                      mimic_pop_size=700, mimic_keep_pct=0.2,
                      plot_name='TSP', plot_ylabel='Cost')


random_seeds = [5 + 5 * i for i in range(5)]
cities_distances = [(0, 1, 0.274), (0, 2, 1.367), (1, 2, 1.091), (0, 3, 1.422), (1, 3, 1.153), (2, 3, 1.038),
                    (0, 4, 1.870), (1, 4, 1.602), (2, 4, 1.495), (3, 4, 0.475), (0, 5, 1.652), (1, 5, 1.381),
                    (2, 5, 1.537), (3, 5, 0.515), (4, 5, 0.539), (0, 6, 1.504), (1, 6, 1.324), (2, 6, 1.862),
                    (3, 6, 1.060), (4, 6, 1.097), (5, 6, 0.664), (0, 7, 1.301), (1, 7, 1.031), (2, 7, 1.712),
                    (3, 7, 1.031), (4, 7, 1.261), (5, 7, 0.893), (6, 7, 0.350), (0, 8, 1.219), (1, 8, 0.948),
                    (2, 8, 1.923), (3, 8, 1.484), (4, 8, 1.723), (5, 8, 1.396), (6, 8, 0.872), (7, 8, 0.526),
                    (0, 9, 0.529), (1, 9, 0.258), (2, 9, 1.233), (3, 9, 1.137), (4, 9, 1.560), (5, 9, 1.343),
                    (6, 9, 1.131), (7, 9, 0.816), (8, 9, 0.704)]

# travel_salesman(length=10, distances=cities_distances, random_seeds=random_seeds)
# flip_plot(length=100, random_seeds=random_seeds)
# four_peaks(length=100, random_seeds=random_seeds)

x_train, x_test, y_train, y_test = load_dataset(split_percentage=0.2)
neural_network(x_train, x_test, y_train, y_test, random_seeds=random_seeds)
