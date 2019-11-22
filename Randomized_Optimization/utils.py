# Plot and time helper functions

import numpy as np
import matplotlib.pyplot as plt
import time

start_time = 0.
times = []


def plot_helper(x_axis, y_axis, label):
    """Plot helper.

        Args:
          x_axis (ndarray): x axis to plot over.
          y_axis (ndarray): y axis to plot.
          label (ndarray): label.


        Returns:
          None.
        """

    y_mean, y_std = np.mean(y_axis, axis=0), np.std(y_axis, axis=0)
    plot = plt.plot(x_axis, y_mean, label=label)
    plt.fill_between(x_axis, y_mean - y_std, y_mean + y_std, alpha=0.1, color=plot[0].get_color())


def set_plot_title_labels(title, x_label, y_label):
    """Set plot title and labels.

        Args:
          title (string): plot title.
          x_label (string): x label.
          y_label (string): y label.

        Returns:
          None.
        """
    plt.title(title)
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def time_callback(iteration, attempt=None, done=None, state=None, fitness=None, curve=None, user_data=None):
    """Time callback for saving time elapsed at each iteration of the algorithm.

        Args:
          iteration (int): current iteration.
          attempt (int): current attempt.
          done (bool): id we are done iterating.
          state (list): current best state.
          fitness (float): current best fitness.
          curve (ndarray): current fitness curve.
          user_data (any): current iteration.

        Returns:
          continue (bool): True, to continue iterating.
        """
    # Define global variables
    global start_time, times

    # At first iteration, save start time and reset list of times, else save time elapsed since start
    if iteration == 0:
        start_time = time.time()
        times = []
    else:
        times.append(time.time() - start_time)

    # Return always True to continue iterating
    return True
