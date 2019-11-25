# Plot utils

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_DIR = 'images/'  # output images directory


def plot_heatmap_policy(policy, V, rows, columns):

    policy_labels = np.empty([rows, columns], dtype='<U10')
    for row in range(rows):
        for col in range(columns):
            state = row * columns + col
            policy_labels[row, col] += '<--' * (policy[state] == 0)
            policy_labels[row, col] += 'v' * (policy[state] == 1)
            policy_labels[row, col] += '-->' * (policy[state] == 2)
            policy_labels[row, col] += '^' * (policy[state] == 3)

    sns.heatmap(V.reshape(rows, columns), annot=policy_labels, fmt='', linewidths=.5)


def plot_heatmap_value_function(V, rows, columns):
    sns.heatmap(V.reshape(rows, columns), annot=True, fmt='.3f', linewidths=.5)


def plot_value_convergence(convergence, iteration):
    """Plot helper.

        Args:
          x_axis (ndarray): x axis to plot over.
          y_axis (ndarray): y axis to plot.

        Returns:
          None.
        """
    plt.plot(np.arange(1, iteration + 1), convergence)


def plot_value_function(V, iteration, show_label=False):
    """Plot helper.

        Args:
          x_axis (ndarray): x axis to plot over.
          y_axis (ndarray): y axis to plot.
          label (string): label.

        Returns:
          None.
        """
    if show_label:
        plt.plot(np.arange(1, len(V) + 1), V, label='iteration = {}'.format(iteration))
    else:
        plt.plot(np.arange(1, len(V) + 1), V)


def plot_optimal_policy(policy):
    plt.bar(np.arange(0, len(policy)), policy, color='blue')


def save_figure(title):
    """Save Figure.

        Args:
          title (string): plot title.

        Returns:
          None.
        """
    plt.savefig(IMAGE_DIR + title)
    plt.close()


def set_plot_title_labels(title, x_label='', y_label='', legend=False):
    """Set plot title and labels.

        Args:
          title (string): plot title.
          x_label (string): x label.
          y_label (string): y label.

        Returns:
          None.
        """
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if legend:
        plt.legend(loc='best')
