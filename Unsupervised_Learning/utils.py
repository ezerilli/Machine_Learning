# Plot utils

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_DIR = 'images/'


def plot_clusters(ax, x_label, y_label, df, y, name):
    clusters_colors = sns.color_palette('hls', len(np.unique(df['c'])))
    y_colors = sns.color_palette('hls', len(np.unique(y)))

    sns.scatterplot(x=x_label, y=y_label, hue='y', palette=y_colors, data=df, legend='full', alpha=0.3, ax=ax[0])
    sns.scatterplot(x=x_label, y=y_label, hue='c', palette=clusters_colors, data=df, legend='full', alpha=0.3, ax=ax[1])

    ax[0].set_title('True Clusters represented with {}'.format(x_label[:-1].upper()))
    ax[1].set_title('{} Clusters represented with {}'.format(name.upper(), x_label[:-1].upper()))


def plot_components(component1, component2, df, y, name):
    plt.figure()
    colors = sns.color_palette('hls', len(np.unique(y)))
    sns.scatterplot(x=component1, y=component2, hue='y', palette=colors, data=df, legend='full', alpha=0.3)
    # plt.quiver(0, 0, pca.singular_values_[0], pca.singular_values_[1], zorder=11, width=0.01, scale=6, color='red')

    plt.title('{} Transformation with first 2 components and true labels'.format(name.upper()))


def plot_multiple_random_runs(x_axis, y_axis, label):
    """Plot helper.

        Args:
          x_axis (ndarray): x axis to plot over.
          y_axis (ndarray): x axis to plot over.
          label (ndarray): x axis to plot over.


        Returns:
          None.
        """

    y_mean, y_std = np.mean(y_axis, axis=0), np.std(y_axis, axis=0)
    plot = plt.plot(x_axis, y_mean, '-o', markersize=2, label=label)
    plt.fill_between(x_axis, y_mean - y_std, y_mean + y_std, alpha=0.1, color=plot[0].get_color())


def plot_ic_bars(ic, ic_label, cv_types, k_range, ax):
    bar_width = 0.2
    for i, cv_type in enumerate(cv_types):
        x = k_range + bar_width / 2 * (i - 2)
        ax.bar(x, ic[i * len(k_range):(i + 1) * len(k_range)], width=bar_width, label=cv_type)

    ax.set_xticks(k_range)
    ax.set_ylim([ic.min() * 1.01 - .01 * ic.max(), ic.max()])
    ax.legend(loc='best')

    set_axis_title_labels(ax=ax, title='EM - Choosing k with the {} method'.format(ic_label),
                          x_label='Number of components k', y_label='{} score'.format(ic_label))


def save_figure(title):
    plt.tight_layout()
    plt.savefig(IMAGE_DIR + title)


def set_axis_title_labels(ax, title, x_label, y_label):
    """Set plot title and axes labels.

        Args:
          title (string): x axis to plot over.
          x_label (string): x axis to plot over.
          y_label (string): x axis to plot over.

        Returns:
          None.
        """
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def set_plot_title_labels(title, x_label, y_label):
    """Set plot title and axes labels.

        Args:
          title (string): x axis to plot over.
          x_label (string): x axis to plot over.
          y_label (string): x axis to plot over.

        Returns:
          None.
        """
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
