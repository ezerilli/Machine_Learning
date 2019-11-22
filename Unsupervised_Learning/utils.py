# Plot utils

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_DIR = 'images/'  # output images directory


def plot_clusters(ax, component1, component2, df, name):
    """Plot clusters for clustering.

        Args:
          ax (axes object): axes to plot at.
          component1 (string): first component to plot (PCA or TSNE).
          component2 (string): second component to plot (PCA or TSNE).
          df (dataframe): dataframe containing input data.
          name (string): clustering technique.

        Returns:
          None.
        """

    # Plot input data onto first two components at given axes
    y_colors = sns.color_palette('hls', len(np.unique(df['y'])))
    c_colors = sns.color_palette('hls', len(np.unique(df['c'])))
    sns.scatterplot(x=component1, y=component2, hue='y', palette=y_colors, data=df, legend='full', alpha=0.3, ax=ax[0])
    sns.scatterplot(x=component1, y=component2, hue='c', palette=c_colors, data=df, legend='full', alpha=0.3, ax=ax[1])

    # Set titles
    ax[0].set_title('True Clusters represented with {}'.format(component1[:-1].upper()))
    ax[1].set_title('{} Clusters represented with {}'.format(name.upper(), component1[:-1].upper()))

    # Set axes limits
    xlim = 1.1 * np.max(np.abs(df[component1]))
    ylim = 1.1 * np.max(np.abs(df[component2]))
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_ylim(-ylim, ylim)
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_ylim(-ylim, ylim)


def plot_components(component1, component2, df, name):
    """Plot components for dimensionality reduction.

        Args:
          component1 (string): first component to plot.
          component2 (string): second component to plot.
          df (dataframe): dataframe containing input data.
          name (string): dimensionality reduction technique.

        Returns:
          None.
        """

    # Create figure and plot input data onto first two components
    plt.figure()
    colors = sns.color_palette('hls', len(np.unique(df['y'])))
    sns.scatterplot(x=component1, y=component2, hue='y', palette=colors, data=df, legend='full', alpha=0.3)

    # Annotate standard deviation arrows for the two components
    plt.annotate('', xy=(np.std(df[component1]), 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    plt.annotate('', xy=(0, np.std(df[component2])), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))

    # Set title and axes limits
    plt.title('{} Transformation with first 2 components and true labels'.format(name.upper()))
    xlim = 1.1 * np.max(np.abs(df[component1]))
    ylim = 1.1 * np.max(np.abs(df[component2]))
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)


def plot_multiple_random_runs(x_axis, y_axis, label):
    """Plot helper.

        Args:
          x_axis (ndarray): x axis to plot over.
          y_axis (ndarray): y axis to plot.
          label (string): label.


        Returns:
          None.
        """
    # Plot mean with area between mean + std and mean - std filled of the same color
    y_mean, y_std = np.mean(y_axis, axis=0), np.std(y_axis, axis=0)
    plot = plt.plot(x_axis, y_mean, '-o', markersize=1, label=label)
    plt.fill_between(x_axis, y_mean - y_std, y_mean + y_std, alpha=0.1, color=plot[0].get_color())


def plot_ic_bars(ic, ic_label, cv_types, k_range, ax):
    """Plot Information Criterion histogram.

        Args:
          ic (ndarray): Information Criterion to plot.
          ic_label (string): Information Criterion label.
          cv_types (list): covariance type.
          k_range (ndarray): range of number of clusters.
          ax (axis object): axis to plot at.

        Returns:
          None.
        """

    bar_width = 0.2  # width of a bar

    # For each covariance type, plot a histogram of different color
    for i, cv_type in enumerate(cv_types):
        x = k_range + bar_width / 2 * (i - 2)  # compute x
        ax.bar(x, ic[i * len(k_range):(i + 1) * len(k_range)], width=bar_width, label=cv_type)

    # Set x values, y limit aand legend
    ax.set_xticks(k_range)
    ax.set_ylim([ic.min() * 1.01 - .01 * ic.max(), ic.max()])
    ax.legend(loc='best')

    # Set title and labels
    set_axis_title_labels(ax=ax, title='EM - Choosing k with the {} method'.format(ic_label),
                          x_label='Number of components k', y_label='{} score'.format(ic_label))


def save_figure(title):
    """Save Figure.

        Args:
          title (string): plot title.

        Returns:
          None.
        """
    plt.savefig(IMAGE_DIR + title)
    plt.close()


def save_figure_tight(title):
    """Save Figure with tight layout.

        Args:
          title (string): plot title.

        Returns:
          None.
        """

    plt.tight_layout()
    save_figure(title)


def set_axis_title_labels(ax, title, x_label, y_label):
    """Set axis title and labels.

        Args:
          ax (axis object): axis.
          title (string): plot title.
          x_label (string): x label.
          y_label (string): y label.

        Returns:
          None.
        """
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
