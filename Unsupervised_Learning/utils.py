# Plot utils

import matplotlib.pyplot as plt


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
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
