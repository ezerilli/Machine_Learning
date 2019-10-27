# Base classifier class

import numpy as np
import matplotlib.pyplot as plt
import time

from abc import ABC, abstractmethod

from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import validation_curve, learning_curve

IMAGE_DIR = 'images/'


class BaseClassifier(ABC):

    def __init__(self, name):
        """Initialize a base classifier.
            Args:
                name (string): name of the classifier.
            Returns:
                None.
            """
        self.model = None  # a base classifier has no model
        self.optimal_params = {}  # nor optimal or default parameters
        self.default_params = {}
        self.name = name  # but it has a name

    def evaluate(self, x_test, y_test):
        """Evaluate the model by reporting the classification report and the confusion matrix.
            Args:
                x_test (ndarray): test set data.
                y_test (ndarray): test set labels.

            Returns:
                None.
            """
        predictions = self.predict(x_test)  # predict test labels

        print("\n\nEvaluate on the Test Set with parameters \n{}\n".format(self.optimal_params))
        print(classification_report(y_test, predictions))  # produce classification report
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, predictions))  # produce confusion matrix

    def fit(self, x_train, y_train):
        """Fit the model by on the training data.
            Args:
                x_train (ndarray): train set data.
                y_train (ndarray): train set labels.

            Returns:
                None.
            """
        # Fit the model and report training time
        start_time = time.time()
        self.model.fit(x_train, y_train)
        end_time = time.time()

        print('\n\nFitting Training Set: {:.4f} seconds'.format(end_time-start_time))

    def predict(self, x):
        """Predict on some test data.
            Args:
                x (ndarray): data to predict.

            Returns:
                predictions (ndarray): array of labels predictions.
            """
        # Predict and report inference time
        start_time = time.time()
        predictions = self.model.predict(x)
        end_time = time.time()

        print('\n\nPredicting on Testing Set: {:.4f} seconds'.format(end_time-start_time))

        return predictions

    def get_model_parameters(self):
        """Get model parameters.
            Args:
               None.

            Returns:
               params (dict): model parameters.
            """
        return self.model.get_params()

    def set_model_parameters(self, **params):
        """Set model parameters.
            Args:
               params (dict): model parameters.

            Returns:
               None.
            """
        self.model.set_params(**params)
        return True

    @staticmethod
    def _plot_helper_(x_axis, train_scores, val_scores, train_label, val_label):
        """Plot helper.
            Args:
               x_axis (ndarray): x axis array.
               train_scores (ndarray): array of training scores.
               val_scores (ndarray): array of validation scores.
               train_label (string): training plot label.
               val_label (string): validation plot label.

            Returns:
               None.
            """

        # Compute training and validation scores mean and standard deviation over cross-validation folds.
        train_scores_mean = np.mean(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        # Plot training and validation mean by filling in between mean + std and mean - std
        train_plot = plt.plot(x_axis, train_scores_mean, '-o', markersize=2, label=train_label)
        val_plot = plt.plot(x_axis, val_scores_mean, '-o', markersize=2, label=val_label)
        plt.fill_between(x_axis, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color=train_plot[0].get_color())
        plt.fill_between(x_axis, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std,
                         alpha=0.1, color=val_plot[0].get_color())

    def plot_learning_curve(self, x_train, y_train, **kwargs):
        """Plot learning curves with cross-validation.
            Args:
               x_train (ndarray): training data.
               y_train (ndarray): training labels.
               kwargs (dict): additional arguments to pass for learning curves plotting:
                    - train_sizes (ndarray): training set sizes to plot the learning curves over.
                    - cv (int): number of k-folds in cross-validation.
                    - y_lim (float): lower y axis limit.

            Returns:
               None.
            """
        print('\n\nLearning Analysis with k-Fold Cross Validation')
        # Clone the model and use clone for training in learning curves
        model = clone(self.model)

        # Set default parameters and produce corresponding learning curves using k-fold cross-validation
        model.set_params(**self.default_params)
        _, train_scores_default, val_scores_default = learning_curve(model, x_train, y_train,
                                                                     train_sizes=kwargs['train_sizes'],
                                                                     cv=kwargs['cv'], n_jobs=-1)

        # Set optimal parameters and produce corresponding learning curves using k-fold cross-validation
        model.set_params(**self.optimal_params)
        _, train_scores_optimal, val_scores_optimal = learning_curve(model, x_train, y_train,
                                                                     train_sizes=kwargs['train_sizes'],
                                                                     cv=kwargs['cv'], n_jobs=-1)

        # Create a new figure and plot learning curves both for the default and the optimal parameters
        plt.figure()
        self._plot_helper_(kwargs['train_sizes'], train_scores_default, val_scores_default,
                           train_label='Training with default params', val_label='Cross-Validation, default params')
        self._plot_helper_(kwargs['train_sizes'], train_scores_optimal, val_scores_optimal,
                           train_label='Training with optimal params', val_label='Cross-Validation, optimal params')

        # Add title, legend, axes labels and eventually set y axis limits
        plt.title('{} - Learning Curves using {}-Fold Cross Validation'.format(self.name, kwargs['cv']))
        plt.legend(loc='lower left')
        plt.xlabel('Training samples')
        plt.ylabel('Accuracy')
        if kwargs['y_lim']:
            plt.ylim(kwargs['y_lim'], 1.01)

        # Save figure and show
        plt.savefig(IMAGE_DIR + '{}_learning_curve'.format(self.name))
        plt.show()

    @abstractmethod
    def plot_model_complexity(self, x_train, y_train, **kwargs):
        """Plot model complexity curves with cross-validation.
            Args:
               x_train (ndarray): training data.
               y_train (ndarray): training labels.
               kwargs (dict): additional arguments to pass for model complexity curves plotting:
                    - param (string): parameter to plot the model complexity curves over.
                    - param_range (ndarray or list): array or list of values for parameters.
                    - cv (int): number of k-folds in cross-validation.
                    - train_label (string): label for training plots.
                    - val_label (string): label for validation plots.
                    - x_label (string): label for x axis.
                    - x_scale (string): scale for x axis.
                    - y_lim (float): lower y axis limit.

            Returns:
               optimal_param_value (int or float): parameter value that maximizes cross-validation.
               optimal_val_score (float): corresponding cross-validation score.
            """

        print('-> params: {}'.format(self.optimal_params))

        # Clone the model and use clone for training for validation curves
        model = clone(self.model)

        # Set optimal parameters and produce model complexity curves
        model.set_params(**self.optimal_params)
        train_scores, val_scores = validation_curve(model, x_train, y_train, kwargs['param'],
                                                    kwargs['param_range'], cv=kwargs['cv'], n_jobs=-1)

        # Plot model complexity curves
        self._plot_helper_(kwargs['param_range'], train_scores, val_scores,
                           train_label=kwargs['train_label'], val_label=kwargs['val_label'])

        # Add title, legend, axes labels, scale of x axis and eventually set y axis limits
        plt.title('{} - Validation curves using {}-Fold Cross Validation'.format(self.name, kwargs['cv']))
        plt.legend(loc='lower left')
        plt.xlabel(kwargs['x_label'])
        plt.ylabel('Accuracy')
        plt.xscale(kwargs['x_scale'])
        if kwargs['y_lim']:
            plt.ylim(kwargs['y_lim'], 1.01)

        # Find the parameter value that maximize the mean of the validation scores over cross-validation folds
        val_scores_mean = np.mean(val_scores, axis=1)
        max_index = np.argmax(val_scores_mean)

        return kwargs['param_range'][max_index], val_scores_mean[max_index]

    def experiment(self, x_train, x_test, y_train, y_test, **kwargs):
        """Run an experiment on the model.

            Run model complexity to find optimal parameters, plot learning curves,
            fit on training data and evaluate the model on the test set.

            Args:
               x_train (ndarray): training data.
               x_test (ndarray): test data.
               y_train (ndarray): training labels.
               y_test (ndarray): test labels.

            Returns:
               None.
            """

        print('\n--------------------------')
        self.plot_model_complexity(x_train, y_train, **kwargs)
        self.plot_learning_curve(x_train, y_train, **kwargs)
        self.fit(x_train, y_train)
        self.evaluate(x_test, y_test)
