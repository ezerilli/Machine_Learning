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
        self.model = None
        self.optimal_params = {}
        self.default_params = {}
        self.name = name

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)

        print("\n\nEvaluate on the Test Set with parameters \n{}\n".format(self.optimal_params))
        print(classification_report(y_test, predictions))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, predictions))

    def fit(self, x_train, y_train):
        start_time = time.time()
        self.model.fit(x_train, y_train)
        end_time = time.time()

        print('\n\nFitting Training Set: {:.4f} seconds'.format(end_time-start_time))

    def predict(self, x_test):
        start_time = time.time()
        predictions = self.model.predict(x_test)
        end_time = time.time()

        print('\n\nPredicting on Testing Set: {:.4f} seconds'.format(end_time-start_time))

        return predictions

    def get_model_parameters(self):
        return self.model.get_params()

    def set_model_parameters(self, **params):
        self.model.set_params(**params)
        return True

    @staticmethod
    def _plot_helper_(x_axis, train_scores, val_scores, train_label, val_label):

        train_scores_mean = np.mean(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        train_plot = plt.plot(x_axis, train_scores_mean, '-o', markersize=2, label=train_label)
        val_plot = plt.plot(x_axis, val_scores_mean, '-o', markersize=2, label=val_label)
        plt.fill_between(x_axis, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color=train_plot[0].get_color())
        plt.fill_between(x_axis, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std,
                         alpha=0.1, color=val_plot[0].get_color())

    def plot_learning_curve(self, x_train, y_train, **kwargs):

        print('\n\nLearning Analysis with k-Fold Cross Validation')
        model = clone(self.model)
        model.set_params(**self.default_params)
        _, train_scores_default, val_scores_default = learning_curve(model, x_train, y_train,
                                                                     train_sizes=kwargs['train_sizes'],
                                                                     cv=kwargs['cv'], n_jobs=-1)
        model.set_params(**self.optimal_params)
        _, train_scores_optimal, val_scores_optimal = learning_curve(model, x_train, y_train,
                                                                     train_sizes=kwargs['train_sizes'],
                                                                     cv=kwargs['cv'], n_jobs=-1)

        plt.figure()
        self._plot_helper_(kwargs['train_sizes'], train_scores_default, val_scores_default,
                           train_label='Training with default params', val_label='Cross-Validation, default params')
        self._plot_helper_(kwargs['train_sizes'], train_scores_optimal, val_scores_optimal,
                           train_label='Training with optimal params', val_label='Cross-Validation, optimal params')

        plt.title('{} - Learning Curves using {}-Fold Cross Validation'.format(self.name, kwargs['cv']))
        plt.legend(loc='lower left')
        plt.xlabel('Training samples')
        plt.ylabel('Accuracy')
        if kwargs['y_lim']:
            plt.ylim(kwargs['y_lim'], 1.01)
        plt.savefig(IMAGE_DIR + '{}_learning_curve'.format(self.name))
        plt.show()

    @abstractmethod
    def plot_model_complexity(self, x_train, y_train, **kwargs):

        print('-> params: {}'.format(self.optimal_params))

        model = clone(self.model)
        model.set_params(**self.optimal_params)
        train_scores, val_scores = validation_curve(model, x_train, y_train, kwargs['param'],
                                                    kwargs['param_range'], cv=kwargs['cv'], n_jobs=-1)

        self._plot_helper_(kwargs['param_range'], train_scores, val_scores,
                           train_label=kwargs['train_label'], val_label=kwargs['val_label'])

        plt.title('{} - Validation curves using {}-Fold Cross Validation'.format(self.name, kwargs['cv']))
        plt.legend(loc='lower left')
        plt.xlabel(kwargs['x_label'])
        plt.ylabel('Accuracy')
        plt.xscale(kwargs['x_scale'])
        if kwargs['y_lim']:
            plt.ylim(kwargs['y_lim'], 1.01)
        val_scores_mean = np.mean(val_scores, axis=1)
        max_index = np.argmax(val_scores_mean)

        return kwargs['param_range'][max_index], val_scores_mean[max_index]

    def experiment(self, x_train, x_test, y_train, y_test, **kwargs):

        print('\n--------------------------')
        self.plot_model_complexity(x_train, y_train, **kwargs)
        self.plot_learning_curve(x_train, y_train, **kwargs)
        self.fit(x_train, y_train)
        self.evaluate(x_test, y_test)
