# AdaBoost class

import matplotlib.pyplot as plt

from base_classifier import BaseClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


class AdaBoost(BaseClassifier):

    def __init__(self, n_estimators=50, learning_rate=1., max_depth=3, random_state=42):
        """Initialize AdaBoost with Decision Trees as weak classifiers.

           Args:
               n_estimators (int): number of weak classifiers.
               learning_rate (float): learning rate.
               max_depth (int): maximum depth of the tree.
               random_state (int): random seed.

           Returns:
               None.
           """

        # Initialize Classifier
        print('AdaBoosted Decision Tree classifier')
        super(AdaBoost, self).__init__(name='AdaBoost')

        # Define AdaBoost model, preceded by a data standard scaler (subtract mean and divide by std)
        self.model = Pipeline([('scaler', StandardScaler()),
                               ('adaboost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth,
                                                                                      random_state=random_state),
                                                               n_estimators=n_estimators, learning_rate=learning_rate,
                                                               random_state=random_state))])

        # Save default parameters
        self.default_params = {'adaboost__n_estimators': n_estimators,
                               'adaboost__learning_rate': learning_rate,
                               'adaboost__base_estimator__max_depth': max_depth}

    def plot_model_complexity(self, x_train, y_train, **kwargs):
        """Plot model complexity curves with cross-validation.

            Args:
               x_train (ndarray): training data.
               y_train (ndarray): training labels.
               kwargs (dict): additional arguments to pass for model complexity curves plotting:
                    - max_depth_range (ndarray or list): array or list of values for maximum depth.
                    - n_estimators_range (ndarray or list): array or list of values for number of estimators.
                    - learning_rate_range (ndarray or list): array or list of values for learning rate.
                    - cv (int): number of k-folds in cross-validation.
                    - y_lim (float): lower y axis limit.

            Returns:
               None.
            """

        # Initially our optimal parameters are simply the default parameters
        print('\n\nModel Complexity Analysis')
        self.optimal_params = self.default_params.copy()

        # Create a new figure for maximum depth validation curve and set proper arguments
        plt.figure()
        kwargs['param'] = 'adaboost__base_estimator__max_depth'
        kwargs['param_range'] = kwargs['max_depth_range']
        kwargs['x_label'] = 'Maximum Tree Depth'
        kwargs['x_scale'] = 'linear'
        kwargs['train_label'] = 'Training'
        kwargs['val_label'] = 'Cross-Validation'

        # Plot validation curve for maximum depth and get its optimal value and corresponding score
        best_max_depth, score = super(AdaBoost, self).plot_model_complexity(x_train, y_train, **kwargs)
        print('--> max_depth = {} --> score = {:.4f}'.format(best_max_depth, score))

        # Save the optimal maximum depth in our dictionary of optimal parameters and save figure
        self.optimal_params['adaboost__base_estimator__max_depth'] = best_max_depth
        plt.savefig(IMAGE_DIR + '{}_max_depth'.format(self.name))

        # Create a new figure for number of estimators validation curve and set proper arguments
        plt.figure()
        kwargs['param'] = 'adaboost__n_estimators'
        kwargs['param_range'] = kwargs['n_estimators_range']
        kwargs['x_label'] = 'Number of Estimators'

        # Plot validation curve for number of estimators and get its optimal value and corresponding score
        best_n_estimators, score = super(AdaBoost, self).plot_model_complexity(x_train, y_train, **kwargs)
        print('--> n_estimators = {} --> score = {:.4f}'.format(best_n_estimators, score))

        # Save the optimal number of estimators in our dictionary of optimal parameters and save figure
        self.optimal_params['adaboost__n_estimators'] = best_n_estimators
        plt.savefig(IMAGE_DIR + '{}_n_estimators'.format(self.name))

        # Create a new figure for learning rate validation curve and set proper arguments
        plt.figure()
        kwargs['param'] = 'adaboost__learning_rate'
        kwargs['param_range'] = kwargs['learning_rate_range']
        kwargs['x_label'] = 'Learning Rate'
        kwargs['x_scale'] = 'log'

        # Plot validation curve for learning rate and get its optimal value and corresponding score
        best_learning_rate, score = super(AdaBoost, self).plot_model_complexity(x_train, y_train, **kwargs)
        print('--> learning_rate = {} --> score = {:.4f}'.format(best_learning_rate, score))

        # Save the optimal learning rate in our dictionary of optimal parameters and save figure
        self.optimal_params['adaboost__learning_rate'] = best_learning_rate
        plt.savefig(IMAGE_DIR + '{}_learning_rate'.format(self.name))

        # Set optimal parameters as model parameters
        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)
