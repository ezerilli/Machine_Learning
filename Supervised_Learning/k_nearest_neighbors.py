# K-Nearest Neighbors class

import matplotlib.pyplot as plt

from base_classifier import BaseClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


class KNN(BaseClassifier):

    def __init__(self, k=1, weights='uniform', p=2):
        """Initialize a Decision Tree.

            Args:
                k (int): number of neighbors.
                weights (string): uniform weights or distance-based.
                p (int): degree of Minkowski distance (1=Manhattan, 2=Euclidean).

            Returns:
                None.
            """

        # Initialize Classifier
        print('kNN classifier')
        super(KNN, self).__init__(name='kNN')

        # Define K-Nearest Neighbors model, preceded by a data standard scaler (subtract mean and divide by std)
        self.model = Pipeline([('scaler', StandardScaler()),
                               ('knn', KNeighborsClassifier(n_neighbors=k,
                                                            weights=weights,
                                                            metric='minkowski',
                                                            p=p,
                                                            n_jobs=-1))])

        # Save default parameters
        self.default_params = {'knn__n_neighbors': k,
                               'knn__weights': weights,
                               'knn__metric': 'minkowski',
                               'knn__p': p}

    def plot_model_complexity(self, x_train, y_train, **kwargs):
        """Plot model complexity curves with cross-validation.

            Args:
               x_train (ndarray): training data.
               y_train (ndarray): training labels.
               kwargs (dict): additional arguments to pass for model complexity curves plotting:
                    - n_neighbors_range (ndarray or list): array or list of values for the number of neighbors k.
                    - p_range (ndarray or list): array or list of values for the Minkowski degree p.
                    - weight_functions (ndarray or list): array or list of values for the weight function.
                    - cv (int): number of k-folds in cross-validation.
                    - y_lim (float): lower y axis limit.

            Returns:
               None.
            """

        # Initially our optimal parameters are simply the default parameters
        print('\n\nModel Complexity Analysis')
        self.optimal_params = self.default_params.copy()

        # Initialize best values
        best_score, best_k, best_weight, best_p = 0., 1, '', 2

        # Create a new figure for number of neighbors and weight function validation curve and set proper arguments
        plt.figure()
        kwargs['param'] = 'knn__n_neighbors'
        kwargs['param_range'] = kwargs['n_neighbors_range']
        kwargs['x_label'] = 'Number of neighbors k'
        kwargs['x_scale'] = 'linear'

        # For all different weight functions
        for weight in kwargs['weight_functions']:

            # Set current weight function as if was an optimal parameter
            self.optimal_params['knn__weights'] = weight

            # Set training and validation label for current weight function
            kwargs['train_label'] = 'Training, {} weights'.format(weight)
            kwargs['val_label'] = 'Cross-Validation, {} weights'.format(weight)

            # Plot validation curve for neighbors number and weight function and get its optimal value and score
            k, score = super(KNN, self).plot_model_complexity(x_train, y_train, **kwargs)
            print('--> k = {}, weight = {} --> score = {:.4f}'.format(k, weight, score))

            # If this score is higher than the best score found so far, update best values
            if score > best_score:
                best_score, best_k, best_weight = score, k, weight

        # Save the optimal neighbors number and weight function in our dictionary of optimal parameters and save figure
        self.optimal_params['knn__n_neighbors'] = best_k
        self.optimal_params['knn__weights'] = best_weight
        plt.savefig(IMAGE_DIR + '{}_k'.format(self.name))

        # Create a new figure for Minkowski degree p validation curve and set proper arguments
        plt.figure()
        kwargs['param'] = 'knn__p'
        kwargs['param_range'] = kwargs['p_range']
        kwargs['x_label'] = 'Minkowski Degree p'
        kwargs['train_label'] = 'Training'
        kwargs['val_label'] = 'Cross-Validation'

        # Plot validation curve for Minkowski degree p and get its optimal value and score
        best_p, score = super(KNN, self).plot_model_complexity(x_train, y_train, **kwargs)
        print('--> p = {} --> score = {:.4f}'.format(best_p, score))

        # Save the optimal Minkowski degree p in our dictionary of optimal parameters and save figure
        self.optimal_params['knn__p'] = best_p
        plt.savefig(IMAGE_DIR + '{}_p'.format(self.name))

        # Set optimal parameters as model parameters
        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)
