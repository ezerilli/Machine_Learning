# Decision Tree class

import matplotlib.pyplot as plt

from base_classifier import BaseClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


class DecisionTree(BaseClassifier):

    def __init__(self, max_depth=1, min_samples_leaf=1, random_state=42):
        """Initialize a Decision Tree.

            Args:
                max_depth (int): maximum depth of the tree.
                min_samples_leaf (int): minimum number of samples per leaf to allow.
                random_state (int): random seed.

            Returns:
                None.
            """

        # Initialize Classifier
        print('Decision Tree classifier')
        super(DecisionTree, self).__init__(name='DT')

        # Define Decision Tree model, preceded by a data standard scaler (subtract mean and divide by std)
        self.model = Pipeline([('scaler', StandardScaler()),
                               ('dt', DecisionTreeClassifier(max_depth=max_depth,
                                                             min_samples_leaf=min_samples_leaf,
                                                             random_state=random_state))])

        # Save default parameters
        self.default_params = {'dt__max_depth': max_depth,
                               'dt__min_samples_leaf': min_samples_leaf}

    def plot_model_complexity(self, x_train, y_train, **kwargs):
        """Plot model complexity curves with cross-validation.

            Args:
               x_train (ndarray): training data.
               y_train (ndarray): training labels.
               kwargs (dict): additional arguments to pass for model complexity curves plotting:
                    - max_depth_range (ndarray or list): array or list of values for maximum depth.
                    - min_samples_leaf_range (ndarray or list): array or list of values for minimum samples per leaf.
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
        kwargs['param'] = 'dt__max_depth'
        kwargs['param_range'] = kwargs['max_depth_range']
        kwargs['x_label'] = 'Maximum Tree Depth'
        kwargs['x_scale'] = 'linear'
        kwargs['train_label'] = 'Training'
        kwargs['val_label'] = 'Cross-Validation'

        # Plot validation curve for maximum depth and get its optimal value and corresponding score
        best_max_depth, score = super(DecisionTree, self).plot_model_complexity(x_train, y_train, **kwargs)
        print('--> max_depth = {} --> score = {:.4f}'.format(best_max_depth, score))

        # Save the optimal maximum depth in our dictionary of optimal parameters and save figure
        self.optimal_params['dt__max_depth'] = best_max_depth
        plt.savefig(IMAGE_DIR + '{}_max_depth'.format(self.name))

        # Create a new figure for minimum samples per leaf validation curve and set proper arguments
        plt.figure()
        kwargs['param'] = 'dt__min_samples_leaf'
        kwargs['param_range'] = kwargs['min_samples_leaf_range']
        kwargs['x_label'] = 'Min Samples per Leaf'

        # Plot validation curve for minimum samples per leaf and get its optimal value and corresponding score
        best_min_samples, score = super(DecisionTree, self).plot_model_complexity(x_train, y_train, **kwargs)
        print('--> min_samples_leaf = {} --> score = {:.4f}'.format(best_min_samples, score))

        # Save the optimal minimum samples per leaf in our dictionary of optimal parameters and save figure
        self.optimal_params['dt__min_samples_leaf'] = best_min_samples
        plt.savefig(IMAGE_DIR + '{}_min_samples_leaf'.format(self.name))

        # Set optimal parameters as model parameters
        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)
