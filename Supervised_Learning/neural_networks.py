# Neural Networks class

import numpy as np
import matplotlib.pyplot as plt

from base_classifier import BaseClassifier

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


class NeuralNetwork(BaseClassifier):

    def __init__(self, alpha=0.0001, layer1_nodes=50, layer2_nodes=30, learning_rate=0.001, max_iter=200):
        """Initialize a Neural Network.

            Args:
                alpha (float): regularization term.
                layer1_nodes (int): number of neurons in first layer.
                layer2_nodes (int): number of neurons in second layer.
                learning_rate (float): learning rate.
                max_iter (int): maximum number of iterations.

            Returns:
                None.
            """

        # Initialize Classifier
        print('Neural Network classifier')
        super(NeuralNetwork, self).__init__(name='ANN')

        # Define Neural Networks model, preceded by a data standard scaler (subtract mean and divide by std)
        self.model = Pipeline([('scaler', StandardScaler()),
                               ('nn', MLPClassifier(hidden_layer_sizes=(layer1_nodes, layer2_nodes), activation='relu',
                                                    solver='sgd', alpha=alpha, batch_size=200, learning_rate='constant',
                                                    learning_rate_init=learning_rate, max_iter=max_iter, tol=1e-8,
                                                    early_stopping=False, validation_fraction=0.1, momentum=0.5,
                                                    n_iter_no_change=max_iter, random_state=42))])

        # Save default parameters
        self.default_params = {'nn__alpha': alpha,
                               'nn__learning_rate_init': learning_rate}

    def plot_model_complexity(self, x_train, y_train, **kwargs):
        """Plot model complexity curves with cross-validation.

            Args:
               x_train (ndarray): training data.
               y_train (ndarray): training labels.
               kwargs (dict): additional arguments to pass for model complexity curves plotting:
                    - learning_rate_range (ndarray or list): array or list of values for the learning rate.
                    - alpha_range (ndarray or list): array or list of values for the regularization term.
                    - cv (int): number of k-folds in cross-validation.
                    - y_lim (float): lower y axis limit.

            Returns:
               None.
            """

        # Initially our optimal parameters are simply the default parameters
        print('\n\nModel Complexity Analysis')
        self.optimal_params = self.default_params.copy()

        # Create a new figure for the learning rate validation curve and set proper arguments
        plt.figure()
        kwargs['param'] = 'nn__learning_rate_init'
        kwargs['param_range'] = kwargs['learning_rate_range']
        kwargs['x_label'] = 'Learning Rate'
        kwargs['x_scale'] = 'log'
        kwargs['train_label'] = 'Training'
        kwargs['val_label'] = 'Cross-Validation'

        # Plot validation curve for the learning rate and get optimal value and score
        best_learning_rate, score = super(NeuralNetwork, self).plot_model_complexity(x_train, y_train, **kwargs)
        print('--> best_learning_rate = {} --> score = {:.4f}'.format(best_learning_rate, score))

        # Save the optimal learning rate in our dictionary of optimal parameters and save figure
        self.optimal_params['nn__learning_rate_init'] = best_learning_rate
        plt.savefig(IMAGE_DIR + '{}_learning_rate'.format(self.name))

        # Create a new figure for the regularization term validation curve and set proper arguments
        plt.figure()
        kwargs['param'] = 'nn__alpha'
        kwargs['param_range'] = kwargs['alpha_range']
        kwargs['x_label'] = r'L2 regularization term $\alpha$'

        # Plot validation curve for the the regularization term and gets optimal value and score
        best_alpha, score = super(NeuralNetwork, self).plot_model_complexity(x_train, y_train, **kwargs)
        print('--> best_alpha = {} --> score = {:.4f}'.format(best_alpha, score))

        # Save the optimal regularization term in our dictionary of optimal parameters and save figure
        self.optimal_params['nn__alpha'] = best_alpha
        plt.savefig(IMAGE_DIR + '{}_alpha'.format(self.name))

        # Set optimal parameters as model parameters
        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)

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

        # Plot learning curves
        super(NeuralNetwork, self).plot_learning_curve(x_train, y_train, **kwargs)

        # Produce and plot training curve vs. epochs
        max_iter = self.model.get_params()['nn__max_iter']  # maximum number of iterations
        epochs = np.arange(1, max_iter + 1, 1)  # epochs array
        train_scores, val_scores = [], []  # list of training and validation scores

        # Clone the model and use clone for training in learning curves
        model = clone(self.model)

        # Set max iter to 1 and warm start to True, such that the model fits one epoch per time and keeps weights
        model.set_params(**{'nn__max_iter': 1, 'nn__warm_start': True})

        # Perform k-fold Stratified cross-validation
        for train_fold, val_fold in StratifiedKFold(n_splits=kwargs['cv']).split(x_train, y_train):

            # Training and validation set for current fold
            x_train_fold, x_val_fold = x_train[train_fold], x_train[val_fold]
            y_train_fold, y_val_split = y_train[train_fold], y_train[val_fold]

            # List of training and validation scores for current fold
            train_scores_fold, val_scores_fold = [], []

            # Loop through epochs
            for _ in epochs:
                # Fit model on the training set of the current fold
                model.fit(x_train_fold, y_train_fold)

                # Append training and validation score to corresponding lists
                train_scores_fold.append(model.score(x_train_fold, y_train_fold))
                val_scores_fold.append(model.score(x_val_fold, y_val_split))

            # Append training and validation scores of current fold to corresponding lists
            train_scores.append(train_scores_fold)
            val_scores.append(val_scores_fold)

        # Convert to numpy arrays to plot
        train_scores = np.array(train_scores)
        val_scores = np.array(val_scores)

        # Plot training and validation scores vs. epochs
        self._plot_helper_(epochs, train_scores.T, val_scores.T, train_label='Training', val_label='Cross-Validation')

        # Add title, legend, axes labels and eventually set y axis limits
        plt.title('Neural Network - Learning Curves using k={}-Fold Cross Validation'.format(kwargs['cv']))
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        if kwargs['y_lim']:
            plt.ylim(kwargs['y_lim'], 1.01)

        # Save figure
        plt.savefig(IMAGE_DIR + '{}_epochs'.format(self.name))
