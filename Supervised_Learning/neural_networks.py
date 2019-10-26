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

        print('Neural Network classifier')
        super(NeuralNetwork, self).__init__(name='ANN')

        self.model = Pipeline([('scaler', StandardScaler()),
                               ('nn', MLPClassifier(hidden_layer_sizes=(layer1_nodes, layer2_nodes), activation='relu',
                                                    solver='sgd', alpha=alpha, batch_size=200, learning_rate='constant',
                                                    learning_rate_init=learning_rate, max_iter=max_iter, tol=1e-8,
                                                    early_stopping=False, validation_fraction=0.1, momentum=0.5,
                                                    n_iter_no_change=max_iter, random_state=42))])

        self.default_params = {'nn__alpha': alpha,
                               'nn__learning_rate_init': learning_rate}

    def plot_model_complexity(self, x_train, y_train, **kwargs):

        print('\n\nModel Complexity Analysis')

        self.optimal_params = self.default_params.copy()

        plt.figure()

        kwargs['param'] = 'nn__learning_rate_init'
        kwargs['param_range'] = kwargs['learning_rate_range']
        kwargs['x_label'] = 'Learning Rate'
        kwargs['x_scale'] = 'log'
        kwargs['train_label'] = 'Training'
        kwargs['val_label'] = 'Cross-Validation'

        best_learning_rate, score = super(NeuralNetwork, self).plot_model_complexity(x_train, y_train, **kwargs)

        print('--> best_learning_rate = {} --> score = {:.4f}'.format(best_learning_rate, score))
        self.optimal_params['nn__learning_rate_init'] = best_learning_rate
        plt.savefig(IMAGE_DIR + '{}_learning_rate'.format(self.name))

        plt.figure()

        kwargs['param'] = 'nn__alpha'
        kwargs['param_range'] = kwargs['alpha_range']
        kwargs['x_label'] = r'L2 regularization term $\alpha$'

        best_alpha, score = super(NeuralNetwork, self).plot_model_complexity(x_train, y_train, **kwargs)

        print('--> best_alpha = {} --> score = {:.4f}'.format(best_alpha, score))
        self.optimal_params['nn__alpha'] = best_alpha
        plt.savefig(IMAGE_DIR + '{}_alpha'.format(self.name))

        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)

        plt.show()
        
    def plot_learning_curve(self, x_train, y_train, **kwargs):

        super(NeuralNetwork, self).plot_learning_curve(x_train, y_train, **kwargs)
        max_iter = self.model.get_params()['nn__max_iter']
        epochs = np.arange(1, max_iter + 1, 1)
        train_scores, val_scores = [], []

        model = clone(self.model)
        model.set_params(**{'nn__max_iter': 1, 'nn__warm_start': True})

        for train_index, val_index in StratifiedKFold(n_splits=kwargs['cv']).split(x_train, y_train):

            x_train_split, x_val_split = x_train[train_index], x_train[val_index]
            y_train_split, y_val_split = y_train[train_index], y_train[val_index]

            train_scores_split, val_scores_split = [], []

            for epoch in range(1, max_iter + 1):
                model.fit(x_train_split, y_train_split)
                if (epoch-1) % 1 == 0:
                    train_scores_split.append(model.score(x_train_split, y_train_split))
                    val_scores_split.append(model.score(x_val_split, y_val_split))

            train_scores.append(train_scores_split)
            val_scores.append(val_scores_split)

        train_scores = np.array(train_scores)
        val_scores = np.array(val_scores)

        self._plot_helper_(epochs, train_scores.T, val_scores.T, train_label='Training', val_label='Cross-Validation')

        plt.title('Neural Network - Learning Curves using k={}-Fold Cross Validation'.format(kwargs['cv']))
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        if kwargs['y_lim']:
            plt.ylim(kwargs['y_lim'], 1.01)
        plt.savefig(IMAGE_DIR + '{}_epochs'.format(self.name))
        plt.show()
