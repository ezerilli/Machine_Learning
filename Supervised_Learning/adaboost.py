import matplotlib.pyplot as plt

from base_classifier import BaseClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


class AdaBoost(BaseClassifier):

    def __init__(self, n_estimators=50, learning_rate=1., max_depth=3, random_state=42):

        print('AdaBoosted Decision Tree classifier')
        super(AdaBoost, self).__init__(name='AdaBoost')

        self.model = Pipeline([('scaler', StandardScaler()),
                               ('adaboost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth,
                                                                                      random_state=random_state),
                                                               n_estimators=n_estimators, learning_rate=learning_rate,
                                                               random_state=random_state))])

        self.default_params = {'adaboost__n_estimators': n_estimators,
                               'adaboost__learning_rate': learning_rate,
                               'adaboost__base_estimator__max_depth': max_depth}

    def plot_model_complexity(self, x_train, y_train, **kwargs):

        print('\n\nModel Complexity Analysis')
        self.optimal_params = self.default_params.copy()

        plt.figure()

        kwargs['param'] = 'adaboost__base_estimator__max_depth'
        kwargs['param_range'] = kwargs['max_depth_range']
        kwargs['x_label'] = 'Maximum Tree Depth'
        kwargs['x_scale'] = 'linear'
        kwargs['train_label'] = 'Training'
        kwargs['val_label'] = 'Cross-Validation'

        best_max_depth, score = super(AdaBoost, self).plot_model_complexity(x_train, y_train, **kwargs)

        print('--> max_depth = {} --> score = {:.4f}'.format(best_max_depth, score))
        self.optimal_params['adaboost__base_estimator__max_depth'] = best_max_depth
        plt.savefig(IMAGE_DIR + '{}_max_depth'.format(self.name))

        plt.figure()

        kwargs['param'] = 'adaboost__n_estimators'
        kwargs['param_range'] = kwargs['n_estimators_range']
        kwargs['x_label'] = 'Number of Estimators'

        best_n_estimators, score = super(AdaBoost, self).plot_model_complexity(x_train, y_train, **kwargs)

        print('--> n_estimators = {} --> score = {:.4f}'.format(best_n_estimators, score))
        self.optimal_params['adaboost__n_estimators'] = best_n_estimators
        plt.savefig(IMAGE_DIR + '{}_n_estimators'.format(self.name))

        plt.figure()

        kwargs['param'] = 'adaboost__learning_rate'
        kwargs['param_range'] = kwargs['learning_rate_range']
        kwargs['x_label'] = 'Learning Rate'
        kwargs['x_scale'] = 'log'

        best_learning_rate, score = super(AdaBoost, self).plot_model_complexity(x_train, y_train, **kwargs)

        print('--> learning_rate = {} --> score = {:.4f}'.format(best_learning_rate, score))
        self.optimal_params['adaboost__learning_rate'] = best_learning_rate
        plt.savefig(IMAGE_DIR + '{}_learning_rate'.format(self.name))

        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)
        plt.show()
