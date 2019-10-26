import matplotlib.pyplot as plt

from base_classifier import BaseClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


class DecisionTree(BaseClassifier):

    def __init__(self, max_depth=1, min_samples_leaf=1, random_state=42):

        print('Decision Tree classifier')
        super(DecisionTree, self).__init__(name='DT')

        self.model = Pipeline([('scaler', StandardScaler()),
                               ('dt', DecisionTreeClassifier(max_depth=max_depth,
                                                             min_samples_leaf=min_samples_leaf,
                                                             random_state=random_state))])

        self.default_params = {'dt__max_depth': max_depth,
                               'dt__min_samples_leaf': min_samples_leaf}

    def plot_model_complexity(self, x_train, y_train, **kwargs):

        print('\n\nModel Complexity Analysis')
        self.optimal_params = self.default_params.copy()

        plt.figure()
        kwargs['param'] = 'dt__max_depth'
        kwargs['param_range'] = kwargs['max_depth_range']
        kwargs['x_label'] = 'Maximum Tree Depth'
        kwargs['x_scale'] = 'linear'
        kwargs['train_label'] = 'Training'
        kwargs['val_label'] = 'Cross-Validation'
        best_max_depth, score = super(DecisionTree, self).plot_model_complexity(x_train, y_train, **kwargs)

        print('--> max_depth = {} --> score = {:.4f}'.format(best_max_depth, score))
        self.optimal_params['dt__max_depth'] = best_max_depth
        plt.savefig(IMAGE_DIR + '{}_max_depth'.format(self.name))

        plt.figure()
        kwargs['param'] = 'dt__min_samples_leaf'
        kwargs['param_range'] = kwargs['min_samples_leaf_range']
        kwargs['x_label'] = 'Min Samples per Leaf'
        best_min_samples, score = super(DecisionTree, self).plot_model_complexity(x_train, y_train, **kwargs)

        print('--> min_samples_leaf = {} --> score = {:.4f}'.format(best_min_samples, score))
        plt.savefig(IMAGE_DIR + '{}_min_samples_leaf'.format(self.name))

        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)
        plt.show()
