import matplotlib.pyplot as plt

from base_classifier import BaseClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


class KNN(BaseClassifier):

    def __init__(self, k=1, weights='uniform', p=2):

        print('kNN classifier')
        super(KNN, self).__init__(name='kNN')

        self.model = Pipeline([('scaler', StandardScaler()),
                               ('knn', KNeighborsClassifier(n_neighbors=k,
                                                            weights=weights,
                                                            metric='minkowski',
                                                            p=p,
                                                            n_jobs=-1))])

        self.default_params = {'knn__n_neighbors': k,
                               'knn__weights': weights,
                               'knn__metric': 'minkowski',
                               'knn__p': p}

    def plot_model_complexity(self, x_train, y_train, **kwargs):

        print('\n\nModel Complexity Analysis')
        best_score, best_k, best_weight, best_p = 0., 1, '', 2
        self.optimal_params = self.default_params.copy()

        kwargs['param'] = 'knn__n_neighbors'
        kwargs['param_range'] = kwargs['n_neighbors_range']
        kwargs['x_label'] = 'Number of neighbors k'
        kwargs['x_scale'] = 'linear'

        plt.figure()
        for weight in kwargs['weight_functions']:

            self.optimal_params['knn__weights'] = weight

            kwargs['train_label'] = 'Training, {} weights'.format(weight)
            kwargs['val_label'] = 'Cross-Validation, {} weights'.format(weight)

            k, score = super(KNN, self).plot_model_complexity(x_train, y_train, **kwargs)

            if score > best_score:
                best_score, best_k, best_weight = score, k, weight

            print('--> k = {}, weight = {} --> score = {:.4f}'.format(k, weight, score))

        self.optimal_params['knn__n_neighbors'] = best_k
        self.optimal_params['knn__weights'] = best_weight
        plt.savefig(IMAGE_DIR + '{}_k'.format(self.name))

        plt.figure()

        kwargs['param'] = 'knn__p'
        kwargs['param_range'] = kwargs['p_range']
        kwargs['x_label'] = 'Minkowski Degree p'
        kwargs['train_label'] = 'Training'
        kwargs['val_label'] = 'Cross-Validation'

        best_p, score = super(KNN, self).plot_model_complexity(x_train, y_train, **kwargs)

        print('--> p = {} --> score = {:.4f}'.format(best_p, score))
        self.optimal_params['knn__p'] = best_p
        plt.savefig(IMAGE_DIR + '{}_p'.format(self.name))

        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)

        plt.show()
