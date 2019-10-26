import matplotlib.pyplot as plt

from base_classifier import BaseClassifier

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


class SVM(BaseClassifier):

    def __init__(self, c=1., kernel='rbf', degree=3, gamma=0.001, random_state=42):

        print('SVM classifier')
        super(SVM, self).__init__(name='SVM')

        self.model = Pipeline([('scaler', StandardScaler()),
                               ('svm', SVC(C=c,
                                           kernel=kernel,
                                           degree=degree,
                                           gamma=gamma,
                                           random_state=random_state))])

        self.default_params = {'svm__C': c,
                               'svm__kernel': kernel,
                               'svm__degree': degree,
                               'svm__gamma': gamma}

    def plot_model_complexity(self, x_train, y_train, **kwargs):

        print('\n\nModel Complexity Analysis')
        best_score, best_c, best_kernel, best_d, best_gamma = 0., 1., '', 3, 0.001
        self.optimal_params = self.default_params.copy()

        kwargs['param'] = 'svm__C'
        kwargs['param_range'] = kwargs['C_range']
        kwargs['x_label'] = 'Penalty parameter C'
        kwargs['x_scale'] = 'linear'

        plt.figure()
        for kernel in kwargs['kernels']:

            self.optimal_params['svm__kernel'] = kernel
            kwargs['train_label'] = 'Training, {} kernel'.format(kernel)
            kwargs['val_label'] = 'Cross-Validation, {} kernel'.format(kernel)

            c, score = super(SVM, self).plot_model_complexity(x_train, y_train, **kwargs)

            if score > best_score:
                best_score, best_c, best_kernel = score, c, kernel

            print('--> c = {}, kernel = {} --> score = {:.4f}'.format(c, kernel, score))

        self.optimal_params['svm__C'] = best_c
        self.optimal_params['svm__kernel'] = best_kernel
        plt.savefig(IMAGE_DIR + '{}_c'.format(self.name))

        plt.figure()

        kwargs['param'] = 'svm__gamma'
        kwargs['param_range'] = kwargs['gamma_range']
        kwargs['x_label'] = r'Kernel coefficient $\gamma$'
        kwargs['x_scale'] = 'log'

        for kernel in kwargs['kernels']:

            self.optimal_params['svm__kernel'] = kernel
            poly_degrees = [] + [3] * (kernel == 'rbf') + kwargs['poly_degrees'] * (kernel == 'poly')

            for d in poly_degrees:

                self.optimal_params['svm__degree'] = d
                label = ', {} kernel'.format(kernel) + ', d = {}'.format(d) * (kernel == 'poly')
                kwargs['train_label'] = 'Training' + label
                kwargs['val_label'] = 'Cross-Validation' + label

                gamma, score = super(SVM, self).plot_model_complexity(x_train, y_train, **kwargs)

                if score > best_score:
                    best_score, best_gamma, best_d, best_kernel = score, gamma, d, kernel

                print('--> gamma = {:.4f}, kernel = {}, d = {} --> score = {:.4f}'.format(gamma, kernel, d, score))

        self.optimal_params['svm__gamma'] = best_gamma
        self.optimal_params['svm__degree'] = best_d
        self.optimal_params['svm__kernel'] = best_kernel
        plt.savefig(IMAGE_DIR + '{}_gamma'.format(self.name))

        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)
        plt.show()
