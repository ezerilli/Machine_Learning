# Support Vector Machines class

import matplotlib.pyplot as plt

from base_classifier import BaseClassifier

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


class SVM(BaseClassifier):

    def __init__(self, c=1., kernel='rbf', degree=3, gamma=0.001, random_state=42):
        """Initialize a Decision Tree.

            Args:
                c (float): regularization term.
                kernel (string): kernel function.
                degree (int): degree for polynomial kernels.
                gamma (float): gamma value for rbf and tanh kernels.
                random_state (int): random seed.

            Returns:
                None.
            """

        # Initialize Classifier
        print('SVM classifier')
        super(SVM, self).__init__(name='SVM')

        # Define Support Vector Machines model, preceded by a data standard scaler (subtract mean and divide by std)
        self.model = Pipeline([('scaler', StandardScaler()),
                               ('svm', SVC(C=c,
                                           kernel=kernel,
                                           degree=degree,
                                           gamma=gamma,
                                           random_state=random_state))])

        # Save default parameters
        self.default_params = {'svm__C': c,
                               'svm__kernel': kernel,
                               'svm__degree': degree,
                               'svm__gamma': gamma}

    def plot_model_complexity(self, x_train, y_train, **kwargs):
        """Plot model complexity curves with cross-validation.

            Args:
               x_train (ndarray): training data.
               y_train (ndarray): training labels.
               kwargs (dict): additional arguments to pass for model complexity curves plotting:
                    - C_range (ndarray or list): array or list of values for the regularization term.
                    - kernels (ndarray or list): array or list of values for kernels.
                    - gamma_range (ndarray or list): array or list of values for the gamma value.
                    - poly_degrees (ndarray or list): array or list of values for polynomial degree p.
                    - cv (int): number of k-folds in cross-validation.
                    - y_lim (float): lower y axis limit.

            Returns:
               None.
            """
        # Initially our optimal parameters are simply the default parameters
        print('\n\nModel Complexity Analysis')
        self.optimal_params = self.default_params.copy()

        # Initialize best values
        best_score, best_c, best_kernel, best_d, best_gamma = 0., 1., '', 3, 0.001

        # Create a new figure for regularization term validation curve and set proper arguments
        plt.figure()
        kwargs['param'] = 'svm__C'
        kwargs['param_range'] = kwargs['C_range']
        kwargs['x_label'] = 'Penalty parameter C'
        kwargs['x_scale'] = 'linear'

        # For all different kernels
        for kernel in kwargs['kernels']:

            # Set current kernel as if was an optimal parameter
            self.optimal_params['svm__kernel'] = kernel

            # Set training and validation label for current kernel
            kwargs['train_label'] = 'Training, {} kernel'.format(kernel)
            kwargs['val_label'] = 'Cross-Validation, {} kernel'.format(kernel)

            # Plot validation curve for regualrization term and kernels and get its optimal value and score
            c, score = super(SVM, self).plot_model_complexity(x_train, y_train, **kwargs)
            print('--> c = {}, kernel = {} --> score = {:.4f}'.format(c, kernel, score))

            # If this score is higher than the best score found so far, update best values
            if score > best_score:
                best_score, best_c, best_kernel = score, c, kernel

        # Save the optimal regularization term and kernel in our dictionary of optimal parameters and save figure
        self.optimal_params['svm__C'] = best_c
        self.optimal_params['svm__kernel'] = best_kernel
        plt.savefig(IMAGE_DIR + '{}_c'.format(self.name))

        # Create a new figure for gamma value validation curve and set proper arguments
        plt.figure()
        kwargs['param'] = 'svm__gamma'
        kwargs['param_range'] = kwargs['gamma_range']
        kwargs['x_label'] = r'Kernel coefficient $\gamma$'
        kwargs['x_scale'] = 'log'

        # For all different kernels
        for kernel in kwargs['kernels']:

            # Set current kernel as if was an optimal parameter
            self.optimal_params['svm__kernel'] = kernel

            # Set list of polynomial degrees depending on kernel
            poly_degrees = [] + [3] * (kernel == 'rbf') + kwargs['poly_degrees'] * (kernel == 'poly')

            # For all different degrees polynomial
            for d in poly_degrees:

                # Set current polynomial degree as if was an optimal parameter
                self.optimal_params['svm__degree'] = d

                # Set training and validation label for current kernel and degree polynomial
                label = ', {} kernel'.format(kernel) + ', d = {}'.format(d) * (kernel == 'poly')
                kwargs['train_label'] = 'Training' + label
                kwargs['val_label'] = 'Cross-Validation' + label

                # Plot validation curve for gamma value and kernels and get its optimal value and score
                gamma, score = super(SVM, self).plot_model_complexity(x_train, y_train, **kwargs)
                print('--> gamma = {:.4f}, kernel = {}, d = {} --> score = {:.4f}'.format(gamma, kernel, d, score))

                # If this score is higher than the best score found so far, update best values
                if score > best_score:
                    best_score, best_gamma, best_d, best_kernel = score, gamma, d, kernel

        # Save the optimal gamma value, poly degree and kernel in our dictionary of optimal parameters and save figure
        self.optimal_params['svm__gamma'] = best_gamma
        self.optimal_params['svm__degree'] = best_d
        self.optimal_params['svm__kernel'] = best_kernel
        plt.savefig(IMAGE_DIR + '{}_gamma'.format(self.name))

        # Set optimal parameters as model parameters
        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)
