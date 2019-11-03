# Clustering script

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

from abc import ABC, abstractmethod

from scipy.stats import kurtosis

from sklearn.decomposition import FastICA, KernelPCA, PCA
from sklearn.random_projection import SparseRandomProjection


class DimensionalityReduction(ABC):

    def __init__(self, name, n_components, random_seed):
        self.name = name
        self.model = None
        self.n_components = n_components
        self.random_seed = random_seed

    def experiment(self, x_train, x_test, y_train, dataset, perform_model_complexity):

        if perform_model_complexity:
            self.plot_model_complexity(x_train, dataset)

        x_train_reduced, mse = self.train(x_train)
        print('\nTrain on training set')
        print('Reconstruction error = {:.3f}'.format(mse))

        x_test_reduced = self.reduce(x_test)

        self.visualize_components(x_train_reduced, y_train, dataset)

        return x_train_reduced, x_test_reduced

    def train(self, x):
        x_reduced = self.model.fit_transform(x)
        mse = self.reconstruct(x, x_reduced)
        return x_reduced, mse

    def reconstruct(self, x, x_reduced):
        x_reconstructed = self.model.inverse_transform(x_reduced)
        mse = np.mean((x - x_reconstructed) ** 2)
        return mse

    def reduce(self, x):
        return self.model.transform(x)

    @abstractmethod
    def plot_model_complexity(self, x, dataset):
        pass

    def set_n_components(self, n_components):
        self.n_components = n_components
        self.model.set_params(**{'n_components': n_components})

    def visualize_components(self, x_reduced, y, dataset):

        component1 = '{}1'.format(self.name)
        component2 = '{}2'.format(self.name)

        df = pd.DataFrame(x_reduced[:, :2], columns=[component1, component2])
        df['y'] = y

        utils.plot_components(component1, component2, df, y, self.name)
        utils.save_figure('{}_{}_components'.format(dataset, self.name))


class IndependentComponents(DimensionalityReduction):

    def __init__(self, n_components=2, random_seed=42):
        super(IndependentComponents, self).__init__(name='ica', n_components=n_components, random_seed=random_seed)
        self.model = FastICA(n_components=n_components, tol=0.01, max_iter=1000, random_state=random_seed)


    def plot_model_complexity(self, x, dataset):

        print('\nPlot Model Complexity')

        average_kurtosis = []
        if x.shape[1] < 100:
            k_range = np.arange(1, x.shape[1] + 1)
        else:
            k_range = np.arange(0, int(0.8 * x.shape[1]) + 1, 20)
            k_range[0] = 1

        for k in k_range:
            ica = FastICA(n_components=k, tol=0.01, max_iter=1000, random_state=self.random_seed)
            ica.fit(x)

            components_kurtosis = kurtosis(ica.components_, axis=1, fisher=False)
            average_kurtosis.append(np.mean(components_kurtosis))

            print('k = {} --> average kurtosis = {:.3f}'.format(k, average_kurtosis[-1]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        ax1.plot(k_range, average_kurtosis,  '-o', markersize=1, label='Kurtosis')
        utils.set_axis_title_labels(ax1, title='ICA - Choosing k with the Average Kurtosis method',
                                    x_label='Number of components k', y_label='Average Kurtosis')

        self.model.fit(x)
        if x.shape[1] < 100:
            x_ticks = np.arange(1, self.n_components + 1)
        else:
            x_ticks = np.arange(0, self.n_components + 1, 50)
            x_ticks[0] = 1

        components_kurtosis = kurtosis(self.model.components_, axis=1, fisher=False)

        ax2.bar(np.arange(1, self.n_components + 1), components_kurtosis, color='cyan')
        utils.set_axis_title_labels(ax2, title='ICA - Components Kurtosis Distribution',
                                    x_label='Independent component', y_label='Kurtosis')
        ax2.set_xticks(x_ticks)

        # Save figure
        utils.save_figure_tight('{}_ica_model_complexity'.format(dataset))


class KernelPrincipalComponents(DimensionalityReduction):

    def __init__(self, n_components=2, kernel='rbf', random_seed=42):
        super(KernelPrincipalComponents, self).__init__(name='kpca', n_components=n_components, random_seed=random_seed)
        self.model = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=True,
                               random_state=random_seed, n_jobs=-1)
        self.kernel = kernel

    def plot_model_complexity(self, x, dataset):
        print('\nPlot Model Complexity')
        k_range = np.arange(1, x.shape[1] + 1)
        kernels = ['rbf', 'poly', 'sigmoid', 'cosine']

        fig, ax = plt.subplots(2, 4, figsize=(15, 10))
        ax = ax.ravel()

        for i, kernel in enumerate(kernels):

            kpca = KernelPCA(n_components=x.shape[1], kernel=kernel, random_state=self.random_seed, n_jobs=-1)
            kpca.fit(x)

            explained_variance_ratio = kpca.lambdas_ / np.sum(kpca.lambdas_)
            explained_variance = np.sum(explained_variance_ratio[:self.n_components])
            print('Kernel = {} - Explained variance [n components = {}]= {:.3f}'.format(kernel,
                                                                                        self.n_components,
                                                                                        explained_variance))

            ax[2*i].bar(k_range, np.cumsum(explained_variance_ratio), color='red', label=kernel)
            utils.set_axis_title_labels(ax[2*i], title='KPCA - Choosing k with the Variance method',
                                        x_label='Number of components k', y_label='Cumulative Variance (%)')
            ax[2*i].legend(loc='best')

            ax[2*i+1].bar(k_range, explained_variance_ratio, color='cyan', label=kernel)
            utils.set_axis_title_labels(ax[2*i+1], title='KPCA - Eigenvalues distributions',
                                        x_label='Number of components k', y_label='Variance (%)')
            ax[2*i+1].legend(loc='best')

        # Save figure
        utils.save_figure_tight('{}_kpca_model_complexity'.format(dataset))


class PrincipalComponents(DimensionalityReduction):

    def __init__(self, n_components=2, random_seed=42):
        super(PrincipalComponents, self).__init__(name='pca', n_components=n_components, random_seed=random_seed)
        self.model = PCA(n_components=n_components, svd_solver='randomized', random_state=random_seed)

    def plot_model_complexity(self, x, dataset):

        print('\nPlot Model Complexity')
        k_range = np.arange(1, x.shape[1]+1)

        pca = PCA(svd_solver='randomized', random_state=self.random_seed)
        pca.fit(x)

        explained_variance = np.sum(pca.explained_variance_ratio_[:self.n_components])
        print('Explained variance [n components = {}]= {:.3f}'.format(self.n_components, explained_variance))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        ax1.bar(k_range, np.cumsum(pca.explained_variance_ratio_), color='red')
        utils.set_axis_title_labels(ax1, title='PCA - Choosing k with the Variance method',
                                    x_label='Number of components k', y_label='Cumulative Variance (%)')

        ax2.bar(k_range, pca.explained_variance_ratio_, color='cyan')
        utils.set_axis_title_labels(ax2, title='PCA - Eigenvalues distributions',
                                    x_label='Number of components k', y_label='Variance (%)')

        # Save figure
        utils.save_figure_tight('{}_pca_model_complexity'.format(dataset))


class RandomProjections(DimensionalityReduction):

    def __init__(self, n_components=2, random_runs=10, random_seed=42):
        super(RandomProjections, self).__init__(name='rp', n_components=n_components, random_seed=random_seed)
        self.model = SparseRandomProjection(n_components=n_components, random_state=random_seed)
        self.random_runs = random_runs

    def experiment(self, x_train, x_test, y_train, dataset, perform_model_complexity):

        if perform_model_complexity:
            self.plot_model_complexity(x_train, dataset)

        train_errors = []
        x_train_reduced = np.zeros((x_train.shape[0], self.n_components))
        x_test_reduced = np.zeros((x_test.shape[0], self.n_components))

        print('\nTrain on training set')

        for seed in range(self.random_seed, self.random_seed + self.random_runs):

            self.model = SparseRandomProjection(n_components=self.n_components, random_state=seed)

            x_reduced, mse = self.train(x_train)
            x_train_reduced += x_reduced
            train_errors.append(mse)

            x_reduced = self.reduce(x_test)
            x_test_reduced += x_reduced

        print('Reconstruction error = {:.3f} with std = {:.3f}'.format(np.mean(train_errors), np.std(train_errors)))

        x_train_reduced /= self.random_runs
        x_test_reduced /= self.random_runs

        self.visualize_components(x_train_reduced, y_train, dataset)

        return x_train_reduced, x_test_reduced

    def plot_model_complexity(self, x, dataset):

        print('\nPlot Model Complexity')

        mse_random_runs = []

        if x.shape[1] < 100:
            k_range = np.arange(1, x.shape[1] + 1)
        else:
            k_range = np.arange(0, int(0.8 * x.shape[1]) + 1, 20)
            k_range[0] = 1

        for seed in range(self.random_seed, self.random_seed + self.random_runs):

            mse = []

            print('Random run {}'.format(seed + 1 - self.random_seed))

            for k in k_range:

                rp = SparseRandomProjection(n_components=k, random_state=seed)
                x_reduced = rp.fit_transform(x)
                P_inv = np.linalg.pinv(rp.components_.toarray())  # k x m -> m x k
                x_reconstructed = (P_inv @ x_reduced.T).T  # m x k x k x n = m x n -> n x m
                mse.append(np.mean((x - x_reconstructed) ** 2))

            mse_random_runs.append(mse)

        np.set_printoptions(precision=2)
        print('k = [2, ..., {}] --> \nReconstruction errors = {}'.format(k_range[-1], np.mean(mse_random_runs, axis=0)))

        plt.figure()
        utils.plot_multiple_random_runs(k_range, mse_random_runs, 'MSE')
        utils.set_plot_title_labels(title='RP - Choosing k with the Reconstruction Error',
                                    x_label='Number of components k', y_label='MSE')

        # Save figure
        utils.save_figure('{}_rp_model_complexity'.format(dataset))

    def reconstruct(self, x, x_reduced):
        P_inv = np.linalg.pinv(self.model.components_.toarray())  # k x m -> m x k
        x_reconstructed = (P_inv @ x_reduced.T).T  # m x k x k x n = m x n -> n x m
        mse = np.mean((x - x_reconstructed) ** 2)
        return mse
