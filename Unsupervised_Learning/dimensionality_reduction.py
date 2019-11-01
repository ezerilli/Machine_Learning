# Clustering script

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

from abc import ABC, abstractmethod

from sklearn.decomposition import FastICA, PCA
from sklearn.random_projection import SparseRandomProjection


class DimensionalityReduction(ABC):

    def __init__(self, name, n_components, random_seed):
        self.name = name
        self.model = None
        self.n_components = n_components
        self.random_seed = random_seed

    def experiment(self, x_train, x_test, y_train, dataset, perfrom_model_complexity):

        if perfrom_model_complexity:
            self.plot_model_complexity(x_train, dataset)

        x_reduced, mse = self.train(x_train)
        print('Train reconstruction error = {:.3f}'.format(mse))

        self.visualize_components(x_reduced, y_train, dataset)
        _, mse = self.test(x_test)
        print('Test reconstruction error = {:.3f}'.format(mse))

    def train(self, x):
        x_reduced = self.model.fit_transform(x)
        mse = self.reconstruct(x, x_reduced)
        return x_reduced, mse

    def reconstruct(self, x, x_reduced):
        x_reconstructed = self.model.inverse_transform(x_reduced)
        mse = np.mean((x - x_reconstructed) ** 2)
        return mse

    def test(self, x):
        x_reduced = self.model.transform(x)
        mse = self.reconstruct(x, x_reduced)
        return x_reduced, mse

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
        self.model = FastICA(n_components=n_components, whiten=False, max_iter=1000, random_state=random_seed)

    def plot_model_complexity(self, x, dataset):

        print('\nPlot Model Complexity')

        ica = FastICA(whiten=False, max_iter=1000, random_state=self.random_seed)
        ica.fit(x)


class PrincipalComponents(DimensionalityReduction):

    def __init__(self, n_components=2, random_seed=42):
        super(PrincipalComponents, self).__init__(name='pca', n_components=n_components, random_seed=random_seed)
        self.model = PCA(n_components=n_components, svd_solver='randomized', random_state=random_seed)

    def plot_model_complexity(self, x, dataset):

        print('\nPlot Model Complexity')
        k_range = np.arange(1, x.shape[1]+1)

        pca = PCA(svd_solver='randomized', random_state=self.random_seed)
        pca.fit(x)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        ax1.bar(k_range, np.cumsum(pca.explained_variance_ratio_), color='red')
        utils.set_axis_title_labels(ax1, title='PCA - Choosing k with the Variance method',
                                    x_label='Number of components k', y_label='Cumulative Variance (%)')

        ax2.bar(k_range, pca.explained_variance_ratio_, color='cyan')
        utils.set_axis_title_labels(ax2, title='PCA - Eigenvalues distributions',
                                    x_label='Number of components k', y_label='Variance (%)')

        # Save figure
        utils.save_figure('{}_pca_model_complexity'.format(dataset))


class RandomProjections(DimensionalityReduction):

    def __init__(self, n_components=2, random_runs=10, random_seed=42):
        super(RandomProjections, self).__init__(name='rp', n_components=n_components, random_seed=random_seed)
        self.model = SparseRandomProjection(n_components=n_components, random_state=random_seed)
        self.random_runs = random_runs

    def plot_model_complexity(self, x, dataset):

        print('\nPlot Model Complexity')

        mse_random_runs = []
        k_range = np.arange(1, x.shape[1] + 1)

        for seed in range(self.random_seed, self.random_seed + self.random_runs):

            mse = []

            for k in k_range:

                rp = SparseRandomProjection(n_components=k, random_state=seed)
                x_reduced = rp.fit_transform(x)
                x_reconstructed = (rp.components_.T @ x_reduced.T).T
                mse.append(np.mean((x - x_reconstructed) ** 2))

            mse_random_runs.append(mse)

        np.set_printoptions(precision=2)
        print('k = \n{} \n--> reconstruction errors = \n{}'.format(k_range, np.mean(mse_random_runs, axis=0)))

        plt.figure()
        utils.plot_multiple_random_runs(k_range, mse_random_runs, 'MSE')
        utils.set_plot_title_labels(title='RP - Choosing k with the Reconstruction Error',
                                    x_label='Number of components k', y_label='MSE')

        # Save figure
        utils.save_figure('{}_{}_model_complexity'.format(dataset, self.name))

    def reconstruct(self, x, x_reduced):
        x_reconstructed = (self.model.components_.T @ x_reduced.T).T
        mse = np.mean((x - x_reconstructed) ** 2)
        return mse

    def experiment(self, x_train, x_test, y_train, dataset, perfrom_model_complexity):

        if perfrom_model_complexity:
            self.plot_model_complexity(x_train, dataset)

        train_errors, test_errors = [], []
        x_reduced_mean = np.zeros((x_train.shape[0], self.n_components))

        for seed in range(self.random_seed, self.random_seed + self.random_runs):

            self.model = SparseRandomProjection(n_components=self.n_components, random_state=seed)

            x_reduced, mse = self.train(x_train)
            x_reduced_mean += x_reduced
            train_errors.append(mse)

            _, test_mse = self.test(x_test)
            test_errors.append(mse)

        print('Train reconstruction error = {:.3f}'.format(np.mean(train_errors)))
        print('Test reconstruction error = {:.3f}'.format(np.mean(test_errors)))

        x_reduced_mean /= self.random_runs
        self.visualize_components(x_reduced_mean, y_train, dataset)
