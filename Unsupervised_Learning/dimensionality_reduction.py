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
        """Initialize Dimensionality Reduction object.

           Args:
               name (string): name of the dimensionality reduction method.
               n_components (int): number of components.
               random_seed (int): random seed.

           Returns:
               None.
           """
        self.name = name
        self.model = None
        self.n_components = n_components
        self.random_seed = random_seed

    def experiment(self, x_train, x_test, y_train, dataset, perform_model_complexity):
        """Perform experiments.

            Args:
               x_train (ndarray): training data.
               x_test (ndarray): test data.
               y_train (ndarray): training labels.
               dataset (string): dataset, WDBC or MNIST.
               perform_model_complexity (bool): True if we want to perform model complexity.

            Returns:
              x_train_reduced (ndarray): reduced training data.
              x_test_reduced (ndarray): reduced test data.
            """

        print('\nTrain on training set')

        if perform_model_complexity:
            self.plot_model_complexity(x_train, dataset)

        x_train_reduced, mse = self.train(x_train)  # fit and reduce training data
        print('Reconstruction error = {:.3f}'.format(mse))
        self.visualize_components(x_train_reduced, y_train, dataset)  # visualize components

        return x_train_reduced,  self.reduce(x_test)  # reduce test data and return reduced training and test data

    @abstractmethod
    def plot_model_complexity(self, x, dataset):
        """Perform and plot model complexity.

            Args:
               x (ndarray): data.
               dataset (string): dataset, WDBC or MNIST.

            Returns:
              None.
            """
        pass

    def reconstruct(self, x, x_reduced):
        """Reconstruct original data and compute MSE.

            Args:
               x (ndarray): data.
               x_reduced (ndarray): reduced data.

            Returns:
               mse (float): Mean Squared Error of reconstruction.
            """
        x_reconstructed = self.model.inverse_transform(x_reduced)  # reconstruct
        mse = np.mean((x - x_reconstructed) ** 2)  # compute MSE
        return mse

    def reduce(self, x):
        """Reduce data.

            Args:
               x (ndarray): data.

            Returns:
               x_reduced (ndarray): reduced data.
            """
        return self.model.transform(x)

    def set_n_components(self, n_components):
        """Set number of components.

            Args:
               n_components (int): number of components.

            Returns:
              None.
            """
        self.n_components = n_components
        self.model.set_params(**{'n_components': n_components})

    def train(self, x):
        """Fit the model on training data and compute MSE on reconstruction.

            Args:
               x (ndarray): data.

            Returns:
               x_reduced (ndarray): reduced data.
               mse (float): Mean Squared Error of reconstruction.
            """
        x_reduced = self.model.fit_transform(x)  # fit model
        mse = self.reconstruct(x, x_reduced)  # reconstruct and compute MSE
        return x_reduced, mse

    def visualize_components(self, x_reduced, y, dataset):
        """Visualize components.

            Args:
               x_reduced (ndarray): reduced data.
               y (ndarray): true labels.
               dataset (string): dataset, WDBC or MNIST.

            Returns:
              None.
            """

        component1 = '{}1'.format(self.name)  # first component
        component2 = '{}2'.format(self.name)  # second component

        # Create dataframe for visualization
        df = pd.DataFrame(x_reduced[:, :2], columns=[component1, component2])
        df['y'] = y

        # Plot components and save figure
        utils.plot_components(component1, component2, df, self.name)
        utils.save_figure('{}_{}_components'.format(dataset, self.name))


class IndependentComponents(DimensionalityReduction):

    def __init__(self, n_components=2, random_seed=42):
        """Initialize ICA object.

           Args:
              n_components (int): number of components.
              random_seed (int): random seed.

           Returns:
              None.
           """

        super(IndependentComponents, self).__init__(name='ica', n_components=n_components, random_seed=random_seed)
        self.model = FastICA(n_components=n_components, tol=0.01, max_iter=1000, random_state=random_seed)

    def plot_model_complexity(self, x, dataset):
        """Perform and plot model complexity.

            Args:
               x (ndarray): data.
               dataset (string): dataset, WDBC or MNIST.

            Returns:
              None.
            """

        print('\nPlot Model Complexity')

        average_kurtosis = []  # list of average kurtosis

        # Define range of number of components k to plot over
        if x.shape[1] < 100:
            k_range = np.arange(1, x.shape[1] + 1)
        else:
            k_range = np.arange(0, int(0.8 * x.shape[1]) + 1, 20)
            k_range[0] = 1

        # For each k in the range
        for k in k_range:

            # Define a new ICA model and fit on training data
            ica = FastICA(n_components=k, tol=0.01, max_iter=1000, random_state=self.random_seed)
            ica.fit(x)

            # Compute kurtosis of components and report the average
            components_kurtosis = kurtosis(ica.components_, axis=1, fisher=False)
            average_kurtosis.append(np.mean(components_kurtosis))
            print('k = {} --> average kurtosis = {:.3f}'.format(k, average_kurtosis[-1]))

        # Create subplots and plot average kurtosis on the first suplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        ax1.plot(k_range, average_kurtosis,  '-o', markersize=1, label='Kurtosis')

        # Set title and labels
        utils.set_axis_title_labels(ax1, title='ICA - Choosing k with the Average Kurtosis method',
                                    x_label='Number of components k', y_label='Average Kurtosis')

        # Fit our ICA model on training data
        self.model.fit(x)

        # Define x values to show on plot
        if x.shape[1] < 100:
            x_ticks = np.arange(1, self.n_components + 1)
        else:
            x_ticks = np.arange(0, self.n_components + 1, 50)
            x_ticks[0] = 1

        # Compute kurtosis of components
        components_kurtosis = kurtosis(self.model.components_, axis=1, fisher=False)

        # Plot kurtosis distribution
        ax2.bar(np.arange(1, self.n_components + 1), components_kurtosis, color='cyan')

        # Set title, labels and x values to show on plot
        ax2.set_xticks(x_ticks)
        utils.set_axis_title_labels(ax2, title='ICA - Components Kurtosis Distribution',
                                    x_label='Independent component', y_label='Kurtosis')

        # Save figure
        utils.save_figure_tight('{}_ica_model_complexity'.format(dataset))


class KernelPrincipalComponents(DimensionalityReduction):

    def __init__(self, n_components=2, kernel='rbf', random_seed=42):
        """Initialize KPCA object.

           Args:
              n_components (int): number of components.
              kernel (string): kernel to use.
              random_seed (int): random seed.

           Returns:
              None.
           """
        super(KernelPrincipalComponents, self).__init__(name='kpca', n_components=n_components, random_seed=random_seed)
        self.model = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=True,
                               random_state=random_seed, n_jobs=-1)
        self.kernel = kernel

    def plot_model_complexity(self, x, dataset):
        """Perform and plot model complexity.

            Args:
               x (ndarray): data.
               dataset (string): dataset, WDBC or MNIST.

            Returns:
              None.
            """

        print('\nPlot Model Complexity')

        k_range = np.arange(1, x.shape[1] + 1)  # range of number of components k to plot over
        kernels = ['rbf', 'poly', 'sigmoid', 'cosine']  # kernels to plot with

        # Create subplots
        fig, ax = plt.subplots(2, 4, figsize=(15, 10))
        ax = ax.ravel()

        # For each kernel
        for i, kernel in enumerate(kernels):

            # Create KPCA object and fit it on training data
            kpca = KernelPCA(n_components=x.shape[1], kernel=kernel, random_state=self.random_seed, n_jobs=-1)
            kpca.fit(x)

            # Compute explained variance ratio from eigenvalues and the total explained variance
            # given by the chosen number of components
            explained_variance_ratio = kpca.lambdas_ / np.sum(kpca.lambdas_)
            explained_variance = np.sum(explained_variance_ratio[:self.n_components])
            print('Kernel = {} - Explained variance [n components = {}]= {:.3f}'.format(kernel,
                                                                                        self.n_components,
                                                                                        explained_variance))

            # Plot histogram of the cumulative explained variance ratio
            ax[2*i].bar(k_range, np.cumsum(explained_variance_ratio), color='red', label=kernel)

            # Set title, labels and legend
            ax[2*i].legend(loc='best')
            utils.set_axis_title_labels(ax[2*i], title='KPCA - Choosing k with the Variance method',
                                        x_label='Number of components k', y_label='Cumulative Variance (%)')

            # Plot histogram of the explained variance ratio, i.e. the eignevalues distribution
            ax[2*i+1].bar(k_range, explained_variance_ratio, color='cyan', label=kernel)

            # Set title, labels and legend
            ax[2*i+1].legend(loc='best')
            utils.set_axis_title_labels(ax[2*i+1], title='KPCA - Eigenvalues distributions',
                                        x_label='Number of components k', y_label='Variance (%)')

        # Save figure
        utils.save_figure_tight('{}_kpca_model_complexity'.format(dataset))


class PrincipalComponents(DimensionalityReduction):

    def __init__(self, n_components=2, random_seed=42):
        """Initialize PCA object.

           Args:
              n_components (int): number of components.
              random_seed (int): random seed.

           Returns:
              None.
           """
        super(PrincipalComponents, self).__init__(name='pca', n_components=n_components, random_seed=random_seed)
        self.model = PCA(n_components=n_components, svd_solver='randomized', random_state=random_seed)

    def plot_model_complexity(self, x, dataset):
        """Perform and plot model complexity.

            Args:
               x (ndarray): data.
               dataset (string): dataset, WDBC or MNIST.

            Returns:
              None.
            """

        print('\nPlot Model Complexity')

        k_range = np.arange(1, x.shape[1]+1)  # range of number of components k to plot over

        # Create PCA object and fit it on training data
        pca = PCA(svd_solver='randomized', random_state=self.random_seed)
        pca.fit(x)

        # Compute the total explained variance given by the chosen number of components
        explained_variance = np.sum(pca.explained_variance_ratio_[:self.n_components])
        print('Explained variance [n components = {}]= {:.3f}'.format(self.n_components, explained_variance))

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

        # Plot histogram of the cumulative explained variance ratio, with title and labels
        ax1.bar(k_range, np.cumsum(pca.explained_variance_ratio_), color='red')
        utils.set_axis_title_labels(ax1, title='PCA - Choosing k with the Variance method',
                                    x_label='Number of components k', y_label='Cumulative Variance (%)')

        # Plot histogram of the explained variance ratio, i.e. the eignevalues distribution, with titles and labels
        ax2.bar(k_range, pca.explained_variance_ratio_, color='cyan')
        utils.set_axis_title_labels(ax2, title='PCA - Eigenvalues distributions',
                                    x_label='Number of components k', y_label='Variance (%)')

        # Save figure
        utils.save_figure_tight('{}_pca_model_complexity'.format(dataset))


class RandomProjections(DimensionalityReduction):

    def __init__(self, n_components=2, random_runs=10, random_seed=42):
        """Initialize RP object.

          Args:
             n_components (int): number of components.
             random_runs (int): number of random restarts.
             random_seed (int): random seed.

          Returns:
             None.
          """
        super(RandomProjections, self).__init__(name='rp', n_components=n_components, random_seed=random_seed)
        self.model = SparseRandomProjection(n_components=n_components, random_state=random_seed)
        self.random_runs = random_runs

    def experiment(self, x_train, x_test, y_train, dataset, perform_model_complexity):
        """Perform experiments.

            Args:
               x_train (ndarray): training data.
               x_test (ndarray): test data.
               y_train (ndarray): training labels.
               dataset (string): dataset, WDBC or MNIST.
               perform_model_complexity (bool): True if we want to perform model complexity.

            Returns:
              x_train_reduced (ndarray): reduced training data.
              x_test_reduced (ndarray): reduced test data.
            """

        print('\nTrain on training set')

        if perform_model_complexity:
            self.plot_model_complexity(x_train, dataset)

        # Initialize training errors, reduced training and test data
        train_errors = []
        x_train_reduced = np.zeros((x_train.shape[0], self.n_components))
        x_test_reduced = np.zeros((x_test.shape[0], self.n_components))

        # Perform the defined number of random restarts
        for seed in range(self.random_seed, self.random_seed + self.random_runs):

            # Define new sparse RP model
            self.model = SparseRandomProjection(n_components=self.n_components, random_state=seed)

            # Train on training data
            x_reduced, mse = self.train(x_train)

            # Account for reduced training data and MSE for current run
            x_train_reduced += x_reduced
            train_errors.append(mse)

            # Account for reduced test data
            x_reduced = self.reduce(x_test)
            x_test_reduced += x_reduced

        print('Reconstruction error = {:.3f} with std = {:.3f}'.format(np.mean(train_errors), np.std(train_errors)))

        # Normalize the mean reduced training data and test data
        x_train_reduced /= self.random_runs
        x_test_reduced /= self.random_runs

        # Visualize components
        self.visualize_components(x_train_reduced, y_train, dataset)

        return x_train_reduced, x_test_reduced

    def plot_model_complexity(self, x, dataset):
        """Perform and plot model complexity.

            Args:
               x (ndarray): data.
               dataset (string): dataset, WDBC or MNIST.

            Returns:
              None.
            """

        print('\nPlot Model Complexity')

        mse_random_runs = []  # list of MSE over random runs

        # Define range of number of components k to plot over
        if x.shape[1] < 100:
            k_range = np.arange(1, x.shape[1] + 1)
        else:
            k_range = np.arange(0, int(0.8 * x.shape[1]) + 1, 20)
            k_range[0] = 1

        # Perform the defined number of random restarts
        for seed in range(self.random_seed, self.random_seed + self.random_runs):

            print('Random run {}'.format(seed + 1 - self.random_seed))
            mse = []  # initialize list of MSE over number of components

            # For each k in the range
            for k in k_range:

                # Define new sparse RP model and reduce training data
                rp = SparseRandomProjection(n_components=k, random_state=seed)
                x_reduced = rp.fit_transform(x)

                # Compute Pseudo inverse, reconstruct data and compute MSE
                P_inv = np.linalg.pinv(rp.components_.toarray())  # dimensions check : k x m -> m x k
                x_reconstructed = (P_inv @ x_reduced.T).T  # dimensions check: m x k x k x n = m x n -> n x m
                mse.append(np.mean((x - x_reconstructed) ** 2))

            mse_random_runs.append(mse)  # append list of MSE over number of components per current random run

        np.set_printoptions(precision=2)
        print('k = [2, ..., {}] --> \nReconstruction errors = {}'.format(k_range[-1], np.mean(mse_random_runs, axis=0)))

        # Create figure, plot multiple runs and set tile and labels
        plt.figure()
        utils.plot_multiple_random_runs(k_range, mse_random_runs, 'MSE')
        utils.set_plot_title_labels(title='RP - Choosing k with the Reconstruction Error',
                                    x_label='Number of components k', y_label='MSE')

        # Save figure
        utils.save_figure('{}_rp_model_complexity'.format(dataset))

    def reconstruct(self, x, x_reduced):
        """Reconstruct original data and compute MSE.

            Args:
               x (ndarray): data.
               x_reduced (ndarray): reduced data.

            Returns:
               mse (float): Mean Squared Error of reconstruction.
            """

        # Compute Pseudo inverse, reconstruct data and compute MSE
        P_inv = np.linalg.pinv(self.model.components_.toarray())  # dimensions check : k x m -> m x k
        x_reconstructed = (P_inv @ x_reduced.T).T  # dimensions check: m x k x k x n = m x n -> n x m
        mse = np.mean((x - x_reconstructed) ** 2)
        return mse
