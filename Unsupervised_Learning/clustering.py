# Clustering script

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import utils

from abc import ABC, abstractmethod

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, completeness_score, \
                            homogeneity_score, v_measure_score, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture


class Clustering(ABC):

    def __init__(self, name, n_clusters, max_n_clusters, name_param, random_seed):
        """Initialize Clustering object.

           Args:
               name (string): name of the clustering method.
               n_clusters (int): number of clusters.
               max_n_clusters (int): maximum number of clusters.
               name_param (string): name of the number of clusters parameter in Scikit-learn classes.
               random_seed (int): random seed.

           Returns:
               None.
           """
        self.name = name
        self.model = None
        self.clusters = None
        self.n_clusters = n_clusters
        self.max_n_clusters = max_n_clusters
        self.name_param = name_param
        self.random_seed = random_seed

    @staticmethod
    def benchmark(x, y, clusters):
        """Benchmark the model.

           Args:
               x (ndarray): data.
               y (ndarray): true labels.
               clusters (ndarray): clusters found by the model.

           Returns:
               None.
           """
        print('homo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
        print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(homogeneity_score(y, clusters),
                                                                      completeness_score(y, clusters),
                                                                      v_measure_score(y, clusters),
                                                                      adjusted_rand_score(y, clusters),
                                                                      adjusted_mutual_info_score(y, clusters, average_method='arithmetic'),
                                                                      silhouette_score(x, clusters, metric='euclidean')))

    def experiment(self, x_train, x_test, y_train, dataset, perform_model_complexity):
        """Perform experiments.

            Args:
               x_train (ndarray): training data.
               x_test (ndarray): test data.
               y_train (ndarray): training labels.
               dataset (string): dataset, WDBC or MNIST.
               perform_model_complexity (bool): True if we want to perform model complexity.

            Returns:
              clusters (ndarray): training clusters.
              predicted_clusters (ndarray): test clusters.
            """

        if perform_model_complexity:
            self.plot_model_complexity(x_train, dataset)

        self.train(x_train, y_train)  # fit the model and benchmark on training data
        self.visualize_clusters(x_train, y_train, dataset)  # visualize clusters with PCA and TSNE

        return self.clusters, self.predict(x_test)  # predict on test data and return training and test clusters

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

    def predict(self, x):
        """Predict.

            Args:
               x (ndarray): data.

            Returns:
              predictions (ndarray): predicted clusters.
            """
        return self.model.predict(x)

    def set_n_clusters(self, n_clusters):
        """Set number of clusters.

            Args:
               n_clusters (int): number of clusters.

            Returns:
              None.
            """
        self.n_clusters = n_clusters
        self.model.set_params(**{'n_clusters': n_clusters})

    def train(self, x, y):
        """Fit and benchmark the model on training data.

            Args:
               x (ndarray): data.
               y (ndarray): true labels.

            Returns:
              None.
            """

        print('\nTrain on training set with k={}'.format(self.n_clusters))
        self.clusters = self.model.fit_predict(x)
        self.benchmark(x, y, self.clusters)

    def visualize_clusters(self, x, y, dataset):
        """Visualize clusters with PCA and TSNE.

            Args:
               x (ndarray): data.
               y (ndarray): true labels.
               dataset (string): dataset, WDBC or MNIST.

            Returns:
              None.
            """

        # Declare PCA and reduce data
        pca = PCA(n_components=2, random_state=self.random_seed)
        x_pca = pca.fit_transform(x)

        # Declare TSNE and reduce data
        tsne = TSNE(n_components=2, random_state=self.random_seed)
        x_tsne = tsne.fit_transform(x)

        n_classes = len(np.unique(y))  # compute number of classes
        print('\nBenchmark Model with k = n classes = {}'.format(n_classes))

        # Benchamark the model with number of clusters (k) = number of classes
        model = clone(self.model)
        model_params = self.model.get_params()
        model_params[self.name_param] = n_classes
        model.set_params(**model_params)
        clusters = model.fit_predict(x)
        self.benchmark(x, y, clusters)

        # Create dataframe for visualization
        df = pd.DataFrame(x_tsne, columns=['tsne1', 'tsne2'])
        df['pca1'] = x_pca[:, 0]
        df['pca2'] = x_pca[:, 1]
        df['y'] = y
        df['c'] = self.clusters

        # Create subplot and plot clusters with PCA and TSNE
        fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 8))
        utils.plot_clusters(ax1, 'pca1', 'pca2', df, self.name)
        utils.plot_clusters(ax2, 'tsne1', 'tsne2', df, self.name)

        # Save figure
        utils.save_figure_tight('{}_{}_clusters'.format(dataset, self.name))


class KMeansClustering(Clustering):
    
    def __init__(self,  n_clusters=2, max_n_clusters=10, random_seed=42):
        """Initialize k-Means Clustering object.

            Args:
               n_clusters (int): number of clusters.
               max_n_clusters (int): maximum number of clusters.
               random_seed (int): random seed.

            Returns:
               None.
            """

        super(KMeansClustering, self).__init__(name='k-means', n_clusters=n_clusters, max_n_clusters=max_n_clusters,
                                               name_param='n_clusters', random_seed=random_seed)
        self.model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, random_state=random_seed, n_jobs=-1)

    def plot_model_complexity(self, x, dataset):
        """Perform and plot model complexity.

            Args:
               x (ndarray): training data.
               dataset (string): dataset, WDBC or MNIST.

            Returns:
               None.
            """
        
        print('\nPlot Model Complexity')

        inertia, inertia_diff = [], []  # inertia and delta inertia
        k_range = np.arange(1, self.max_n_clusters + 2)  # range of number of clusters k to plot over

        # For each k in the range
        for k in k_range:

            # Define a new k-Means model, fit on training data and report inertia
            model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=self.random_seed, n_jobs=-1)
            model.fit(x)
            inertia.append(model.inertia_)
            print('k = {} -->  inertia = {:.3f}'.format(k, inertia[-1]))

            # Except for k=1, report also the delta of inertia
            if k > 1:
                inertia_diff.append(abs(inertia[-1] - inertia[-2]))

        # Create subplots and plot inertia and delta inertia on the first suplot
        fig, ax = plt.subplots(2, math.ceil(self.max_n_clusters / 2), figsize=(24, 12))
        ax = ax.flatten()
        ax[0].plot(k_range, inertia, '-o', markersize=2, label='Inertia')
        ax[0].plot(k_range[1:], inertia_diff, '-o', markersize=2, label=r'Inertia |$\Delta$|')

        # Set legend, title and labels
        ax[0].legend(loc='best')
        utils.set_axis_title_labels(ax[0], title='K-MEANS - Choosing k with the Elbow method',
                                    x_label='Number of clusters k', y_label='Inertia')

        # Plot silhouette for the different k values
        self.visualize_silhouette(x, ax)

        # Save figure
        utils.save_figure_tight('{}_{}_model_complexity'.format(dataset, self.name))

    def visualize_silhouette(self, x, ax):
        """Visualize silhouette.

            Args:
               x (ndarray): training data.
               ax (ndarray): vector of axes to plot at.

            Returns:
               None.
            """

        # For all k values starting from k=2 (for k=1, the silhouette score is not defined)
        for k in range(2, self.max_n_clusters + 1):

            # Define a new k-Means model, fit on training data and report clusters and  average silhouette
            model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=self.random_seed, n_jobs=-1)
            clusters = model.fit_predict(x)
            silhouette_avg = silhouette_score(x, clusters, metric='euclidean')

            # Plot red dashed vertical line for average silhouette
            ax[k-1].axvline(x=silhouette_avg, color="red", linestyle="--")

            # Compute silhouette scores and maximum silhouette
            silhouette = silhouette_samples(x, clusters, metric='euclidean')
            max_silhouette = np.max(silhouette)

            y_lower = 10  # starting y for plotting silhouette

            # For each cluster found
            for i in range(k):

                # Sort silhouette of current cluster
                silhouette_cluster = silhouette[clusters == i]
                silhouette_cluster.sort()

                # Compute the upper y for plotting silhouette
                y_upper = y_lower + silhouette_cluster.shape[0]

                # Fill the area corresponding to the current cluster silhouette scores
                color = plt.cm.nipy_spectral(float(i) / k)
                ax[k-1].fill_betweenx(np.arange(y_lower, y_upper), 0, silhouette_cluster,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax[k-1].text(-0.1, y_lower + 0.5 * silhouette_cluster.shape[0], str(i))

                # Compute the new lower y for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            # Set title and labels
            utils.set_axis_title_labels(ax[k-1], title='K-MEANS - Silhouette for k = {}'.format(k),
                                        x_label='Silhouette', y_label='Silhouette distribution per Cluster')

            # Clear the y axis labels and set the x ones
            ax[k-1].set_yticks([])
            ax[k-1].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # Set x and y limits
            ax[k-1].set_xlim(-0.2, 0.1 + round(max_silhouette, 1))
            ax[k-1].set_ylim(0, len(x) + (k + 1) * 10)


class MixtureOfGaussians(Clustering):
    
    def __init__(self, n_clusters=2, covariance='full', max_n_clusters=10, random_seed=42):
        """Initialize Gaussian Mixture Models Clustering object.

            Args:
               n_clusters (int): number of clusters.
               covariance (string): covariance type.
               max_n_clusters (int): maximum number of clusters.
               random_seed (int): random seed.

            Returns:
               None.
            """

        super(MixtureOfGaussians, self).__init__(name='em', n_clusters=n_clusters, max_n_clusters=max_n_clusters,
                                                 name_param='n_components', random_seed=random_seed)
        self.model = GaussianMixture(n_components=n_clusters, covariance_type=covariance, max_iter=1000,
                                     n_init=10, init_params='random', random_state=random_seed)

    def plot_model_complexity(self, x, dataset):
        """Perform and plot model complexity.

            Args:
               x (ndarray): training data.
               dataset (string): dataset, WDBC or MNIST.

            Returns:
               None.
            """

        print('\nPlot Model Complexity')

        aic, bic = [], []  # AIC and BIC  scores lists
        k_range = np.arange(2, self.max_n_clusters + 1)  # range of number of clusters k to plot over

        cv_types = ['spherical', 'tied', 'diag', 'full']  # covariance types to plot over

        # For all pair covariance, number of clusters k
        for cv_type in cv_types:
            for k in k_range:

                # Define a new Gaussian Mixture model, fit on training data and report AIC and BIC
                gmm = GaussianMixture(n_components=k, covariance_type=cv_type, max_iter=1000,
                                      n_init=10, init_params='random', random_state=self.random_seed)
                gmm.fit(x)
                aic.append(gmm.aic(x))
                bic.append(gmm.bic(x))

                print('cv = {}, k = {} --> aic = {:.3f}, bic = {:.3f}'.format(cv_type, k, aic[-1], bic[-1]))

        # Create subplots and plot AIC and BIC histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        utils.plot_ic_bars(np.array(aic), 'AIC', cv_types, k_range, ax1)
        utils.plot_ic_bars(np.array(bic), 'BIC', cv_types, k_range, ax2)

        # Save figure
        utils.save_figure_tight('{}_{}_model_complexity'.format(dataset, self.name))
