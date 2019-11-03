# Clustering script

import numpy as np
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import utils

from abc import ABC, abstractmethod

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, adjusted_rand_score, adjusted_mutual_info_score, completeness_score, \
                            homogeneity_score, v_measure_score, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture


class Clustering(ABC):

    def __init__(self, name, n_clusters, max_n_clusters, name_param, random_seed):
        self.name = name
        self.model = None
        self.clusters = None
        self.n_clusters = n_clusters
        self.max_n_clusters = max_n_clusters
        self.name_param = name_param
        self.random_seed = random_seed

    @staticmethod
    def benchmark(x, y, labels):
        print('homo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
        print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(homogeneity_score(y, labels),
                                                                      completeness_score(y, labels),
                                                                      v_measure_score(y, labels),
                                                                      adjusted_rand_score(y, labels),
                                                                      adjusted_mutual_info_score(y, labels, average_method='arithmetic'),
                                                                      silhouette_score(x, labels, metric='euclidean')))

    def experiment(self, x_train, x_test, y_train, dataset, perform_model_complexity):

        if perform_model_complexity:
            self.plot_model_complexity(x_train, dataset)

        self.train(x_train, y_train)
        self.visualize_clusters(x_train, y_train, dataset)

        return self.clusters, self.predict(x_test)

    @abstractmethod
    def plot_model_complexity(self, x, dataset):
        pass

    def set_n_clusters(self, n_clusters):
        self.n_clusters = n_clusters
        self.model.set_params(**{'n_clusters': n_clusters})

    def predict(self, x):
        return self.model.predict(x)

    def train(self, x, y):

        print('\nTrain on training set with k={}'.format(self.n_clusters))
        self.clusters = self.model.fit_predict(x)
        self.benchmark(x, y, self.clusters)

    def visualize_clusters(self, x, y, dataset):

        pca = PCA(n_components=2, random_state=self.random_seed)
        x_pca = pca.fit_transform(x)

        tsne = TSNE(n_components=2, random_state=self.random_seed)
        x_tsne = tsne.fit_transform(x)

        n_classes = len(np.unique(y))

        model = clone(self.model)
        model_params = self.model.get_params()
        model_params[self.name_param] = n_classes
        model.set_params(**model_params)
        print('\nBenchmark Model with k={}'.format(n_classes))
        clusters = model.fit_predict(x)
        self.benchmark(x, y, clusters)

        df = pd.DataFrame(x_tsne, columns=['tsne1', 'tsne2'])
        df['pca1'] = x_pca[:, 0]
        df['pca2'] = x_pca[:, 1]
        df['y'] = y
        df['true c'] = clusters
        df['c'] = self.clusters

        fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(15, 8))

        utils.plot_clusters(ax1, 'pca1', 'pca2', df, y, self.name)
        utils.plot_clusters(ax2, 'tsne1', 'tsne2', df, y, self.name)
        utils.save_figure_tight('{}_{}_clusters'.format(dataset, self.name))


class KMeansClustering(Clustering):
    
    def __init__(self,  n_clusters=2, max_n_clusters=10, random_seed=42):
        super(KMeansClustering, self).__init__(name='k-means', n_clusters=n_clusters, max_n_clusters=max_n_clusters,
                                               name_param='n_clusters', random_seed=random_seed)

        self.model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, random_state=random_seed, n_jobs=-1)

    def plot_model_complexity(self, x, dataset):
        
        print('\nPlot Model Complexity')

        silhouette = []
        k_range = np.arange(2, self.max_n_clusters + 1)

        for k in k_range:
            model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=self.random_seed, n_jobs=-1)
            clusters = model.fit_predict(x)

            silhouette_avg = silhouette_score(x, clusters, metric='euclidean')
            silhouette.append(silhouette_avg)

            print('k = {} --> average silhouette = {:.3f}'.format(k, silhouette_avg))

        fig, ax = plt.subplots(2, math.ceil(self.max_n_clusters / 2), figsize=(24, 12))
        ax = ax.flatten()
        ax[0].plot(k_range, silhouette, '-o', markersize=2, label='Silhouette')
        utils.set_axis_title_labels(ax[0], title='K-MEANS - Choosing k with the Silhouette method',
                                    x_label='Number of clusters k', y_label='Average Silhouette')

        self.visualize_silhouette(x, ax)

        # Save figure
        utils.save_figure_tight('{}_{}_model_complexity'.format(dataset, self.name))

    def visualize_silhouette(self, x, ax):

        for k in range(2, self.max_n_clusters + 1):

            model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=self.random_seed, n_jobs=-1)

            clusters = model.fit_predict(x)
            silhouette_avg = silhouette_score(x, clusters, metric='euclidean')

            ax[k-1].axvline(x=silhouette_avg, color="red", linestyle="--")
            silhouette = silhouette_samples(x, clusters, metric='euclidean')
            max_silhouette = np.max(silhouette)

            y_lower = 10

            for i in range(k):
                silhouette_cluster = silhouette[clusters == i]
                silhouette_cluster.sort()

                y_upper = y_lower + silhouette_cluster.shape[0]

                color = plt.cm.nipy_spectral(float(i) / k)
                ax[k-1].fill_betweenx(np.arange(y_lower, y_upper), 0, silhouette_cluster,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax[k-1].text(-0.1, y_lower + 0.5 * silhouette_cluster.shape[0], str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            utils.set_axis_title_labels(ax[k-1], title='K-MEANS - Silhouette for k = {}'.format(k),
                                        x_label='Silhouette', y_label='Silhouette distribution per Cluster')

            ax[k-1].set_yticks([])  # Clear the yaxis labels / ticks
            ax[k-1].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            ax[k-1].set_xlim(-0.2, 0.1 + round(max_silhouette, 1))
            ax[k-1].set_ylim(0, len(x) + (k + 1) * 10)


class MixtureOfGaussians(Clustering):
    
    def __init__(self, n_clusters=2, covariance='full', max_n_clusters=10, random_seed=42):
        super(MixtureOfGaussians, self).__init__(name='em', n_clusters=n_clusters, max_n_clusters=max_n_clusters,
                                                 name_param='n_components', random_seed=random_seed)

        self.model = GaussianMixture(n_components=n_clusters, covariance_type=covariance, max_iter=1000,
                                     n_init=10, init_params='random', random_state=random_seed)

    def plot_model_complexity(self, x, dataset):
        print('\nPlot Model Complexity')

        aic, bic = [], []
        k_range = np.arange(2, self.max_n_clusters + 1)

        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for k in k_range:

                gmm = GaussianMixture(n_components=k, covariance_type=cv_type, max_iter=1000,
                                      n_init=10, init_params='random', random_state=self.random_seed)
                gmm.fit(x)

                aic.append(gmm.aic(x))
                bic.append(gmm.bic(x))

                print('cv = {}, k = {} --> aic = {:.3f}, bic = {:.3f}'.format(cv_type, k, aic[-1], bic[-1]))

        aic = np.array(aic)
        bic = np.array(bic)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        utils.plot_ic_bars(aic, 'AIC', cv_types, k_range, ax1)
        utils.plot_ic_bars(bic, 'BIC', cv_types, k_range, ax2)

        # Save figure
        utils.save_figure_tight('{}_{}_model_complexity'.format(dataset, self.name))
