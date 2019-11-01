# Clustering script

import numpy as np
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import utils

from abc import ABC, abstractmethod

from scipy.stats import mode

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, adjusted_rand_score, adjusted_mutual_info_score, completeness_score, \
                            homogeneity_score, v_measure_score, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture

IMAGE_DIR = 'images/'


class Clustering(ABC):

    def __init__(self, name, n_clusters, random_seed):
        self.name = name
        self.model = None
        self.clusters = None
        self.n_clusters = n_clusters
        self.random_seed = random_seed

    @staticmethod
    def benchmark(x, y, labels):
        print('acc\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
        print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(accuracy_score(y, labels),
                                                                              homogeneity_score(y, labels),
                                                                              completeness_score(y, labels),
                                                                              v_measure_score(y, labels),
                                                                              adjusted_rand_score(y, labels),
                                                                              adjusted_mutual_info_score(y, labels, average_method='arithmetic'),
                                                                              silhouette_score(x, labels, metric='euclidean')))

    def experiment(self, x_train, x_test, y_train, y_test, dataset, max_n_clusters):
        self.plot_model_complexity(x_train, max_n_clusters, dataset)
        self.train(x_train, y_train)
        self.visualize_clusters(x_train, y_train, dataset)
        self.test(x_test, y_test)

    def _match_labels_(self, y, clusters):

        labels = np.zeros_like(clusters)

        for i in range(self.n_clusters):
            mask = (clusters == i)
            labels[mask] = mode(y[mask])[0]

        return labels

    def _plot_clusters_(self, x, y, df, y_colors, clusters_colors, dataset):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        sns.scatterplot(x=x, y=y, hue='y', palette=y_colors, data=df, legend='full', alpha=0.3, ax=ax1)
        sns.scatterplot(x=x, y=y, hue='c', palette=clusters_colors, data=df, legend='full', alpha=0.3, ax=ax2)

        plt.title('{} - {} clusters represented by {}'.format(dataset, self.name, x[:-1].upper()))
        plt.tight_layout()
        plt.savefig(IMAGE_DIR + '{}_{}_clusters_{}'.format(dataset, self.name, x[:-1]))

    @abstractmethod
    def plot_model_complexity(self, x, max_n_clusters, dataset):
        pass

    def set_n_clusters(self, n_clusters):
        self.n_clusters = n_clusters
        self.model.set_params(**{'n_clusters': n_clusters})

    def test(self, x, y):

        print('\nTest on test set')
        clusters = self.model.predict(x)
        labels = self._match_labels_(y, clusters)
        self.benchmark(x, y, labels)

    def train(self, x, y):

        print('\nTrain on training set')
        clusters = self.model.fit_predict(x)
        self.clusters = self._match_labels_(y, clusters)
        self.benchmark(x, y, self.clusters)

    def visualize_clusters(self, x, y, dataset):

        print('\nVisualize Clusters')
        clusters_colors = sns.color_palette('hls', self.n_clusters)
        y_colors = sns.color_palette('hls', len(np.unique(y)))

        pca = PCA(n_components=2, random_state=self.random_seed)
        x_pca = pca.fit_transform(x)
        print('Explained variation per principal component {:3f}'.format(np.sum(pca.explained_variance_ratio_)))

        tsne = TSNE(n_components=2, verbose=1, random_state=self.random_seed)
        x_tsne = tsne.fit_transform(x)

        df = pd.DataFrame(x_tsne, columns=['tsne1', 'tsne2'])
        df['pca1'] = x_pca[:, 0]
        df['pca2'] = x_pca[:, 1]
        df['y'] = y
        df['c'] = self.clusters

        self._plot_clusters_('tsne1', 'tsne2', df, y_colors, clusters_colors, dataset)
        self._plot_clusters_('pca1', 'pca2', df, y_colors, clusters_colors, dataset)


class KMeansClustering(Clustering):
    
    def __init__(self,  n_clusters=2, random_seed=42):
        super(KMeansClustering, self).__init__(name='kmeans', n_clusters=n_clusters, random_seed=random_seed)
        self.model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, random_state=random_seed, n_jobs=-1)

    def plot_model_complexity(self, x, max_n_clusters, dataset):
        
        print('\nPlot Model Complexity')

        self.visualize_silhouette(x, max_n_clusters, dataset)

        silhouette = []
        k_range = np.arange(2, max_n_clusters)

        for k in k_range:
            model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=self.random_seed, n_jobs=-1)
            clusters = model.fit_predict(x)

            silhouette_avg = silhouette_score(x, clusters, metric='euclidean')
            silhouette.append(silhouette_avg)

            print('k = {} --> average silhouette = {:.3f}'.format(k, silhouette_avg))

        plt.figure()
        plt.plot(k_range, silhouette, '-o', markersize=2, label='Silhouette')
        utils.set_plot_title_labels(title='k-Means - Choosing k with the Silhouette method',
                                    x_label='Number of clusters k',
                                    y_label='Average Silhouette')

        # Save figure
        plt.savefig(IMAGE_DIR + '{}_{}_model_complexity'.format(dataset, self.name))

    def visualize_silhouette(self, x, max_n_clusters, dataset):

        fig, ax = plt.subplots(math.ceil((max_n_clusters - 2) / 3), 3, figsize=(12, 12))
        ax = ax.flatten()

        for k in range(2, max_n_clusters):

            model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=self.random_seed, n_jobs=-1)

            clusters = model.fit_predict(x)
            silhouette_avg = silhouette_score(x, clusters, metric='euclidean')

            ax[k-2].axvline(x=silhouette_avg, color="red", linestyle="--")
            silhouette = silhouette_samples(x, clusters, metric='euclidean')
            max_silhouette = np.max(silhouette)

            y_lower = 10

            for i in range(k):
                silhouette_cluster = silhouette[clusters == i]
                silhouette_cluster.sort()

                y_upper = y_lower + silhouette_cluster.shape[0]

                color = cm.nipy_spectral(float(i) / k)
                ax[k-2].fill_betweenx(np.arange(y_lower, y_upper), 0, silhouette_cluster,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax[k-2].text(-0.1, y_lower + 0.5 * silhouette_cluster.shape[0], str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax[k-2].set_title('k-Means - Silhouette for k = {}'.format(k))
            ax[k-2].set_xlabel('Silhouette')
            ax[k-2].set_ylabel('Cluster')

            ax[k-2].set_yticks([])  # Clear the yaxis labels / ticks
            ax[k-2].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            ax[k-2].set_xlim(-0.2, 0.1 + round(max_silhouette, 1))
            ax[k-2].set_ylim(0, len(x) + (k + 1) * 10)

        # Save figure
        plt.tight_layout()
        plt.savefig(IMAGE_DIR + '{}_{}_silhouette'.format(dataset, self.name))


class MixtureOfGaussians(Clustering):
    
    def __init__(self, n_clusters=2, random_seed=42):
        super(MixtureOfGaussians, self).__init__(name='gmm', n_clusters=n_clusters, random_seed=random_seed)
        self.model = GaussianMixture(n_components=n_clusters, covariance_type='full', max_iter=1000,
                                     n_init=10, init_params='random', random_state=random_seed)

    def plot_model_complexity(self, x, max_n_clusters, dataset):
        print('\nPlot Model Complexity')

        bic = []
        k_range = np.arange(2, max_n_clusters)

        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for k in k_range:

                gmm = GaussianMixture(n_components=k, covariance_type=cv_type, max_iter=1000,
                                      n_init=10, init_params='random', random_state=self.random_seed)
                gmm.fit(x)
                bic.append(gmm.bic(x))

                print('cv = {}, k = {} --> bic = {:.3f}'.format(cv_type, k, bic[-1]))

        bic = np.array(bic)

        plt.figure(figsize=(8, 6))
        width = 0.2
        for i, cv_type in enumerate(cv_types):
            x = k_range + width/2 * (i-2)
            plt.bar(x, bic[i * len(k_range):(i + 1) * len(k_range)], width=width, label=cv_type)

        plt.xticks(k_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        utils.set_plot_title_labels(title='EM - Choosing k with the BIC method',
                                    x_label='Number of components k',
                                    y_label='BIC score')

        # Save figure
        plt.savefig(IMAGE_DIR + '{}_{}_model_complexity'.format(dataset, self.name))
