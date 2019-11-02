# Script to run experiments

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import utils

from clustering import KMeansClustering, MixtureOfGaussians
from dimensionality_reduction import IndependentComponents, NonNegativeMatrix, PrincipalComponents, RandomProjections

# from neural_networks import NeuralNetwork

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


def load_dataset(dataset='WDBC', split_percentage=0.2, visualize=False):
    """Load MNIST or WDBC.

       Args:
           dataset (string): dataset, WDBC or MNIST.
           split_percentage (float): validation split.
           visualize (bool): True if some of the dataset images have to been shown.

       Returns:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           y_test (ndarray): test labels.
       """

    datasets = ['WDBC', 'MNIST']  # datasets names
    print('\nLoading {} Dataset'.format(dataset))

    if dataset == datasets[0]:

        # Load WDBC
        data = load_breast_cancer()
        x, y, labels, features = data.data, data.target, data.target_names, data.feature_names

        # If some of the dataset images have to been shown
        if visualize:

            # Build dataset and assign labels
            df = pd.DataFrame(x, columns=features)
            df['labels'] = y
            df['labels'] = df['labels'].map({1: 'B', 0: 'M'})

            # Plot instances distribution
            plt.figure()
            sns.set(style='darkgrid')
            sns.countplot(x='labels', data=df, palette={'B': 'b', 'M': 'r'})
            plt.title('{} Instances Distribution'.format(dataset))
            utils.save_figure('{}_Instances_Distribution'.format(dataset))

            # Plot heatmap of correlations
            plt.figure(figsize=(15, 15))
            sns.heatmap(df.corr(), annot=True, square=True, cmap='coolwarm')
            utils.save_figure('{}_Features_Correlation'.format(dataset))

            # Plot scatter matrix of features
            plt.figure(figsize=(15, 15))
            sns.pairplot(df, hue='labels', palette={'B': 'b', 'M': 'r'})
            utils.save_figure('{}_Scatter_Matrix_of_Features'.format(dataset))

            # Plot features distributions
            bins = 12
            plt.figure(figsize=(15, 15))
            for i, feature in enumerate(features):
                plt.subplot(5, 2, i + 1)

                sns.distplot(x[y == 0, i], bins=bins, color='red', label='M')
                sns.distplot(x[y == 1, i], bins=bins, color='blue', label='B')

                plt.legend(loc='lower left')
                plt.xlabel(feature)

            utils.save_figure('{}_Features_Discrimination'.format(dataset))
            plt.close(fig='all')

    elif dataset == datasets[1]:

        # Load original MNIST
        data = fetch_openml('mnist_784')
        x, y = data.data, data.target

        # Generate a statified smaller subset of MNIST
        x, _, y, _ = train_test_split(x, y, test_size=0.9, shuffle=True, random_state=42, stratify=y)
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # labels
        y = y.astype(int)

        if visualize:
            # Show images with labels
            for i in range(2 * 3):
                plt.subplot(2, 3, i + 1)
                plt.imshow(x[i].reshape((28, 28)), cmap=plt.cm.gray)
                plt.title('{}'.format(y[i]), size=12)
                plt.xticks(())
                plt.yticks(())

            plt.show()

    else:
        # Else dataset not available
        raise Exception('Wrong dataset name. Datasets available = {}'.format(datasets))

    # Split dataset in training and validation sets, preserving classes representation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_percentage, shuffle=True,
                                                        random_state=42, stratify=y)

    # Normalize feature data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print('\nTotal dataset size:')
    print('Number of instances: {}'.format(x.shape[0]))
    print('Number of features: {}'.format(x.shape[1]))
    print('Number of classes: {}'.format(len(labels)))
    print('Training Set : {}'.format(x_train.shape))
    print('Testing Set : {}'.format(x_test.shape))

    return x_train, x_test, y_train, y_test


def clustering(x_train, x_test, y_train, y_test, **kwargs):
    """Perform experiment.

        Args:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           y_test (ndarray): test labels.

        Returns:
           None.
        """

    print('\n--------------------------')
    print('kMeans')
    kmeans = KMeansClustering(n_clusters=kwargs['kmeans_n_clusters'], max_n_clusters=10)
    kmeans.experiment(x_train, x_test, y_train, y_test, dataset=kwargs['dataset'], perform_model_complexity=False)

    print('\n--------------------------')
    print('EM')
    gmm = MixtureOfGaussians(n_clusters=kwargs['em_n_clusters'], covariance=kwargs['em_covariance'], max_n_clusters=10)
    gmm.experiment(x_train, x_test, y_train, y_test, dataset=kwargs['dataset'], perform_model_complexity=False)


def dimensionality_reduction(x_train, x_test, y_train, **kwargs):
    """Perform experiment.

        Args:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.

        Returns:
           None.
        """

    print('\n--------------------------')
    print('PCA')
    pca = PrincipalComponents(n_components=kwargs['pca_n_components'])
    pca.experiment(x_train, x_test, y_train, dataset=kwargs['dataset'], perform_model_complexity=False)

    print('\n--------------------------')
    print('ICA')
    ica = IndependentComponents(n_components=kwargs['ica_n_components'])
    ica.experiment(x_train, x_test, y_train, dataset=kwargs['dataset'], perform_model_complexity=False)

    print('\n--------------------------')
    print('RP')
    rp = RandomProjections(n_components=kwargs['rp_n_components'])
    rp.experiment(x_train, x_test, y_train, dataset=kwargs['dataset'], perform_model_complexity=False)

    print('\n--------------------------')
    print('NMF')
    nmf = NonNegativeMatrix(n_components=kwargs['nmf_n_components'])
    nmf.experiment(x_train, x_test, y_train, dataset=kwargs['dataset'], perform_model_complexity=True)


if __name__ == "__main__":

    # Run experiment 1 on WDBC
    print('\n--------------------------')
    dataset = 'WDBC'
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    # clustering(x_train, x_test, y_train, y_test,
    #            dataset=dataset,
    #            kmeans_n_clusters=2,
    #            em_n_clusters=2, em_covariance='full')
    dimensionality_reduction(x_train, x_test, y_train,
                             dataset=dataset,
                             pca_n_components=10,
                             ica_n_components=12,
                             rp_n_components=20,
                             nmf_n_components=15)

    # Run experiment 2 on MNIST
    print('\n--------------------------')
    dataset = 'MNIST'
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    # clustering(x_train, x_test, y_train, y_test,
    #            dataset=dataset,
    #            kmeans_n_clusters=2,
    #            em_n_clusters=10, em_covariance='diag')
    dimensionality_reduction(x_train, x_test, y_train,
                             dataset=dataset,
                             pca_n_components=250,
                             ica_n_components=320,
                             rp_n_components=500,
                             nmf_n_components=400)
