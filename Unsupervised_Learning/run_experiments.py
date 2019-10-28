# Script to run experiments

import numpy as np
import matplotlib.pyplot as plt

# from neural_networks import NeuralNetwork

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'


def load_dataset(dataset='WDBC', split_percentage=0.2):
    """Load MNIST or WDBC.

       Args:
           dataset (string): dataset, WDBC or MNIST.
           split_percentage (float): validation split.

       Returns:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           y_test (ndarray): test labels.
       """

    datasets = ['WDBC', 'MNIST']  # datasets names
    print('Loading {} Dataset'.format(dataset))

    if dataset == datasets[0]:

        # Load WDBC
        data = load_breast_cancer()
        x, y, labels, features = data.data, data.target, data.target_names, data.feature_names

    elif dataset == datasets[1]:

        # Load original MNIST
        data = fetch_openml('mnist_784')
        x, y = data.data, data.target

        # Generate a statified smaller subset of MNIST
        x, _, y, _ = train_test_split(x, y, test_size=0.98, shuffle=True, random_state=42, stratify=y)
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # labels

    else:
        # Else dataset not available
        raise Exception('Wrong dataset name. Datasets available = {}'.format(datasets))

    # Slit dataset in training and validation sets, preserving classes representation
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


def experiment(x_train, x_test, y_train, y_test):
    """Perform experiment.

        Args:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           y_test (ndarray): test labels.

        Returns:
           None.
        """

    # TODO : write unsupervised learning


if __name__ == "__main__":

    # Run experiment 1 on WDBC
    print('\n--------------------------')
    x_train, x_test, y_train, y_test = load_dataset(dataset='WDBC')
    experiment(x_train, x_test, y_train, y_test)

    # Run experiment 2 on MNIST
    x_train, x_test, y_train, y_test = load_dataset(dataset='MNIST')
    experiment(x_train, x_test, y_train, y_test)
