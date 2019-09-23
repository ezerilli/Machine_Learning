import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from adaboost import AdaBoost
from decision_trees import DecisionTree
from k_nearest_neighbors import KNN
from neural_networks import NeuralNetwork
from support_vector_machines import SVM

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split

IMAGE_DIR = 'images/'


def load_dataset(dataset='WDBC', split_percentage=0.2, visualize=False):

    datasets = ['WDBC', 'MNIST']

    if dataset == datasets[0]:
        data = load_breast_cancer()
        x, y, labels, features = data.data, data.target, data.target_names, data.feature_names

        if visualize:
            df = pd.DataFrame(x, columns=features)
            df['labels'] = y
            df['labels'] = df['labels'].map({1: 'B', 0: 'M'})

            plt.figure()
            sns.set(style='darkgrid')
            sns.countplot(x='labels', data=df, palette={'B': 'b', 'M': 'r'})
            plt.title('{} Instances Distribution'.format(dataset))
            plt.savefig(IMAGE_DIR + '{}_Instances_Distribution'.format(dataset))

            plt.figure(figsize=(15, 15))
            sns.heatmap(df.corr(), annot=True, square=True, cmap='coolwarm')
            plt.savefig(IMAGE_DIR + '{}_Features_Correlation'.format(dataset))

            plt.figure(figsize=(15, 15))
            sns.pairplot(df, hue='labels', palette={'B': 'b', 'M': 'r'})
            plt.savefig(IMAGE_DIR + '{}_Scatter_Matrix_of_Features'.format(dataset))

            bins = 12
            plt.figure(figsize=(15, 15))
            for i, feature in enumerate(features):
                plt.subplot(5, 2, i + 1)

                sns.distplot(x[y == 0, i], bins=bins, color='red', label='M')
                sns.distplot(x[y == 1, i], bins=bins, color='blue', label='B')

                plt.legend(loc='lower left')
                plt.xlabel(feature)

            plt.tight_layout()
            plt.savefig(IMAGE_DIR + '{}_Features_Discrimination'.format(dataset))
            plt.close(fig='all')

    elif dataset == datasets[1]:
        print('Loading {} Dataset'.format(dataset))
        data = fetch_openml('mnist_784')
        x, y = data.data, data.target
        x, _, y, _ = train_test_split(x, y, test_size=0.98, shuffle=True, random_state=42, stratify=y)
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        if visualize:
            for i in range(2 * 3):
                plt.subplot(2, 3, i + 1)
                plt.imshow(x[i].reshape((28, 28)), cmap=plt.cm.gray)
                # plt.title(titles[i], size=12)
                plt.xticks(())
                plt.yticks(())

        plt.show()

    else:
        raise Exception('Wrong dataset name. Datasets available = {}'.format(datasets))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_percentage, shuffle=True,
                                                        random_state=42, stratify=y)

    print('\nTotal dataset size:')
    print('Number of instances: {}'.format(x.shape[0]))
    print('Number of features: {}'.format(x.shape[1]))
    print('Number of classes: {}'.format(len(labels)))
    print('Training Set : {}'.format(x_train.shape))
    print('Testing Set : {}'.format(x_test.shape))

    return x_train, x_test, y_train, y_test


def experiment1(x_train, x_test, y_train, y_test):

    training_sizes = np.arange(20, int(len(x_train) * 0.9), 10)

    print('\n--------------------------')
    knn = KNN(k=1, weights='uniform', p=2)
    knn.experiment(x_train, x_test, y_train, y_test,
                   cv=10,
                   y_lim=0.75,
                   n_neighbors_range=np.arange(1, 100, 2),
                   p_range=np.arange(1, 50),
                   weight_functions=['uniform', 'distance'],
                   train_sizes=training_sizes)

    print('\n--------------------------')
    svm = SVM(c=1., kernel='rbf', degree=3, gamma=0.001, random_state=42)
    svm.experiment(x_train, x_test, y_train, y_test,
                   cv=10,
                   y_lim=0.2,
                   C_range=[1, 5] + list(range(10, 100, 20)) + list(range(100, 1000, 50)),
                   kernels=['linear', 'poly', 'rbf'],
                   gamma_range=np.logspace(-6, 0, 50),
                   poly_degrees=[2, 3, 4],
                   train_sizes=training_sizes)

    print('\n--------------------------')
    dt = DecisionTree(max_depth=1, min_samples_leaf=1, random_state=42)
    dt.experiment(x_train, x_test, y_train, y_test,
                  cv=10,
                  y_lim=0.5,
                  max_depth_range=list(range(1, 50)),
                  min_samples_leaf_range=list(range(1, 50)),
                  train_sizes=training_sizes)

    print('\n--------------------------')
    boosted_dt = AdaBoost(n_estimators=50, learning_rate=1., max_depth=3, random_state=42)
    boosted_dt.experiment(x_train, x_test, y_train, y_test,
                          cv=10,
                          y_lim=0.7,
                          max_depth_range=list(range(1, 30)),
                          n_estimators_range=[1, 3, 5, 8] + list(range(10, 100, 5)) + list(range(100, 1000, 50)),
                          learning_rate_range=np.logspace(-6, 1, 50),
                          train_sizes=training_sizes)

    print('\n--------------------------')
    nn = NeuralNetwork(alpha=0.01, layer1_nodes=50, layer2_nodes=30, learning_rate=0.001, max_iter=100)
    nn.experiment(x_train, x_test, y_train, y_test,
                  cv=10,
                  y_lim=0.65,
                  alpha_range=np.logspace(-5, 1, 50),
                  learning_rate_range=np.logspace(-4, 0, 50),
                  train_sizes=training_sizes)


def experiment2(x_train, x_test, y_train, y_test):

    training_sizes = np.arange(20, int(len(x_train) * 0.89), 20)

    # print('\n--------------------------')
    # knn = KNN(k=1, weights='uniform', p=2)
    # knn.experiment(x_train, x_test, y_train, y_test,
    #                cv=10,
    #                y_lim=0.3,
    #                n_neighbors_range=np.arange(1, 50, 2),
    #                p_range=np.arange(1, 20),
    #                weight_functions=['uniform', 'distance'],
    #                train_sizes=training_sizes)
    #
    # print('\n--------------------------')
    # svm = SVM(c=1., kernel='rbf', degree=3, gamma=0.001, random_state=42)
    # svm.experiment(x_train, x_test, y_train, y_test,
    #                cv=10,
    #                y_lim=0.2,
    #                C_range=[1, 5] + list(range(10, 100, 20)) + list(range(100, 1000, 50)),
    #                kernels=['linear', 'poly', 'rbf'],
    #                gamma_range=np.logspace(-7, -1, 50),
    #                poly_degrees=[2, 3, 4],
    #                train_sizes=training_sizes)

    # print('\n--------------------------')
    # dt = DecisionTree(max_depth=1, min_samples_leaf=1, random_state=42)
    # dt.experiment(x_train, x_test, y_train, y_test,
    #               cv=10,
    #               y_lim=0.1,
    #               max_depth_range=list(range(1, 100, 1)),
    #               min_samples_leaf_range=list(range(1, 30)),
    #               train_sizes=training_sizes)

    print('\n--------------------------')
    boosted_dt = AdaBoost(n_estimators=50, learning_rate=1., max_depth=3, random_state=42)
    boosted_dt.experiment(x_train, x_test, y_train, y_test,
                          cv=10,
                          y_lim=0.2,
                          max_depth_range=list(range(1, 20)),
                          n_estimators_range=[1, 5] + list(range(5, 100, 5)),
                          learning_rate_range=np.logspace(-6, 1, 30),
                          train_sizes=training_sizes)

    print('\n--------------------------')
    nn = NeuralNetwork(alpha=0.01, layer1_nodes=150, layer2_nodes=100, learning_rate=0.001, max_iter=20)
    nn.experiment(x_train, x_test, y_train, y_test,
                  cv=10,
                  y_lim=0.1,
                  alpha_range=np.logspace(-5, 1, 30),
                  learning_rate_range=np.logspace(-4, 0, 30),
                  train_sizes=training_sizes)


print('\n--------------------------')
# x_train, x_test, y_train, y_test = load_dataset(dataset='WDBC', visualize=False)
# experiment1(x_train, x_test, y_train, y_test)

x_train, x_test, y_train, y_test = load_dataset(dataset='MNIST', visualize=False)
experiment2(x_train, x_test, y_train, y_test)

