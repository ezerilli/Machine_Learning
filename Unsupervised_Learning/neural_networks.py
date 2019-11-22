# Neural Network class

import time

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:

    def __init__(self, layer1_nodes, layer2_nodes, learning_rate):
        """Initialize a Neural Network classifier.

            Args:
                layer1_nodes (int): number of neurons in the first layer.
                layer2_nodes (int): number of neurons in the second layer.
                learning_rate (float): learning rate.

            Returns:
                None.
            """
        self.model = MLPClassifier(hidden_layer_sizes=(layer1_nodes, layer2_nodes), activation='relu',
                                   solver='sgd', alpha=0.01, batch_size=200, learning_rate='constant',
                                   learning_rate_init=learning_rate, max_iter=100, tol=1e-4,
                                   early_stopping=False, validation_fraction=0.1, momentum=0.5,
                                   n_iter_no_change=100, random_state=42)

    def evaluate(self, x_test, y_test):
        """Evaluate the model by reporting the classification report and the confusion matrix.

            Args:
                x_test (ndarray): test data.
                y_test (ndarray): test labels.

            Returns:
                None.
            """
        predictions = self.predict(x_test)  # predict on test data

        print('\nEvaluate on the Test Set')
        print(classification_report(y_test, predictions))  # produce classification report
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, predictions))  # produce confusion matrix

    def fit(self, x_train, y_train):
        """Fit the model by on the training data.

            Args:
                x_train (ndarray): training data.
                y_train (ndarray): training labels.

            Returns:
                None.
            """
        # Fit the model and report training time
        start_time = time.time()
        self.model.fit(x_train, y_train)
        end_time = time.time()

        print('\nFitting Training Set: {:.4f} seconds'.format(end_time-start_time))

    def predict(self, x):
        """Predict on some data.

            Args:
                x (ndarray): data to predict.

            Returns:
                predictions (ndarray): array of labels predictions.
            """
        # Predict and report inference time
        start_time = time.time()
        predictions = self.model.predict(x)
        end_time = time.time()

        print('\nPredicting on Testing Set: {:.4f} seconds'.format(end_time-start_time))

        return predictions

    def experiment(self, x_train, x_test, y_train, y_test):
        """Run an experiment on the model.

            Fit on training data and evaluate the model on the test set.

            Args:
               x_train (ndarray): training data.
               x_test (ndarray): test data.
               y_train (ndarray): training labels.
               y_test (ndarray): test labels.

            Returns:
               None.
            """
        self.fit(x_train, y_train)
        self.evaluate(x_test, y_test)
