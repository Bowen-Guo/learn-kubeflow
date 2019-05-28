"""Convolutional Neural Network.

Build and train_component a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
This example is using TensorFlow layers API, see 'convolutional_network_raw'
example for a raw implementation with variables.

Refractoring from Project: https://github.com/aymericdamien/TensorFlow-Examples/
by Aymeric Damien

This module is provided by users, not developers.
"""


from __future__ import division, print_function, absolute_import

import gzip
import os
import pickle
import json

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
from sklearn.metrics import accuracy_score


TRAIN_IMAGES = 'train_component-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train_component-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

TRAINED_MODEL_NAME = 'trained_model.pickle'


class CustomLoader:
    """
    To load custom defined model, label and images from disk
    This class is specifically for loading MNIST dataset
    """
    _validation_size = 5000

    def __init__(self, mode: str = 'train_component'):
        """
        :param mode: str, either 'train_component' or 'predict'
        """
        if mode != 'train_component' and mode != 'predict':
            raise ValueError("input mode must be either 'train_component' or 'predict'")
        self._mode = mode

    def load_images_labels(self, data_folder_path):
        """
        Read MNIST images from its local file path

        :param data_folder_path: str
        :return: np.ndarray
        """
        if not os.path.isdir(data_folder_path):
            raise FileNotFoundError(f"{data_folder_path} is not found")

        mnist = input_data.read_data_sets(data_folder_path, one_hot=False)

        if self._mode == 'train_component':
            return mnist.train.images, mnist.train.labels

        return mnist.test.images, mnist.test.labels

    @staticmethod
    def load_prediction_data(data_file_path: str):
        """
        Read prediction data from its local file path and process

        :param data_file_path: str
        :return: np.ndarray
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError(f"{data_file_path} is not found")

        with gfile.Open(data_file_path, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as bytestream:
                buf = bytestream.read()
                return np.frombuffer(buf, dtype=np.uint8)

    @staticmethod
    def load_model(model_folder_path: str):
        """
        Load CNNModel instance with pickle
        :param model_folder_path: str
        :return: custom-defined model instance
        """
        if not os.path.isdir(model_folder_path):
            raise FileNotFoundError(f"{model_folder_path} is not found")

        model_full_path = os.path.join(model_folder_path, TRAINED_MODEL_NAME)

        with open(model_full_path, 'rb') as f:
            return pickle.load(f)


class CustomDumper:
    """
    Dump custom defined model, label, images to disk
    """

    @staticmethod
    def dump_labels(labels, output_file_path: str):
        """
        Dump labels into gzip file with file path of labels_file_path
        :param labels: np.ndarray, to be dumped
        :param output_file_path: str
        """
        with gfile.Open(output_file_path, 'wb') as f:
            with gzip.GzipFile(fileobj=f) as bytestream:
                bytestream.write(labels)

    @staticmethod
    def dump_model(model, output_file_path: str):
        """
        Dump CNNModel instance into disk by pickle serialization
        :param model: user-defined custom model
        :param output_file_path: str, local file path to dump the model
        """
        with open(output_file_path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def dump_dict(dictionary, output_file_path: str):
        """
        Dump dictionary into local json file
        :param dictionary: dict
        :param output_file_path: str
        """
        with open(output_file_path, 'w') as f:
            json.dump(dictionary, f)


class CNNModel:
    def __init__(self):
        # Training Parameters
        self.learning_rate = 0.001
        self.num_steps = 10
        self.batch_size = 128

        # Network Parameters
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.num_classes = 10  # MNIST total classes (0-9 digits)
        self.dropout = 0.25  # Dropout, probability to drop a unit

        self.model = tf.estimator.Estimator(self._model_fn)

    def train(self, data_folder_path):
        """
        Train self.model
        :param data_folder_path: str, local folder path of train_component images and labels
        """
        # Load train_component images and labels
        image_label_loader = CustomLoader()

        train_images, train_labels = image_label_loader.load_images_labels(data_folder_path)

        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': train_images}, y=train_labels,
            batch_size=self.batch_size, num_epochs=None, shuffle=True)

        # Train the model
        self.model.train(input_fn, steps=self.num_steps)

        # Dump the trained model
        output_folder_path = os.environ.get('OUTPUT_0')
        if output_folder_path is None:
            raise ValueError('Environmental variable OUTPUT_0 is not defined')
        output_file_path = os.path.join(output_folder_path, TRAINED_MODEL_NAME)
        CustomDumper.dump_model(self, output_file_path)

    def predict(self, data_folder_path):
        """
        Predict labels
        :param data_folder_path: str, local folder path of predict images
        :return: np.ndarray,
        """
        # Load predict images
        image_label_loader = CustomLoader(mode='predict')
        predict_images, _ = image_label_loader.load_images_labels(data_folder_path)

        # Define the input function for prediction
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': predict_images}, y=None,
            batch_size=1, num_epochs=None, shuffle=False)

        # Predict and return a generator
        predict_generator = self.model.predict(input_fn)

        # Convert the generator into a np.ndarray with dtype of 'uint8'
        predict_ndarray = np.empty(predict_images.shape[0])
        for i in range(predict_images.shape[0]):
            predict_ndarray[i] = next(predict_generator)

        # Dump the predicted result
        output_folder_path = os.environ.get('OUTPUT_0')
        if output_folder_path is None:
            raise ValueError('Environmental variable OUTPUT_0 is not defined')
        output_file_path = os.path.join(output_folder_path, 'predict_labels.gz')
        CustomDumper.dump_labels(predict_ndarray, output_file_path)

        return predict_ndarray.astype('uint8')

    def evaluate(self, predict_labels_folder_path, ground_truth_folder_path):
        """
        Evaluate prediction results by comparing with ground truth result
        :param predict_labels_folder_path: str
        :param ground_truth_folder_path: str
        :return: dict
        """
        # Load predicted and true labels
        predict_labels_file_path = os.path.join(predict_labels_folder_path, 'predict_labels.gz')
        ground_truth_file_path = os.path.join(ground_truth_folder_path, TEST_LABELS)

        predicted_labels = CustomLoader(mode='predict').load_prediction_data(
            predict_labels_file_path)
        true_labels = CustomLoader(mode='predict').load_images_labels(ground_truth_file_path)

        evaluation_result = {
            'Accuracy': accuracy_score(true_labels, predicted_labels)
        }

        # Dump the evaluation result
        output_folder_path = os.environ.get('OUTPUT_0')
        if output_folder_path is None:
            raise ValueError('Environmental variable OUTPUT_0 is not defined')
        output_file_path = os.path.join(output_folder_path, 'evaluation_result.json')
        CustomDumper.dump_dict(evaluation_result, output_file_path)

        return evaluation_result

    # Create the neural network
    def _conv_net(self, x_dict, n_classes, dropout, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs
            x = x_dict['images']

            # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(x, shape=[-1, 28, 28, 1])

            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv2)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, 1024)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc1, n_classes)

        return out

    # Define the model function (following TF Estimator Template)
    def _model_fn(self, features, labels, mode):
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = self._conv_net(features, self.num_classes, self.dropout, reuse=False,
                                      is_training=True)
        logits_test = self._conv_net(features, self.num_classes, self.dropout, reuse=True,
                                     is_training=False)

        # Predictions
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_probas = tf.nn.softmax(logits_test)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs
