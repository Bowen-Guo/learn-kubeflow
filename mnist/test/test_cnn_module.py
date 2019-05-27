import os
import pytest

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from azureml.studio.common.utils.fileutil import ensure_folder
from azureml.studio.modules.python_language_modules.cnn_demo.cnn_module import CNNModel, CustomLoader, CustomDumper


TRAIN_IMAGES = 'train_component-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train_component-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


def input_test_folder_path():
    """
    :return: str, input folder path
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')


def output_test_folder_path():
    """
    :return: str, output folder path
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen')


def set_output_port_folder(output_folder: str, port_number: int):
    """
    Set the output_port folder as an environmental variable
    :param output_folder: str, output folder path
    :param port_number: int, index of the output port
    """
    output_port_path = os.path.join(output_folder, f'output{port_number}')
    ensure_folder(output_port_path)
    os.environ.update({
        f'OUTPUT_{port_number}': output_port_path
    })


@pytest.fixture()
def load_data():
    """
    Load MNIST data
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
    mnist = input_data.read_data_sets(file_path)
    return mnist


@pytest.fixture()
def train_images_labels(load_data):
    """
    :param load_data: MNIST data
    :return: (np.ndarray, np.ndarray)
    """
    return load_data.train.images, load_data.train.labels


@pytest.fixture()
def predict_images_labels(load_data):
    """
    :param load_data: MNIST data
    :return: (np.ndarray, np.ndarray)
    """
    return load_data.test.images, load_data.test.labels


"""
Test class CustomLoader
"""


def test_load_images_train_mode(train_images_labels):
    input_images_file_path = os.path.join(input_test_folder_path(), TRAIN_IMAGES)
    image_label_loader = CustomLoader(mode='train_component')
    computed_images = image_label_loader.load_images(input_images_file_path)
    expected_images = train_images_labels[0]
    assert np.array_equal(computed_images, expected_images)


def test_load_images_predict_mode(predict_images_labels):
    input_images_file_path = os.path.join(input_test_folder_path(), TEST_IMAGES)
    image_label_loader = CustomLoader(mode='predict')
    computed_images = image_label_loader.load_images(input_images_file_path)
    expected_images = predict_images_labels[0]
    assert np.array_equal(computed_images, expected_images)


def test_load_labels_train_mode(train_images_labels):
    input_labels_file_path = os.path.join(input_test_folder_path(), TRAIN_LABELS)
    image_label_loader = CustomLoader(mode='train_component')
    computed_labels = image_label_loader.load_labels(input_labels_file_path)
    expected_labels = train_images_labels[1]
    assert np.array_equal(computed_labels, expected_labels)


def test_load_labels_predict_mode(predict_images_labels):
    input_labels_file_path = os.path.join(input_test_folder_path(), TEST_LABELS)
    image_label_loader = CustomLoader(mode='predict')
    computed_labels = image_label_loader.load_labels(input_labels_file_path)
    expected_labels = predict_images_labels[1]
    assert np.array_equal(computed_labels, expected_labels)


def test_load_model():
    # Smoke test
    input_model_file_path = os.path.join(input_test_folder_path(), 'untrained_model.pickle')
    assert CustomLoader(mode='predict').load_model(input_model_file_path)


"""
Test class CustomDumper
"""


def test_dump_labels():
    # Load labels
    input_labels_file_path = os.path.join(input_test_folder_path(), TEST_LABELS)
    image_label_loader = CustomLoader(mode='predict')
    labels = image_label_loader.load_labels(input_labels_file_path)

    # Prepare output file path
    output_file_path = os.path.join(output_test_folder_path(), 'labels.gz')
    ensure_folder(output_test_folder_path())
    CustomDumper.dump_labels(labels, output_file_path)


def test_dump_load_labels():
    """
    Test if dumped labels can be loaded correctly
    """
    # Load labels
    input_labels_file_path = os.path.join(input_test_folder_path(), TEST_LABELS)
    image_label_loader = CustomLoader(mode='predict')
    input_labels = image_label_loader.load_labels(input_labels_file_path)

    # Prepare output file path
    output_file_path = os.path.join(output_test_folder_path(), 'labels.gz')
    ensure_folder(output_test_folder_path())
    CustomDumper.dump_labels(input_labels, output_file_path)

    loaded_labels = image_label_loader.load_labels(output_file_path, start_from_magic_number=False)
    assert np.array_equal(input_labels, loaded_labels)


def test_dump_model():
    cnn_model = CNNModel()
    output_file_path = os.path.join(output_test_folder_path(), 'untrained_model.pickle')
    ensure_folder(output_test_folder_path())
    CustomDumper.dump_model(cnn_model, output_file_path)


def test_dump_load_model():
    """
    Smoke test if dumped model can be loaded
    """
    # Dump model
    cnn_model = CNNModel()
    output_file_path = os.path.join(output_test_folder_path(), 'untrained_model.pickle')
    ensure_folder(output_test_folder_path())
    CustomDumper.dump_model(cnn_model.model, output_file_path)

    # Load model
    loaded_model = CustomLoader(mode='predict').load_model(output_file_path)
    assert loaded_model


"""
Test class CNNModel
"""


def test_init():
    # Smoke test
    cnn_model = CNNModel()
    assert cnn_model.model


def test_train(train_images_labels):
    # Smoke test
    cnn_model = CNNModel()

    # Set the environmental variable
    set_output_port_folder(
        output_folder=output_test_folder_path(),
        port_number=0
    )

    cnn_model.train(
        images_folder_path=input_test_folder_path(),
        labels_folder_path=input_test_folder_path()
    )
    assert cnn_model.model


def test_predict(train_images_labels, predict_images_labels):
    # Smoke test

    # Load the trained model
    model_file_path = os.path.join(input_test_folder_path(), 'trained_model.pickle')
    cnn_model = CustomLoader().load_model(model_file_path)

    # Set the environmental variable
    set_output_port_folder(
        output_folder=output_test_folder_path(),
        port_number=0
    )

    # Predict labels
    predict_labels = cnn_model.predict(
        images_folder_path=input_test_folder_path()
    )
    assert isinstance(predict_labels, np.ndarray)


def test_evaluate(train_images_labels, predict_images_labels):
    # Smoke test
    cnn_model = CNNModel()

    # Set the environmental variable
    set_output_port_folder(
        output_folder=output_test_folder_path(),
        port_number=0
    )

    evaluate_result = cnn_model.evaluate(
        predict_labels_folder_path=input_test_folder_path(),
        ground_truth_folder_path=input_test_folder_path()
    )
    print(f"Accuracy = {evaluate_result.get('Accuracy')}")
    assert evaluate_result
