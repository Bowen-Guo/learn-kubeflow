"""
Interface to invoke methods of custom module, which is CNN in this case.
This module is provided by developers, not users.
"""


import os
from cnn_module import CNNModel, CustomLoader


# class FolderWrapper:
#     """
#     Record the folder path of a port
#     """
#     def __init__(self, folder_path):
#         self._folder_path = folder_path
#
#     @property
#     def folder_path(self):
#         if not os.path.isdir(self._folder_path):
#             raise FileNotFoundError(f"Folder path {self._folder_path} is not found")
#         return self._folder_path


def run_load_data():
    """
    Load training data
    """
    output_folder_path = os.environ.get('OUTPUT_0')
    if output_folder_path is None:
        raise ValueError('Environmental variable OUTPUT_0 is not defined')

    # Download MNIST data to output_folder_path
    CustomLoader().load_images_labels(output_folder_path)


def run_train(data_folder):
    """
    Train a custom model

    :param data_folder: str, local folder path of train_component images and labels
    """
    # Initialize custom-defined learner instance
    learner = CNNModel()
    learner.train(
        data_folder_path=data_folder
    )


def run_score(learner_folder, data_folder):
    """
    Score a custom model

    :param learner_folder: folder path of learner
    :param data_folder: folder path of data
    """
    # Load custom-defined learner by custom-provided loading interface
    learner = CustomLoader().load_model(learner_folder)

    learner.predict(
        data_folder_path=data_folder
    )


def run_evaluate(scored_data_folder, true_data_folder):
    """
    Evaluate a custom model

    :param scored_data_folder: str, folder path of scored data
    :param true_data_folder: str, folder path of true data
    """
    # Initialize custom-defined learner instance
    learner = CNNModel()
    learner.evaluate(
        predict_labels_folder_path=scored_data_folder,
        ground_truth_folder_path=true_data_folder
    )
