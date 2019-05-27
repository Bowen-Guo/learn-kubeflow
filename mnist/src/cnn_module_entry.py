"""
Interface to invoke methods of custom module, which is CNN in this case.
This module is provided by developers, not users.
"""
import os
from cnn_module import CNNModel, CustomLoader


class FolderWrapper:
    """
    Record the folder path of a port
    """
    def __init__(self, folder_path):
        self._folder_path = folder_path

    @property
    def folder_path(self):
        if not os.path.isdir(self._folder_path):
            raise FileNotFoundError(f"Folder path {self._folder_path} is not found")
        return self._folder_path


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


def run_evaluate(scored_folder_wrapper, ground_truth_folder_wrapper):
    """
    Evaluate a custom model

    :param scored_folder_wrapper: FolderWrapper instance
    """
    # Initialize custom-defined learner instance
    learner = CNNModel()
    learner.evaluate(
        predict_labels_folder_path=scored_folder_wrapper.folder_path,
        ground_truth_folder_path=ground_truth_folder_wrapper.folder_path
    )
