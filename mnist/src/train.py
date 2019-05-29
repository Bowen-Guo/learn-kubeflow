import os
import argparse

from cnn_module_entry import run_train
from load_data import mkdir


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train CNN with MNIST data")
    parser.add_argument("--input-data-folder-path", type=str, help="Folder path of training data")
    parser.add_argument("--output-model-folder-path", type=str, help="Folder path of output model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    print(f"input_data_folder_path = {args.input_data_folder_path}")
    print(f"output_model_folder_path = {args.output_model_folder_path}")

    os.environ.update({
        'OUTPUT_0': args.output_model_folder_path
    })
    mkdir(args.output_model_folder_path)

    run_train(args.input_data_folder_path)
