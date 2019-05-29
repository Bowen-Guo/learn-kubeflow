import os
import argparse

from cnn_module_entry import run_evaluate
from load_data import mkdir


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate scored MNIST data")
    parser.add_argument("--input-prediction-folder-path", type=str, help="Folder path of predicted data ")
    parser.add_argument("--input-true-folder-path", type=str, help="Folder path of ground truth data")
    parser.add_argument("--output-data-folder-path", type=str, help="Folder path of evaluation result")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    print(f"input_prediction_folder_path = {args.input_prediction_folder_path}")
    print(f"input_true_folder_path = {args.input_true_folder_path}")
    print(f"output-data-folder-path = {args.output_data_folder_path}")

    os.environ.update({
        'OUTPUT_0': args.output_data_folder_path
    })
    mkdir(args.output_data_folder_path)

    run_evaluate(args.input_prediction_folder_path, args.input_true_folder_path)
