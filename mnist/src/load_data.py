import os
import argparse

from cnn_module_entry import run_load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Load MNIST training data")
    parser.add_argument("--output-data-folder-path", type=str, help="Folder path of training data")
    return parser.parse_args()


def mkdir(dir):
    """
    create directory
    :param dir: str, directory path
    """
    os.makedirs(dir, exist_ok=True)


if __name__ == '__main__':
    args = parse_arguments()
    print(f"output_data_folder_path = {args.output_data_folder_path}")

    os.environ.update({
        'OUTPUT_0': args.output_data_folder_path
    })
    mkdir(args.output_data_folder_path)
    run_load_data()
