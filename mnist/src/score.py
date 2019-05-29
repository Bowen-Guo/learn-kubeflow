import os
import argparse

from cnn_module_entry import run_score
from load_data import mkdir



def parse_arguments():
    parser = argparse.ArgumentParser(description="Score CNN with MNIST data")
    parser.add_argument("--input-learner-folder-path", type=str, help="Folder path of learner ")
    parser.add_argument("--input-data-folder-path", type=str, help="Folder path of scoring data")
    parser.add_argument("--output-data-folder-path", type=str, help="Folder path of scored data")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    print(f"input_data_folder_path = {args.input_data_folder_path}")
    print(f"input_learner_folder_path = {args.input_learner_folder_path}")
    print(f"output-data-folder-path = {args.output_data_folder_path}")

    os.environ.update({
        'OUTPUT_0': args.output_data_folder_path
    })
    mkdir(args.output_data_folder_path)

    run_score(args.input_learner_folder_path, args.input_data_folder_path)
