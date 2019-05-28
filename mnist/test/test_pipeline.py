import os
import shutil

from cnn_module_entry import run_train, run_score


def make_folder(path):
    os.makedirs(path, exist_ok=True)


def delete_folder(path):
    shutil.rmtree(path)


def test_train(delete_dir=True):
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_train")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_path")
    make_folder(input_path)
    make_folder(output_path)

    os.system(f"python ../src/train.py --input-data-folder-path={input_path} --output-model-folder-path={output_path}")

    if not delete_dir:
        return input_path, output_path

    delete_folder(output_path)
    delete_folder(input_path)


def test_score(delete_dir=True):
    input_data_path, input_learner_path = test_train(delete_dir=False)
    output_path = input_data_path

    os.system(f"python ../src/score.py --input-learner-folder-path={input_learner_path}  "
              f"--input-data-folder-path={input_data_path}  --output-data-folder-path={output_path}")

    if not delete_dir:
        return output_path

    delete_folder(input_learner_path)
    delete_folder(output_path)


def test_evaluate():
    pass
