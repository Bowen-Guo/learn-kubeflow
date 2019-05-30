import os
import json
import shutil


def make_folder(path):
    os.makedirs(path, exist_ok=True)


def delete_folder(path):
    shutil.rmtree(path)


def test_load_data():
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_data_path")
    os.system(f"python ../src/load_data.py --output-data-folder-path={output_path}")
    delete_folder(output_path)


def test_train():
    number_of_steps = 10
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_model_path")
    os.system(f"python ../src/train.py --input-data-folder-path={input_path}  "
              f"--number-of-steps={number_of_steps}  "
              f"--output-model-folder-path={output_path}")
    delete_folder(output_path)


def test_score():
    input_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
    input_learner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_predict_path")
    os.system(f"python ../src/score.py --input-learner-folder-path={input_learner_path}  "
              f"--input-data-folder-path={input_data_path}  --output-data-folder-path={output_path}")
    delete_folder(output_path)


def test_evaluate():
    predict_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
    true_folder_path = predict_folder_path
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_evaluation_path")
    os.system(f"python ../src/evaluate.py --input-prediction-folder-path={predict_folder_path}  "
              f"--input-true-folder-path={true_folder_path}  "
              f"--output-data-folder-path={output_path}")

    output_json_path = os.path.join(output_path, 'evaluation_result.json')
    with open(output_json_path) as f:
        eval_result = json.load(f)
    print(f"eval_result = {eval_result}")
    delete_folder(output_path)


def test_evaluate_from_aml():
    # predict_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aml_container')
    predict_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aml_container', 'output')

    true_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
    output_path = os.path.join(predict_folder_path, 'output')
    os.system(f"python ../src/evaluate.py --input-prediction-folder-path={predict_folder_path}  "
              f"--input-true-folder-path={true_folder_path}  "
              f"--output-data-folder-path={output_path}")
    output_json_path = os.path.join(output_path, 'evaluation_result.json')
    with open(output_json_path) as f:
        eval_result = json.load(f)
    print(f"eval_result = {eval_result}")


def test_score_from_aml():
    input_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
    input_learner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aml_container")
    output_path = os.path.join(input_learner_path, "output")
    os.system(f"python ../src/score.py --input-learner-folder-path={input_learner_path}  "
              f"--input-data-folder-path={input_data_path}  --output-data-folder-path={output_path}")
    # delete_folder(output_path)

