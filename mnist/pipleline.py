import os
import kfp

dir_path = os.path.dirname(os.path.realpath(__file__))
train_component_path = os.path.join(dir_path, 'train_component', 'load_data_component.yaml')
train_op = kfp.components.load_component_from_file(train_component_path)
score_component_path = os.path.join(dir_path, 'score_component', 'score_component.yaml')
score_op = kfp.components.load_component_from_file(score_component_path)


@kfp.dsl.pipeline(name='MNIST pipeline', description='MNIST pipeline')
def mnist_pipeline():
    train_task = train_op(
        input_data_folder_path='./input/',
        output_model_folder_path='./output/'
    )
    score_task = score_op(
        input_learner_folder_path=train_task.output,
        input_data_folder_path='./input/',
        output_data_folder_path='./output/'
    )
