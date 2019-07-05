import os

from tools.amlservice_scaffold.amlservice_pipeline import Module, PipelineStep, run_pipeline
from azureml.core import RunConfiguration, Workspace
from azureml.core.environment import DEFAULT_GPU_IMAGE

MODULE_SPECS_FOLDER = 'module_specs'


def spec_file_path(spec_file_name):
    return os.path.join(MODULE_SPECS_FOLDER, spec_file_name)


def get_workspace(name, subscription_id, resource_group):
    return Workspace.get(
        name=name,
        subscription_id=subscription_id,
        resource_group=resource_group
    )


def get_run_config(comp, compute_name, use_gpu=False):
    if comp.image:
        run_config = RunConfiguration()
        run_config.environment.docker.base_image = comp.image
    else:
        run_config = RunConfiguration(conda_dependencies=comp.conda_dependencies)
    run_config.target = compute_name
    run_config.environment.docker.enabled = True
    if use_gpu:
        run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
        run_config.environment.docker.gpu_support = True

    return run_config


def create_pipeline_steps(compute_name):
    # Load module spec from yaml file
    import_train_data = Module(
        spec_file_path=spec_file_path('import_data.yaml'),
        source_directory='script'
    )

    import_test_data = Module(
        spec_file_path=spec_file_path('import_data.yaml'),
        source_directory='script'
    )

    preprocess_train_data = Module(
        spec_file_path=spec_file_path('preprocess.yaml'),
        source_directory='script'
    )

    preprocess_test_data = Module(
        spec_file_path=spec_file_path('preprocess.yaml'),
        source_directory='script'
    )

    train = Module(
        spec_file_path=spec_file_path('train.yaml'),
        source_directory='script',
    )

    score = Module(
        spec_file_path=spec_file_path('score.yaml'),
        source_directory='script',
    )

    # Run config setting
    run_config_import_train_data = get_run_config(import_train_data, compute_name)
    run_config_import_test_data = get_run_config(import_test_data, compute_name)
    run_config_preprocess_train_data = get_run_config(preprocess_train_data, compute_name)
    run_config_preprocess_test_data = get_run_config(preprocess_test_data, compute_name)
    run_config_train = get_run_config(train, compute_name, use_gpu=True)
    run_config_score = get_run_config(score, compute_name, use_gpu=True)

    # Assign parameters
    import_train_data.params['command json'].assign('import_train_data.json')
    import_test_data.params['command json'].assign('import_test_data.json')

    preprocess_train_data.params['BERT pretrained model'].assign('bert-base-cased')
    preprocess_train_data.params['Maximum sequence length'].assign(128)

    preprocess_test_data.params['BERT pretrained model'].assign('bert-base-cased')
    preprocess_test_data.params['Maximum sequence length'].assign(128)

    train.params['BERT pretrained model'].assign("bert-base-cased")
    train.params['Maximum sequence length'].assign(128)
    train.params['Number of training epochs'].assign(1)
    train.params['Warmup proportion'].assign(0.4)

    # Connect ports
    preprocess_train_data.inputs['Input data'].connect(
        import_train_data.outputs['9e5eade8_fa58_4a3b_9c51_d9b3f704b756'])
    preprocess_test_data.inputs['Input data'].connect(
        import_test_data.outputs['9e5eade8_fa58_4a3b_9c51_d9b3f704b756'])
    train.inputs['Input train data'].connect(preprocess_train_data.outputs['Output feature'])
    score.inputs['Input test data'].connect(preprocess_test_data.outputs['Output feature'])
    score.inputs['Trained model'].connect(train.outputs['Output model'])

    # Convert to a list of PipelineStep, which can be ran by AML Service
    pipeline_step_list = [
        PipelineStep(import_train_data, run_config=run_config_import_train_data),
        PipelineStep(import_test_data, run_config=run_config_import_test_data),
        PipelineStep(preprocess_train_data, run_config=run_config_preprocess_train_data),
        PipelineStep(preprocess_test_data, run_config=run_config_preprocess_test_data),
        PipelineStep(train, run_config=run_config_train),
        PipelineStep(score, run_config=run_config_score)
    ]
    return pipeline_step_list


if __name__ == '__main__':
    workspace = get_workspace(
        name="chjinche-test-service",
        subscription_id="e9b2ec51-5c94-4fa8-809a-dc1e695e4896",
        resource_group="chjinche"
    )
    compute_name = 'gpu-compute0'
    pipeline_steps = create_pipeline_steps(compute_name)
    run_pipeline(steps=pipeline_steps, experiment_name='NER-BERT-test', workspace=workspace)
