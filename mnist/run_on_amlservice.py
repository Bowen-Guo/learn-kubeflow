import os
import yaml

from azureml.core import Experiment, RunConfiguration, Workspace
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import PythonScriptStep


class Port:
    def __init__(self, name, type, description):
        self.name = name
        self.type = type
        self.description = description
        self.value = None  # to be set by code.
        self._connected = False
        self._assigned = False

    @staticmethod
    def from_dct(dct: dict):
        return Port(
            name=dct.get('name'),
            type=dct.get('type'),
            description=dct.get('description'),
        )

    def prepare(self):
        def _regular_name(port_name):
            # AML Service does not allow name with spaces. Replace them with underscore.
            return '_'.join(port_name.split())

        if not self.prepared:
            conn = PipelineData(_regular_name(self.name))
            self.value = conn

    @property
    def prepared(self):
        return self.value is not None

    @property
    def connected(self):
        return self._connected

    def connect(self, another_port):
        if not another_port:
            raise ValueError(f"Cannot connect to a empty port")
        if not another_port.prepared:
            raise ValueError(f"Port({another_port.name}) is not ready yet")

        self.value = another_port.value
        self._connected = True

    def assign(self, value):
        self.value = value
        self._assigned = True


class KubeflowComponent:
    def __init__(self, name):
        self._name = name
        self._file = os.path.join('kubeflow_components', name, f'{name}_component.yaml')
        with open(self._file) as f:
            self._dct = yaml.load(f)

        self._input_ports = [Port.from_dct(p) for p in self._get_value('inputs')]
        self._output_ports = [Port.from_dct(p) for p in self._get_value('outputs')]

        for p in self._output_ports:
            p.prepare()

    def _get_value(self, key_path):
        if not key_path:
            raise ValueError("key_path must not be empty")
        if not self._dct:
            raise ValueError("dct is empty")

        segments = key_path.split('/')

        walked = []

        cur_obj = self._dct
        for seg in segments:
            if cur_obj is None:
                raise ValueError(f"Missing {'/'.join(walked)} block in yaml file: {self._file}")
            if not isinstance(cur_obj, dict):
                raise ValueError(f"Block {'/'.join(walked)} cannot contain a child. "
                                 f"Confirm yaml file: {self._file}")

            cur_obj = cur_obj.get(seg)
            walked.append(seg)

        if cur_obj is None:
            raise ValueError(f"Missing {'/'.join(walked)} block in yaml file: {self._file}")
        return cur_obj

    @property
    def name(self):
        return self._get_value('name')

    @property
    def description(self):
        return self._get_value('description')

    @property
    def inputs(self):
        return {p.name: p for p in self._input_ports}

    @property
    def input_refs(self):
        return [p.value for p in self._input_ports if p.connected]

    @property
    def params(self):
        disconnected_inputs = [p for p in self._input_ports if not p.connected]
        return {p.name: p for p in disconnected_inputs}

    @property
    def outputs(self):
        return {p.name: p for p in self._output_ports}

    @property
    def output_refs(self):
        return [p.value for p in self._output_ports]

    @property
    def image(self):
        return self._get_value('implementation/container/image')

    @property
    def command(self):
        return self._get_value('implementation/container/command')

    @property
    def args(self):
        def handle_placeholder(value):
            if isinstance(value, str):
                return value
            elif isinstance(value, dict):
                input_port_name = value.get('inputValue')
                if input_port_name:
                    port = self.inputs.get(input_port_name)
                    if not port:
                        raise ValueError(f"Input port '{input_port_name}' not defined.")
                    return port.value

                output_port_name = value.get('outputPath')
                if output_port_name:
                    port = self.outputs.get(output_port_name)
                    if not port:
                        raise ValueError(f"Output port '{output_port_name}' not defined.")
                    return port.value

                raise ValueError(f"'inputValue' or 'outputPath' must be specified in placeholder: {value}")
            else:
                raise ValueError(f"Incorrect type for arg {value}")

        raw_args = self._get_value('implementation/container/args')
        return list(map(handle_placeholder, raw_args))

    @property
    def command_and_args(self):
        return self.command + self.args


class KubeflowComponentStep(PythonScriptStep):
    def __init__(self, kubeflow_component):
        self._comp = kubeflow_component

        run_config = RunConfiguration()
        run_config.target = 'zhizhu-compute'
        run_config.environment.docker.enabled = True
        run_config.environment.docker.base_image = self._comp.image

        print(f"== Creating KubeflowComponentStep: name={self._comp.name}\n"
              f"   arguments={self._comp.command_and_args}\n"
              f"   inputs={self._comp.input_refs}\n"
              f"   outputs={self._comp.output_refs}\n"
              )

        super().__init__(
            name=self._comp.name,
            source_directory='script',
            script_name='invoker.py',
            arguments=self._comp.command_and_args,
            inputs=self._comp.input_refs,
            outputs=self._comp.output_refs,
            compute_target='zhizhu-compute',
            allow_reuse=True,
            runconfig=run_config
        )


def create_pipeline_steps():
    # Load Kubeflow components from yaml file
    load_data = KubeflowComponent('load_data')
    train = KubeflowComponent('train')
    score = KubeflowComponent('score')
    evaluate = KubeflowComponent('evaluate')

    # Assign parameters
    train.params['Number of steps'].assign(10)

    # Connect ports
    train.inputs['MNIST data'].connect(load_data.outputs['MNIST data'])
    score.inputs['Trained learner'].connect(train.outputs['Trained model'])
    score.inputs['MNIST data'].connect(load_data.outputs['MNIST data'])
    evaluate.inputs['Scored data'].connect(score.outputs['Scored data'])
    evaluate.inputs['MNIST labels'].connect(load_data.outputs['MNIST data'])

    # Convert to a list of PipelineStep, which can be ran by AML Service
    return [KubeflowComponentStep(c) for c in [load_data, train, score, evaluate]]


def run_pipeline(steps):
    workspace = Workspace.get(
        name='zhizhu-test-ws',
        subscription_id='e9b2ec51-5c94-4fa8-809a-dc1e695e4896',
        resource_group='zhizhu-test-ws-rg'
    )

    exp = Experiment(workspace=workspace, name='Kubeflow_AMLService')

    pipeline = Pipeline(workspace=workspace, steps=steps)
    pipeline.validate()

    run = exp.submit(pipeline)
    run.wait_for_completion(show_output=True)
    run.get_metrics()


if __name__ == '__main__':
    p = create_pipeline_steps()
    run_pipeline(p)
