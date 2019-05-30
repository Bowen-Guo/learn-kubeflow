import os
from enum import Enum

import yaml


class DataType(Enum):
    STRING = 'string'
    FLOAT = 'float'
    INT = 'int'
    PATH = 'path'
    URI = 'uri'

    @classmethod
    def parse(cls, text):
        for member in cls:
            if member.value == text:
                return member
        raise ValueError(f"Unrecognized data type: '{text}'")


class Parameter:
    def __init__(self, name, data_type, default_value):
        self._name = name
        self._data_type = data_type
        self._default_value = default_value

    @classmethod
    def create(cls, name, value):
        data_type_str = None
        default_value = None
        if isinstance(value, str):
            data_type_str = value
        elif isinstance(value, dict):
            data_type_str = value.get('type')
            default_value = value.get('default')

        if not data_type_str:
            raise ValueError(f"Data type must be specified for parameter '{name}'.")

        return Parameter(name, DataType.parse(data_type_str), default_value)


class EntryPoint:
    def __init__(self, name, command, parameters):
        self._name = name
        self._command = command
        self._parameters = parameters

    @classmethod
    def from_dct(cls, name, dct: dict):
        return EntryPoint(
            name=name,
            command=dct.get('command'),
            parameters={name: Parameter.create(name, value) for name, value in dct.get('parameters').items()}
        )


class MLProject:
    def __init__(self):
        self._file = os.path.join('mlflow_projects', 'MLproject')
        with open(self._file) as f:
            self._dct = yaml.load(f)

        self._entry_points = {name: EntryPoint.from_dct(name, value)
                              for name, value in self._dct.get('entry_points').items()}

    @property
    def entry_points(self):
        return self._entry_points


if __name__ == '__main__':
    p = MLProject()
    print(p.entry_points)





