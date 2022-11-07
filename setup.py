#!/usr/bin/env python
import io
import os
import re
from datetime import datetime
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('src', 'earthformer', '__init__.py')

if VERSION.endswith('dev'):
    VERSION = VERSION + datetime.today().strftime('%Y%m%d')

requirements = [
    'absl-py',
    'boto3',
    'javalang>=0.13.0',
    'h5py>=2.10.0',
    'yacs>=0.1.8',
    'protobuf',
    'unidiff',
    'scipy',
    'tqdm',
    'regex',
    'requests',
    'jsonlines',
    'contextvars',
    'pyarrow>=3',
    'transformers>=4.3.0',
    'tensorboard',
    'pandas',
    'contextvars;python_version<"3.7"',  # Contextvars for python <= 3.6
    'dataclasses;python_version<"3.7"',  # Dataclass for python <= 3.6
    'pickle5;python_version<"3.8"',  # pickle protocol 5 for python <= 3.8
    'graphviz',
    'networkx',
    'fairscale>=0.3.0',
    'fvcore>=0.1.5',
    'pympler',
    'einops>=0.3.0',
    'timm',
    'omegaconf',
    'matplotlib',
    'awscli',
    'boto3',
    'botocore',
]

setup(
    # Metadata
    name='earthformer',
    version=VERSION,
    python_requires='>=3.6',
    description='Earthformer: Exploring Space-Time Transformers for Earth System Forecasting',
    long_description_content_type='text/markdown',
    license='Apache-2.0',

    # Package info
    packages=find_packages(where="src", exclude=(
        'tests',
        'scripts',
    )),
    package_dir={"": "src"},
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
)
