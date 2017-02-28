from glob import glob

from setuptools import Extension
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from Cython.Distutils import build_ext

import os
import sys


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


def compile_cython(use_cython=True):
    source_root = os.path.abspath(os.path.dirname(__file__))
    cython_extensions = [
        'algorithms.sorting._sorting',
    ]

    extensions = []

    for ext in cython_extensions:
        source_file = os.path.join(source_root, *ext.split('.'))
        source_dir = os.path.dirname(source_file)

        depends = []

        pxd_source = source_file + '.pxd'
        pyx_source = source_file + '.pyx'

        if os.path.exists(pxd_source):
            depends.append(pxd_source)

        # handles src/*.cpp
        depends.extend(glob(os.path.join(source_dir, 'src', '*.c*')))

        extensions.append(
            Extension(ext, sources=[pyx_source], depends=depends,
                      language='c++'))

    return extensions


requirements = [
    'numpy==1.12.0',
    'scipy==0.18.1',
    'matplotlib==2.0.0'
]

test_requirements = [
    'pytest==3.0.6',
    'pytest-cov==2.4.0',
    'pytest-mock==1.5.0',
]

setup(
    name='algorithm',
    version='0.1',
    packages=find_packages(exclude=['test']),
    url='https://github.com/vadimadr/python-algorithms.py',
    license='MIT',
    author='Vadim Andronov', author_email='vadimadr@gmail.com',
    description='Implementation of some common algorithms',
    ext_modules=compile_cython(),
    zip_safe=False,
    package_data={'': 'LICENSE'},
    install_requires=requirements,
    setup_requires=[
        'cython',
    ],
    tests_require=test_requirements,
    test_suite='tests',
    cmdclass={'test': PyTest, 'build_ext': build_ext},
)
