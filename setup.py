from setuptools import Extension
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

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


package_data = {'': 'LICENSE'}

# Change if you want to compile *.pyx sources to *.cpp
USE_CYTHON = bool(os.environ.get('USE_CYTHON', True))


def compile_cython(use_cython=True):
    source_root = os.path.abspath(os.path.dirname(__file__))
    cython_extensions = [
        'algorithms.sorting._sorting',
    ]

    extensions = []

    for ext in cython_extensions:
        source_file = os.path.join(source_root, *ext.split('.'))
        depends = []
        ext_package = '.'.join(ext.split('.')[:-1])

        if use_cython:
            ext_data = ['*.pyx', '*.cc', '*.cpp', '*.hpp']
            pxd_source = source_file + '.pxd'
            pyx_source = source_file + '.pyx'
            if os.path.exists(pxd_source):
                depends.append(pxd_source)
                ext_data.append('*.pxd')
        else:
            ext_data = []
            extensions.append(source_file + '.cpp')
            pyx_source = source_file + '.cpp'

        extensions.append(
            Extension(ext, sources=[pyx_source], depends=depends,
                      language='c++'))

        package_data[ext_package] = ext_data

    if use_cython:
        from Cython.Build import cythonize
        return cythonize(extensions)
    else:
        return extensions


extensions = compile_cython()

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
    ext_modules=extensions,
    zip_safe=False,
    package_data=package_data,
    install_requires=requirements,
    setup_requires=[
        'cython',
    ],
    tests_require=test_requirements,
    test_suite='tests',
    cmdclass={'test': PyTest},
)
