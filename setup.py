import os
from setuptools import Extension
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

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


# Change if you want to compile *.pyx sources to *.cpp
USE_CYTHON = False

extensions = [Extension('algorithms.sorting._sorting',
                        ['algorithms/sorting/_sorting.pyx'], language='c++')]


def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)
else:
    extensions = no_cythonize(extensions)

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
    package_data={'': ['LICENSE']},
    install_requires=requirements,
    tests_require=test_requirements,
    test_suite='tests',
    cmdclass={'test': PyTest},
)
