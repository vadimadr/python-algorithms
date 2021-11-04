import sys

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


with open("requirements-dev.txt", "r") as f:
    dev_requirements = f.read().strip().splitlines()

dev_requirements = [
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "hypothesis",
]

setup(
    name="algorithm",
    version="0.1",
    packages=find_packages(exclude=["test"]),
    url="https://github.com/vadimadr/python-algorithms.py",
    license="MIT",
    author="Vadim Andronov",
    author_email="vadimadr@gmail.com",
    description="Implementation of some common algorithms",
    zip_safe=False,
    tests_require=dev_requirements,
    test_suite="tests",
    cmdclass={"test": PyTest},
)
