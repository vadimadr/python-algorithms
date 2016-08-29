import os
from setuptools import Extension
from setuptools import setup, find_packages

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

setup(name='algorithm', version='0.1', packages=find_packages(exclude=['test']),
      url='https://github.com/vadimadr/python-algorithms.py', license='MIT',
      author='Vadim Andronov', author_email='vadimadr@gmail.com',
      description='Implementation of some common algorithms',
      ext_modules=extensions)
