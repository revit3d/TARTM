import io
from setuptools import setup, find_packages

from pyartm import __version__

def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


readme = read('README.rst')


setup(
    # metadata
    name='pyartm',
    version=__version__,
    license='MIT',
    author='Diyakov Ilya',
    author_email="s02210378@gse.cs.msu.ru",
    description='Fast and flexible corpus decomposition model on PyTorch',
    long_description=readme,
    url='https://github.com/Intelligent-Systems-Phystech/ProjectTemplate',

    # options
    packages=find_packages(),
)
