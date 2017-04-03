from setuptools import setup
from setuptools import find_packages


setup(name='ann4brains',
    version='0.0.1',
    description='Adjacency matrices filters for Python',
    url='https://github.com/jeremykawahara/ann4brains',
    install_requires=['caffe','numpy','scipy','h5py', 'cPickle', 'matplotlib'],
	packages=find_packages())
