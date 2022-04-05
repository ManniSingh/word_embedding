from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("word2vec_inner.pyx"),
    include_dirs=[numpy.get_include()],
) 