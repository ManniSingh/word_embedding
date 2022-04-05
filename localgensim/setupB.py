from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize(Extension('gensim2.models.word2vecB_inner', sources=['gensim2/models/word2vecB_inner.pyx'], language='c')),
    include_dirs=[numpy.get_include()],
)