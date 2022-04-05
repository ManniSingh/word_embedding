from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

'''
setup(
    ext_modules=cythonize("gensim.models.word2vec_inner.pyx"),
    include_dirs=[numpy.get_include()],
) 
'''

setup(
    ext_modules=cythonize(Extension('gensim2.models.word2vec_inner', sources=['gensim2/models/word2vec_inner.pyx'], language='c')),
    include_dirs=[numpy.get_include()],
)