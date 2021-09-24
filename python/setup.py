import numpy
import setuptools
from Cython.Build import cythonize

# True, if annotated Cython source files that highlight Python interactions should be created
ANNOTATE = False

# True, if all Cython compiler optimizations should be disabled
DEBUG = False

sources = [
    '**/*.pyx'
]

library_dirs = [
    '../cpp/build/subprojects/common',
    '../cpp/build/subprojects/tsa'
]

runtime_library_dirs = [
    'cpp/build/subprojects/common',
    'cpp/build/subprojects/tsa'
]

libraries = [
    'rlcommon',
    'rltsa'
]

include_dirs = [
    '../cpp/subprojects/common/include',
    '../cpp/subprojects/tsa/include'
]

define_macros = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
]

compiler_directives = {
    'boundscheck': DEBUG,
    'wraparound': DEBUG,
    'cdivision': not DEBUG,
    'initializedcheck': DEBUG
}

extensions = [
    setuptools.Extension(name='*', language='c++', sources=sources, library_dirs=library_dirs, libraries=libraries,
                         runtime_library_dirs=runtime_library_dirs, include_dirs=include_dirs,
                         define_macros=define_macros)
]

setuptools.setup(
    name='syndrome-learner',
    version='0.1.0',
    description='A rule learning algorithm for learning syndrome definitions',
    author='Michael Rapp',
    author_email='mrapp@ke.tu-darmstadt.de',
    license='MIT',
    packages=['rl.common', 'rl.tsa', 'rl.testbed'],
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'Cython>=0.29.0',
        'scikit-learn>=0.24.0',
        'liac-arff>=2.5.0',
        'pandas>=1.2.0'
    ],
    python_requires='>=3.7',
    ext_modules=cythonize(extensions, language_level='3', annotate=ANNOTATE, compiler_directives=compiler_directives),
    include_dirs=[numpy.get_include()])
