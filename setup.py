from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "drone_env.drone_cython",
        ["drone_env/drone_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="drone_env_cython",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
