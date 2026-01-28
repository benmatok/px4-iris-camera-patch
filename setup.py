from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "drone_env.drone_cython",
        ["drone_env/drone_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-fopenmp", "-ffast-math", "-mavx2", "-mfma"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "drone_env.texture_features",
        ["drone_env/texture_features.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-fopenmp", "-ffast-math", "-mavx2", "-mfma"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "drone_env.tracker",
        ["drone_env/tracker.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-fopenmp", "-ffast-math", "-mavx2", "-mfma"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "ghost_dpc.ghost_dpc",
        ["ghost_dpc/ghost_dpc.pyx"],
        include_dirs=[numpy.get_include(), "ghost_dpc"],
        extra_compile_args=["-O3", "-fopenmp", "-ffast-math", "-mavx2", "-mfma"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="drone_env_cython",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
