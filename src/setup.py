import numpy
from setuptools import Extension, setup
import sys

carmcmc = Extension(
    name="carmcmc._carmcmc",
    sources=["boost_python_wrapper.cpp", "carmcmc.cpp", "carpack.cpp", "kfilter.cpp",
             "proposals.cpp", "samplers.cpp", "random.cpp", "steps.cpp"],
    extra_compile_args=["-O3", "-fpermissive"],
    include_dirs=['./include', numpy.get_include()],
    libraries=[f'boost_python{str(sys.version_info[0])+str(sys.version_info[1])}', 'boost_filesystem', 'boost_system', 'boost_timer', 'armadillo'],
    library_dirs=[],
)

setup(
    name='carmcmc',
    version='0.1.0',
    packages=['carmcmc'],
    description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=[carmcmc],
)
