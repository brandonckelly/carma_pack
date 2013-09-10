from distutils.core import setup, Extension
import numpy.distutils.misc_util
import os
import platform

system_name= platform.system()
desc = open("README.rst").read()
extension_version = "0.1.0"
extension_url = "https://github.com/bckelly80/carma_pack"
BOOST_DIR = os.environ["BOOST_DIR"]
ARMADILLO_DIR = os.environ["ARMADILLO_DIR"]
YAMCMCPP_DIR = os.environ["YAMCMCPP_DIR"]
NUMPY_DIR = os.environ["NUMPY_DIR"]
include_dirs = [YAMCMCPP_DIR + "/include", NUMPY_DIR + "/include", BOOST_DIR + "/include", ARMADILLO_DIR + "/include",
                "/usr/include/"]
for include_dir in numpy.distutils.misc_util.get_numpy_include_dirs():
    include_dirs.append(include_dir)
library_dirs = [YAMCMCPP_DIR + "/lib", NUMPY_DIR + "/lib", BOOST_DIR + "/lib", ARMADILLO_DIR + "/lib", "/usr/lib/"]
if system_name != 'Darwin':
    # /usr/lib64 does not exist under Mac OS X
    library_dirs.append("/usr/lib64")

compiler_args = ["-O3"]
if system_name == 'Darwin':
    compiler_args.append("-std=c++11")
    # need to build against libc++ for Mac OS X
    compiler_args.append("-stdlib=libc++")
else:
    compiler_args.append("-std=c++0x")


def configuration(parent_package='', top_path=None):
    # http://docs.scipy.org/doc/numpy/reference/distutils.html#numpy.distutils.misc_util.Configuration
    from numpy.distutils.misc_util import Configuration

    config = Configuration("carmcmc", parent_package, top_path)
    config.version = extension_version
    config.add_data_dir((".", "carmcmc"))
    config.add_library(
        "carmcmc",
        sources=["carmcmc.cpp", "carpack.cpp", "kfilter.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["boost_python", "boost_filesystem", "boost_system", "armadillo", "yamcmcpp"],
        extra_compiler_args=compiler_args
    )
    config.add_extension(
        "_carmcmc",
        sources=["boost_python_wrapper.cpp", "carmcmc.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["boost_python", "boost_filesystem", "boost_system", "armadillo", "yamcmcpp", "carmcmc"],
        extra_compile_args=compiler_args
    )
    config.add_data_dir(("../../../../include", "include"))
    config.add_data_dir(("../../../../examples", "examples"))
    config.test_suite = "tests/testCarmcmc"
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
