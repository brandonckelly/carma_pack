from distutils.core import setup, Extension
import numpy.distutils.misc_util
import os

desc = open("README.rst").read()
extension_version = "0.1.0"
extension_url = "https://github.com/bckelly80/carma_pack"
CFLAGS = "-O3"
BOOST_DIR = os.environ["BOOST_DIR"]
ARMADILLO_DIR = os.environ["ARMADILLO_DIR"]
YAMCMCPP_DIR = os.environ["YAMCMCPP_DIR"]
NUMPY_DIR = os.environ["NUMPY_DIR"]
include_dirs = [NUMPY_DIR + "/include", BOOST_DIR + "/include", ARMADILLO_DIR + "/include", YAMCMCPP_DIR + "/include",
                "/usr/include/"]
for include_dir in numpy.distutils.misc_util.get_numpy_include_dirs():
    include_dirs.append(include_dir)
library_dirs = [NUMPY_DIR + "/lib", BOOST_DIR + "/lib", ARMADILLO_DIR + "/lib", YAMCMCPP_DIR + "/lib", "/usr/lib64/",
                "/usr/lib/"]


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("carmcmc", parent_package, top_path)
    config.version = extension_version
    config.add_data_dir((".", "carmcmc"))
    config.add_library(
        "carmcmc",
        sources=["carmcmc.cpp", "carpack.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["boost_python", "boost_filesystem", "boost_system", "armadillo", "yamcmcpp"]
    )
    config.add_extension(
        "_carmcmc",
        sources=["boost_python_wrapper.cpp", "carmcmc.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["boost_python", "boost_filesystem", "boost_system", "armadillo", "yamcmcpp", "carmcmc"]
    )
    config.add_data_dir(("../../../../include", "include"))
    config.add_data_dir(("../../../../examples", "examples"))
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
