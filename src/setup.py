from distutils.core import setup, Extension
import numpy.distutils.misc_util
import os

desc = open("README.rst").read()
required = ["numpy"]

# define the name of the extension to use
extension_name    = "carmcmcLib"
extension_version = "0.1.0"
extension_url     = "https://github.com/bckelly80/carma_pack"

BOOST_DIR         = os.environ["BOOST_DIR"]
ARMADILLO_DIR     = os.environ["ARMADILLO_DIR"]
YAMCMCPP_DIR      = os.environ["YAMCMCPP_DIR"]
NUMPY_DIR         = os.environ["NUMPY_DIR"]
include_dirs      = [NUMPY_DIR+"/include", BOOST_DIR+"/include", ARMADILLO_DIR+"/include", YAMCMCPP_DIR, "/usr/include/"]
for include_dir in numpy.distutils.misc_util.get_numpy_include_dirs():
    include_dirs.append(include_dir)
library_dirs      = [NUMPY_DIR+"/lib", BOOST_DIR+"/lib", ARMADILLO_DIR+"/lib64", YAMCMCPP_DIR+"/lib/yamcmcpp/", "/usr/lib64/", "/usr/lib/"]

# define the libraries to link with the boost python library
libraries = [ "boost_python", "boost_filesystem", "boost_system", "armadillo", ":yamcmcppLib.so"]

# define the source files for the extension
source_files = [ "boost_python_wrapper.cpp", "carmcmc.cpp", "carpack.cpp" ]
 
# create the extension and add it to the python distribution
setup( name=extension_name, 
       version=extension_version, 
       author="Brandon Kelly and Andrew Becker",
       author_email="acbecker@gmail.com",
       packages=[extension_name],
       package_dir = { "": "lib/carmcmc" },
       url=extension_url,
       description="MCMC Sampler for Performing Bayesian Inference on Continuous Time Autoregressive Models",
       long_description=desc,
       install_requires=required,
       ext_modules=[Extension( extension_name, source_files, 
                               include_dirs=include_dirs, 
                               library_dirs=library_dirs, 
                               libraries=libraries )] )
