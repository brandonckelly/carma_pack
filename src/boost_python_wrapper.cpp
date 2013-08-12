#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#endif

#include <utility>
#include <boost/shared_ptr.hpp>
#include <boost/intrusive/options.hpp>
#include <boost/python.hpp>
#include <boost/python/call_method.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <armadillo>
#include "Python.h"
#include "numpy/ndarrayobject.h"
#include "include/carmcmc.hpp"
#include "include/carpack.hpp"

using namespace boost::python;

/*
boost::shared_ptr<CAR1> runWrapper(int sample_size, int burnin,
                                   std::vector<double> x, 
                                   std::vector<double> y, 
                                   std::vector<double> dy, 
                                   int p, int nwalkers, int thin=1) {

    // We may still need this around to cast the return object 
    boost::shared_ptr<CAR1> retObject = 
        RunEnsembleCarSampler(sample_size, burnin, x, y, dy, p, nwalkers, thin);

    return retObject;
}
*/

BOOST_PYTHON_MODULE(_carmcmc){
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    class_<std::vector<double> >("vecD")
        .def(vector_indexing_suite<std::vector<double> >());

    class_<std::vector<std::vector<double > > >("vecvecD")
        .def(vector_indexing_suite<std::vector<std::vector<double> > >());

    class_<CAR1, boost::shared_ptr<CAR1> >("CAR1", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,optional<double> >())
        .def("getLogPrior", &CAR1::getLogPrior)
        .def("getLogDensity", &CAR1::getLogDensity)
        .def("getSamples", &CAR1::getSamples)
        .def("GetLogLikes", &CAR1::GetLogLikes)
    ;

    class_<CARp, bases<CAR1>, boost::shared_ptr<CARp> >("CARp", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,int,optional<double> >())
        .def("getLogPrior", &CARp::getLogPrior)
        .def("getLogDensity", &CARp::getLogDensity)
    ;

    class_<CARMA, bases<CAR1>, boost::shared_ptr<CARMA> >("CARMA", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,int,optional<double> >())
        .def("getLogPrior", &CARMA::getLogPrior)
        .def("getLogDensity", &CARMA::getLogDensity)
    ;
    
    def("run_mcmc1", RunEnsembleCarSampler1);
    def("run_mcmcp", RunEnsembleCarSamplerp);
};
