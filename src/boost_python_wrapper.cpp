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
#include "include/kfilter.hpp"

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

    /*
    class_<CARMA_Base<double> >("CARMA_Base_double", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,optional<double> >())
        .def("getLogDensity", &CARMA_Base<double>::getLogDensity)
        .def("getLogPrior", &CARMA_Base<double>::getLogPrior)
        .def("getSamples", &CARMA_Base<double>::getSamples)
    ;
    */

    // carpack.hpp
    class_<CARMA_Base<double>, boost::noncopyable>("CARMA_Base_double", no_init);
    class_<CARMA_Base<arma::vec>, boost::noncopyable>("CARMA_Base_arma", no_init);

    class_<CAR1, bases<CARMA_Base<double> >, boost::shared_ptr<CAR1> >("CAR1", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,optional<double> >())
        .def("getLogPrior", &CAR1::getLogPrior)
        .def("getLogDensity", &CAR1::getLogDensity)
        .def("getSamples", &CAR1::getSamples)
    ;

    class_<CARp, bases<CARMA_Base<arma::vec> >, boost::shared_ptr<CARp> >("CARp", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,int,optional<double> >())
        .def("getLogPrior", &CARp::getLogPrior)
        .def("getLogDensity", &CARp::getLogDensity)
        .def("getSamples", &CARp::getSamples)
    ;

    class_<CARMA, bases<CARp>, boost::shared_ptr<CARMA> >("CARMA", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,int,int,optional<double> >())
        .def("getLogPrior", &CARMA::getLogPrior)
        .def("getLogDensity", &CARMA::getLogDensity)
        .def("getSamples", &CARMA::getSamples)
    ;

    class_<ZCARMA, bases<CARp>, boost::shared_ptr<ZCARMA> >("ZCARMA", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,int,optional<double> >())
        .def("getLogPrior", &ZCARMA::getLogPrior)
        .def("getLogDensity", &ZCARMA::getLogDensity)
        .def("getSamples", &ZCARMA::getSamples)
    ;
 
    // carmcmc.hpp
    def("run_mcmc_car1", RunCar1Sampler);
    def("run_mcmc_carma", RunCarmaSampler);

    // kfilter.hpp
};
