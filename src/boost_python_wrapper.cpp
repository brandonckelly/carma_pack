#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#endif

#include "Python.h"
#include <boost/intrusive/options.hpp>
#include <boost/python.hpp>
#include <boost/python/object.hpp>
#include <boost/python/object/function_object.hpp>
#include <boost/python/object/py_function.hpp>
#include <boost/python/call_method.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <armadillo>
#include <utility>
#include "numpy/ndarrayobject.h"
#include "include/carmcmc.hpp"
#include "include/carpack.hpp"
#include "include/kfilter.hpp"

using namespace boost::python;

BOOST_PYTHON_FUNCTION_OVERLOADS(car1Overloads, RunCar1Sampler, 5, 7);
BOOST_PYTHON_FUNCTION_OVERLOADS(carmaOverloads, RunCarmaSampler, 8, 11);

BOOST_PYTHON_MODULE(_carmcmc){
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    class_<std::vector<double> >("vecD")
        .def(vector_indexing_suite<std::vector<double> >());

    class_<std::vector<std::vector<double > > >("vecvecD")
        .def(vector_indexing_suite<std::vector<std::vector<double> > >());

    class_<std::vector<std::complex<double> > >("vecC")
        .def(vector_indexing_suite<std::vector<std::complex<double> > >());
    
    class_<std::pair<double, double> >("pairD")
        .def_readwrite("first", &std::pair<double, double>::first)
        .def_readwrite("second", &std::pair<double, double>::second);

    // carpack.hpp
    class_<CARMA_Base<double>, boost::noncopyable>("CARMA_Base_double", no_init);
    class_<CARMA_Base<arma::vec>, boost::noncopyable>("CARMA_Base_arma", no_init);

    class_<CAR1, bases<CARMA_Base<double> >, std::shared_ptr<CAR1> >("CAR1", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,optional<double> >())
        .def("getLogPrior", &CAR1::getLogPrior)
        .def("getLogDensity", &CAR1::getLogDensity)
        .def("getSamples", &CAR1::getSamples)
        .def("GetLogLikes", &CAR1::GetLogLikes)  // Base class parameters.hpp
    ;

    class_<CARp, bases<CARMA_Base<arma::vec> >, std::shared_ptr<CARp> >("CARp", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,int,optional<double> >())
        .def("getLogPrior", &CARp::getLogPrior)
        .def("getLogDensity", &CARp::getLogDensity)
        .def("getSamples", &CARp::getSamples)
        .def("GetLogLikes", &CARp::GetLogLikes)
        .def("SetMLE", &CARp::SetMLE)
    ;

    class_<CARMA, bases<CARp>, std::shared_ptr<CARMA> >("CARMA", no_init)
        .def(init<bool,std::string,std::vector<double>,std::vector<double>,std::vector<double>,int,int,optional<double> >())
        .def("getLogPrior", &CARMA::getLogPrior)
        .def("getLogDensity", &CARMA::getLogDensity)
        .def("getSamples", &CARMA::getSamples)
        .def("GetLogLikes", &CARMA::GetLogLikes)
        .def("SetMLE", &CARMA::SetMLE)
    ;

    // carmcmc.hpp
    def("run_mcmc_car1", RunCar1Sampler, car1Overloads());
    def("run_mcmc_carma", RunCarmaSampler, carmaOverloads());

    // kfilter.hpp
    class_<KalmanFilter<double>, boost::noncopyable>("KalmanFilter_double", no_init);
    class_<KalmanFilter<arma::cx_vec>, boost::noncopyable>("KalmanFilter_arma", no_init);

    class_<KalmanFilter1, bases<KalmanFilter<double> >, std::shared_ptr<KalmanFilter1> >("KalmanFilter1", no_init)
        .def(init<std::vector<double>,std::vector<double>,std::vector<double> >())
        .def(init<std::vector<double>,std::vector<double>,std::vector<double>,double,double>())
        .def("Simulate", &KalmanFilter1::Simulate)
        .def("Filter", &KalmanFilter1::Filter)
        .def("Predict", &KalmanFilter1::Predict)
        .def("GetMean", &KalmanFilter1::GetMeanSvec)
        .def("GetVar", &KalmanFilter1::GetVarSvec)
    ;
    class_<KalmanFilterp, bases<KalmanFilter<arma::cx_vec> >, std::shared_ptr<KalmanFilterp> >("KalmanFilterp", no_init)
        .def(init<std::vector<double>,std::vector<double>,std::vector<double> >())
        .def(init<std::vector<double>,std::vector<double>,std::vector<double>,double,
             std::vector<std::complex<double> >,std::vector<double> >())
        .def("Simulate", &KalmanFilterp::Simulate)
        .def("Filter", &KalmanFilterp::Filter)
        .def("Predict", &KalmanFilterp::Predict)
        .def("GetMean", &KalmanFilterp::GetMeanSvec)
        .def("GetVar", &KalmanFilterp::GetVarSvec)
    ;
};
