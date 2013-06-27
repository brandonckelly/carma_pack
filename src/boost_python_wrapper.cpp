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

arma::vec numericToArma(numeric::array na) {
    int npts = extract<int>(na.attr("shape")[0]);
    arma::vec aa(npts);
    for (int i = 0; i < npts; i++) {
        aa[i] = extract<double>(na[i]);
    }
    return aa;
}

/* No sanity checks */
/*
arma::vec numpyToArma2(PyObject* input) {
    PyArrayObject* array = obj_to_array_no_conversion(input, NPY_DOUBLE);
    unsigned p = array->dimensions[0];
    *m = new arma::vec( (typename arma::vec::elem_type *)array_data(array), p, false, false );
    return m;
}
*/


numeric::array armaToNumeric2(std::vector<arma::vec> aa) {
    int nx = aa.size();
    int ny = aa[0].n_elem;
    int dimens[2] = {nx, ny};
    object na(handle<>(PyArray_FromDims(2, dimens, PyArray_DOUBLE)));
    for (int i = 0; i < dimens[0]; i++) {
        for (int j = 0; j < dimens[1]; j++) {
            na[i][j] = aa[i][j];
        }
    }
    return extract<numeric::array>(na);
}

boost::shared_ptr<CAR1> runWrapper(int sample_size, int burnin,
                                   numeric::array x, 
                                   numeric::array y, 
                                   numeric::array dy, 
                                   int p, int nwalkers, int thin=1) {
    
    arma::vec ax(numericToArma(x));
    arma::vec ay(numericToArma(y));
    arma::vec ady(numericToArma(dy));

    boost::shared_ptr<CAR1> retObject = 
        RunEnsembleCarSampler(sample_size, burnin, ax, ay, ady, p, nwalkers, thin);

    return retObject;
}

struct CAR1Wrap : CAR1, wrapper<CAR1>
{
    // Constructors storing initial self parameter
    CAR1Wrap(PyObject *p, bool track, std::string name, 
             numeric::array time, numeric::array y, numeric::array yerr, 
             double temperature=1.0):
        CAR1(track, name, numericToArma(time), numericToArma(y), numericToArma(yerr), temperature), self(p) {}
    
    // In case its returned by-value from a wrapped function
    CAR1Wrap(PyObject *p, const CAR1& x)
        : CAR1(x), self(p) {}

    double LogPrior(numeric::array car1_value) {
        arma::vec car1_convert(numericToArma(car1_value));
        return call_method<double>(self, "LogPrior", car1_convert);
    }

    double LogPrior2(numeric::array car1_value) {
        arma::vec car1_convert(numericToArma(car1_value));
        return CAR1::LogPrior(car1_convert);
    }

    double LogDensity(numeric::array car1_value) {
        return CAR1::LogDensity(numericToArma(car1_value));
    }

    numeric::array GetSamples() {
        return armaToNumeric2(CAR1::GetSamples());
    }


private:
    PyObject* self;
};

struct CARpWrap : CARp, wrapper<CARp>
{
    // Constructors storing initial self parameter
    CARpWrap(PyObject *p, bool track, std::string name, 
             numeric::array time, numeric::array y, numeric::array yerr, 
             int order, double temperature=1.0):
        CARp(track, name, numericToArma(time), numericToArma(y), numericToArma(yerr), order, temperature), self(p) {}
    
    // In case its returned by-value from a wrapped function
    CARpWrap(PyObject *p, const CARp& x)
        : CARp(x), self(p) {}

    double LogPrior(numeric::array theta) {
        return CARp::LogPrior(numericToArma(theta));
    }

    double LogDensity(numeric::array theta) {
        return CARp::LogDensity(numericToArma(theta));
    }

    numeric::array GetSamples() {
        return armaToNumeric2(CARp::GetSamples());
    }

private:
    PyObject* self;
};

struct CARMAWrap : CARMA, wrapper<CARMA>
{
    // Constructors storing initial self parameter
    CARMAWrap(PyObject *p, bool track, std::string name, 
              numeric::array time, numeric::array y, numeric::array yerr, 
              int order, double temperature=1.0):
        CARMA(track, name, numericToArma(time), numericToArma(y), numericToArma(yerr), order, temperature), self(p) {}
    
    // In case its returned by-value from a wrapped function
    CARMAWrap(PyObject *p, const CARMA& x)
        : CARMA(x), self(p) {}

    double LogPrior(numeric::array car_value) {
        return CARMA::LogPrior(numericToArma(car_value));
    }

    double LogDensity(numeric::array car_value) {
        return CARMA::LogDensity(numericToArma(car_value));
    }

    numeric::array GetSamples() {
        return armaToNumeric2(CARMA::GetSamples());
    }

private:
    PyObject* self;
};

// NOT WORKING
//http://stackoverflow.com/questions/10680691/why-do-i-need-comparison-operators-in-boost-python-vector-indexing-suite
/*
namespace indexing {
template<>
struct value_traits<arma::vec> : public value_traits<int>
{
    static bool const equality_comparable = false;
    static bool const lessthan_comparable = false;
};
*/


BOOST_PYTHON_MODULE(_carmcmc){
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    class_<std::vector<double> >("vecD")
        .def(vector_indexing_suite<std::vector<double> >());

    class_<std::vector<std::vector<double > > >("vecvecD")
        .def(vector_indexing_suite<std::vector<std::vector<double> > >());

    def("run_mcmc", runWrapper);

    // carpack.hpp
    class_<CAR1, CAR1Wrap>("CAR1", no_init)
        .def(init<bool,std::string,numeric::array,numeric::array,numeric::array,optional<double> >())
        .def("LogPrior", &CAR1Wrap::LogPrior)
        .def("LogDensity", &CAR1Wrap::LogDensity)
        .def("GetSamples", &CAR1Wrap::GetSamples)
        .def("GetLogLikes", &CAR1::GetLogLikes)
    ;

    class_<CARp, CARpWrap>("CARp", no_init)
        .def(init<bool,std::string,numeric::array,numeric::array,numeric::array,int,optional<double> >())
        .def("LogPrior", &CARpWrap::LogPrior)
        .def("LogDensity", &CARpWrap::LogDensity)
        .def("GetSamples", &CARpWrap::GetSamples)
        .def("GetLogLikes", &CARp::GetLogLikes)
    ;

    class_<CARMA, CARMAWrap>("CARMA", no_init)
        .def(init<bool,std::string,numeric::array,numeric::array,numeric::array,int,optional<double> >())
        .def("LogPrior", &CARMAWrap::LogPrior)
        .def("LogDensity", &CARMAWrap::LogDensity)
        .def("GetSamples", &CARMAWrap::GetSamples)
        .def("GetLogLikes", &CARMA::GetLogLikes)
    ;

    register_ptr_to_python< boost::shared_ptr<CAR1> >();
    register_ptr_to_python< boost::shared_ptr<CARp> >();
    register_ptr_to_python< boost::shared_ptr<CARMA> >();

};
