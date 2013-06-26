#include <vector>
#include <parameters.hpp>
#include "boost/shared_ptr.hpp"

//std::pair<std::vector<arma::vec>, std::vector<double> >
boost::shared_ptr<Parameter<arma::vec> >
RunEnsembleCarSampler(int sample_size, int burnin,
                      arma::vec time, arma::vec y,
                      arma::vec yerr, int p, int nwalkers, int thin=1);
