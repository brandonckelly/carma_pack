#include <vector>
#include "carpack.hpp"
#include "boost/shared_ptr.hpp"

boost::shared_ptr<CAR1>
RunEnsembleCarSampler(int sample_size, int burnin,
                      arma::vec time, arma::vec y,
                      arma::vec yerr, int p, int nwalkers, int thin=1);
