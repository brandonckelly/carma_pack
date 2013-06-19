#include <vector>
#include "yamcmc++.hpp"

std::vector<arma::vec> RunEnsembleCarSampler(int sample_size, int burnin,
                                             arma::vec time, arma::vec y,
                                             arma::vec yerr, int p, int nwalkers, int thin=1);
