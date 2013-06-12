#include <vector>
#include <samplers.hpp>

std::vector<arma::vec> RunEnsembleCarSampler(MCMCOptions mcmc_options, 
                                             arma::vec time, arma::vec y,
                                             arma::vec yerr, int p, int nwalkers);
