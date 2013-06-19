#include <vector>
#include <samplers.hpp>

std::pair<std::vector<arma::vec>, std::vector<double> >
RunEnsembleCarSampler(MCMCOptions mcmc_options, 
                      arma::vec time, arma::vec y,
                      arma::vec yerr, int p, int nwalkers);
