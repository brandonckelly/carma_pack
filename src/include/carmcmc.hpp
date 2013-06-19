#include <vector>
#include <samplers.hpp>

std::pair<std::vector<arma::vec>, std::vector<double> >
RunEnsembleCarSampler(int sample_size, int burnin,
                      arma::vec time, arma::vec y,
                      arma::vec yerr, int p, int nwalkers, int thin=1);
