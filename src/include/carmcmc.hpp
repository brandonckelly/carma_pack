#include <vector>
#include <samplers.hpp>

std::pair<std::vector<arma::vec>, std::vector<double> >
RunEnsembleCarmaSampler(int sample_size, int burnin, arma::vec time, arma::vec y,
                        arma::vec yerr, int p, int q, int nwalkers, bool do_zcarma=false,
                        int thin=1);

std::pair<std::vector<arma::vec>, std::vector<double> >
RunCar1Sampler(int sample_size, int burnin, arma::vec time, arma::vec y,
               arma::vec yerr, int nwalkers, int thin);

std::pair<std::vector<arma::vec>, std::vector<double> >
RunCarmaSampler(int sample_size, int burnin, arma::vec time, arma::vec y,
                arma::vec yerr, int nwalkers, int p, int q, bool do_zcarma,
                int thin);