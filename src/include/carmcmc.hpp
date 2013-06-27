#include <vector>
#include "carpack.hpp"
#include "boost/shared_ptr.hpp"

boost::shared_ptr<CAR1>
RunEnsembleCarSampler1(int sample_size, int burnin,
                       std::vector<double> time, std::vector<double> y,
                       std::vector<double> yerr, int p, int nwalkers, int thin=1);

boost::shared_ptr<CARp>
RunEnsembleCarSamplerp(int sample_size, int burnin,
                       std::vector<double> time, std::vector<double> y,
                       std::vector<double> yerr, int p, int nwalkers, int thin=1);
