#include <vector>
#include <memory>
#include "carpack.hpp"

std::shared_ptr<CAR1>
RunCar1Sampler(int sample_size, int burnin, std::vector<double> time, std::vector<double> y,
               std::vector<double> yerr, int thin=1);

std::shared_ptr<CARp>
RunCarmaSampler(int sample_size, int burnin, std::vector<double> time, std::vector<double> y,
                std::vector<double> yerr, int p, int q, int nwalkers, bool do_zcarma=false,
                int thin=1);
