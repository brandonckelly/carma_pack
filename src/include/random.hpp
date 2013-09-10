//
//  random.h
//  yamcmc++
//
//  Created by Brandon Kelly on 3/2/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef __yamcmc____random__
#define __yamcmc____random__

#include <iostream>
//
//  random.hpp
//  yamcmc++
//
//  Created by Dr. Brandon C. Kelly on 11/21/12.
//
//  Header file for a class that generates pseudorandom numbers from common distributions.
//  This is basically a wrapper for BOOST::RANDOM
//

// Standard includes
#include <string>
#include <vector>
// Boost includes
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/lognormal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/student_t_distribution.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>

// Other includes
#include <armadillo>

// Class containing methods to generate random numbers from various
// distributions. These should be self-explanatory, but see
// random.cpp for more details.
class RandomGenerator {
public:
    void SetSeed(unsigned long seed) const; // Set the random number generator seed. Be Careful with this!
    void SaveSeed(std::string seed_filename = "seed.txt") const; // Save the random number generator seed to a file.
    void RecoverSeed(std::string seed_filename = "seed.txt") const;
    double exp(double lambda=1.0);
    double normal(double mu=0.0, double sigma=1.0); // Univariate normal
    arma::vec normal(arma::mat covar); // Multivariate normal
    double lognormal(double logmean=0.0, double frac_sigma=1.0);
    double uniform(double lowbound=0.0, double upbound=1.0);
    int uniform(int lowbound, int upbound);
    double powerlaw(double lower, double upper, double slope);
    double tdist(double dof=1.0, double mean=0.0, double scale=1.0);
    double chisqr(int dof=1);
    double scaled_inverse_chisqr(int dof=1, double ssqr=1.0);
    double gamma(double alpha=1.0, double beta=1.0);
    double invgamma(double alpha=1.0, double beta=1.0);
    int uniform_integer(int lowbound, int upbound);
    // Additional methods to be added later
    arma::vec mtdist(arma::mat covar, double dof=1.0);
    double beta();
    double weibull();
private:
    // Private functors for the various distributions.
    boost::random::exponential_distribution<> exp_;
    boost::random::normal_distribution<> normal_;
    boost::random::lognormal_distribution<> lognormal_;
    boost::random::uniform_real_distribution<> uniform_;
    boost::random::uniform_int_distribution<> uniform_integer_;
    boost::random::student_t_distribution<> tdist_;
    boost::random::chi_squared_distribution<> chisqr_;
    boost::random::gamma_distribution<> gamma_;
};

#endif /* defined(__yamcmc____random__) */
