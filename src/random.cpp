//
//  random.cpp
//  yamcmc++
//
//  Created by Dr. Brandon C. Kelly on 11/21/12.
//  Source file for a class that generates pseudorandom numbers from common distributions.
//  This is basically a wrapper for BOOST::RANDOM
//

// Standard includes
#include <iostream>
#include <fstream>
// Local include
#include "include/random.hpp"

// Global random number generator. This same generator should be used for
// generating all random variates for a MCMC sampler. The default random
// number generator is the Mersenne Twister mt19937 from the BOOST library.

boost::random::mt19937 rng(time(NULL));


// Method to set the seed of the global random number generator.
void RandomGenerator::SetSeed(unsigned long seed) const
{
    rng.seed(seed);
}

// Method to save the seed of the global random number generator to a file.
// The default filename is "seed.txt"
void RandomGenerator::SaveSeed(std::string seed_filename) const
{
    std::ofstream seed_file(seed_filename.c_str());
	if (seed_file.is_open()) {
		seed_file << rng;
	} else {
		std::cout << "Cannot write random number generator seed to file "
        << seed_filename << ".\n";
	}
	seed_file.close();
}

// Method to recover the seed of the global random number generator from a file.
// The default filename is "seed.txt"
void RandomGenerator::RecoverSeed(std::string seed_filename) const
{
	std::ifstream seed_file(seed_filename.c_str());
	if (seed_file.is_open()) {
		seed_file >> rng;
	} else {
		std::cout << "Cannot read random number generator seed from file "
		<< seed_filename <<	".\n";
	}
	seed_file.close();
}

// Method to return a exponentially distributed random variate. The parameter
// lambda denotes the scale parameter for the exponential distribution:
//
//	p(x|lambda) = lambda * exp(-lambda * x)

double RandomGenerator::exp(double lambda)
{	// Initialize an exponential parameter object with parameter lambda.
	boost::random::exponential_distribution<>::param_type exp_params(lambda);
	// Set the parameter object owned by the exp_ distribution object to be
	// the new parameter object with scale parameter lambda.
	exp_.param(exp_params);
	// Generate and return a exponentially-distribution random deviate. Note
	// that rng is the global random number generator defined in the header file.
    return exp_(rng);
}

// Method to return a normally distributed random variate. The parameters are
// the mean, mu, and the standard deviation, sigma.

double RandomGenerator::normal(double mu, double sigma)
{
	boost::random::normal_distribution<>::param_type normal_params(mu, sigma);
	normal_.param(normal_params);
	return normal_(rng);
}

// Over-loaded Method to return a random vector drawn from a multivariate normal distribution
// of mean zero:
//
// p(x|covar) \propto 1 / |covar|^{1/2} exp(-0.5 x^T covar^{-1} x)

arma::vec RandomGenerator::normal(arma::mat covar)
{
	// Get matrix square root of covar via Cholesky decomposition
	arma::mat R = arma::chol(covar);
	
	// Rest normal distribution parameters
	boost::random::normal_distribution<>::param_type normal_params(0.0, 1.0);
	normal_.param(normal_params);
	
	// Vector of random variate independently drawn from a standard normal
	arma::vec z(covar.n_rows);
	for (int i=0; i<z.n_elem; i++) {
		z(i) = normal_(rng);
	}
	
	arma::vec x = R.t() * z;
	return x;
}

// Method to return a log-normally distributed random variate. The parameters are
// the geometric mean, geomean, and the fractional standard deviation, frac_sigma:
//
// p(x|geomean,frac_sigma) /propto
//		(1 / (x * frac_sigma)) * exp(-0.5 * (log(x) - geomean)^2 / frac_sigma^2)

double RandomGenerator::lognormal(double logmean, double frac_sigma)
{
	boost::random::lognormal_distribution<>::param_type lognormal_params(logmean, frac_sigma);
	lognormal_.param(lognormal_params);
	return lognormal_(rng);
}

// Method to return a uniformaly distributed random variate between lowbound and upbound.
double RandomGenerator::uniform(double lowbound, double upbound)
{
	boost::random::uniform_real_distribution<>::param_type unif_params(lowbound, upbound);
	uniform_.param(unif_params);
	return uniform_(rng);
}

// Overloaded method to return a uniformaly distributed integer between lowbound and upbound.
int RandomGenerator::uniform(int lowbound, int upbound)
{
	boost::random::uniform_int_distribution<>::param_type unif_params(lowbound, upbound);
	uniform_integer_.param(unif_params);
	return uniform_integer_(rng);
}

// Method to return a random variate drawn from a bounded power-law distribution. The
// parameters are the slope, slope, the lowerbound, lower, and the upperbound, upper:
//
// p(x|slope,lower,upper) \propto x^slope, lower < x < upper

double RandomGenerator::powerlaw(double lower, double upper, double slope)
{
	// First draw Y ~ Uniform(0,1)
	double unif = uniform(0.0, 1.0);
	
	// Now do transformation Y -> X such that X ~ Powerlaw(slope,lower,upper)
	double power = 1.0 / (slope + 1.0);
	double base_value= (pow(upper, slope+1) - pow(lower, slope+1)) * unif + pow(lower,slope+1);
	
	return pow(base_value, power);
}

// Method to return a random variate drawn from a Student's t-distribution. The parameters
// are the degrees of freedom, dof, the mean, mean, and the scale parameter, scale:
//
// p(x|dof,mean,scale) \propto (1 + (x - mean)^2 / (dof * scale^2))^(-(n+1)/2)

double RandomGenerator::tdist(double dof, double mean, double scale)
{
	boost::random::student_t_distribution<>::param_type t_param(dof);
	tdist_.param(t_param);
	double zdraw = tdist_(rng);
	return mean + scale * zdraw;
}

// Method to return a chi-squared random variate. The parameters are the
// degrees of freedom, dof.
double RandomGenerator::chisqr(int dof)
{
	boost::random::chi_squared_distribution<>::param_type chisqr_param(dof);
	chisqr_.param(chisqr_param);
	return chisqr_(rng);
}

// Method to return a random variate drawn from a scaled inverse chi-square distribution. The
// parameters are the degrees of freedom, dof, and the scale parameter, ssqr:
//
// p(x|dof,ssqr) \propto 1 / x^(1 + dof/2) * exp(-dof * ssqr / (dof * x))

double RandomGenerator::scaled_inverse_chisqr(int dof, double ssqr)
{
	boost::random::chi_squared_distribution<>::param_type chisqr_param(dof);
	chisqr_.param(chisqr_param);
	double chi2 = chisqr_(rng);
    return ssqr / chi2 * ((double)(dof));
}

// Method to return a random variate drawn from a gamma distribution with shape
// parameter alpha and scale parameter beta:
//
// p(x|alpha,beta) \propto x^(alpha-1) exp(-x / beta)

double RandomGenerator::gamma(double alpha, double beta)
{
	boost::random::gamma_distribution<>::param_type gamma_params(alpha,beta);
	gamma_.param(gamma_params);
	return gamma_(rng);
}

// Method to return a random variate drawn from an inverse gamma distribution
// with shape parameter alpha and inverse scale parameter beta:
//
// p(x|alpha,beta) \propto x^(-alpha-1) exp(-beta / x)

double RandomGenerator::invgamma(double alpha, double beta)
{
	boost::random::gamma_distribution<>::param_type gamma_params(alpha, 1.0 / beta);
	gamma_.param(gamma_params);
	return 1.0 / gamma_(rng);
}

// Method to return a random vector drawn from a multivariate Student's t-distribution.
// More to come later.
arma::vec RandomGenerator::mtdist(arma::mat covar, double dof)
{
	arma::vec t(covar.n_rows);
	return t;
}

// Method to return a random variate drawn from a beta distribution. More to come.
double RandomGenerator::beta()
{
	return 0.0;
}

// Method to return a random variate drawn from a Weibull distribution. More to come.
double RandomGenerator::weibull()
{
	return 0.0;
}
