/*
 MCMC SAMPLER FOR A CAR(p) PROCESS.
 
 AUTHOR: DR. BRANDON C. KELLY, DEC 2012
		 DEPT. OF PHYSICS
		 UNIV. OF CALIFORNIA, SANTA BARBARA
 
 
 TODO:
	- Include optimizer to start the sampler off at the maximum-likelihood estimate,
	  use Fisher information matrix as initial guess for proposal scale matrix.
 
 */

// Standard includes
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
// Include the MCMC sampler header files
#include <random.hpp>
#include <proposals.hpp>
#include <samplers.hpp>
#include <steps.hpp>
#include <parameters.hpp>
// Local include
#include "include/carpack.hpp"
#include "include/carmcmc.hpp"

// Run the MCMC sampler for a CARMA(p,q) process
std::pair<std::vector<arma::vec>, std::vector<double> > 
RunEnsembleCarmaSampler(int sample_size, int burnin, arma::vec time, arma::vec y,
                      arma::vec yerr, int p, int q, int nwalkers, bool do_zcarma,
                      int thin)
{
    std::pair<std::vector<arma::vec>, std::vector<double> > mcmc_output;
    if (p == 1) {
        mcmc_output = RunCar1Sampler(sample_size, burnin, time, y, yerr, nwalkers, thin);
    } else {
        mcmc_output = RunCarmaSampler(sample_size, burnin, time, y, yerr, nwalkers, p, q, do_zcarma, thin);
    }
    return mcmc_output;
}

std::pair<std::vector<arma::vec>, std::vector<double> >
RunCar1Sampler(int sample_size, int burnin, arma::vec time, arma::vec y, arma::vec yerr,
               int nwalkers, int thin)
{
    int p = 1;
    double max_stdev = 10.0 * arma::stddev(y); // For prior: maximum standard-deviation of CAR(1) process
    
    // Set the temperature ladder. The logarithms of the temperatures are on a linear grid
    double max_temperature = 100.0;
    
    arma::vec temp_ladder = arma::linspace<arma::vec>(0.0, log(max_temperature), nwalkers);
    temp_ladder = arma::exp(temp_ladder);
    
    Ensemble<CAR1> CarEnsemble;
    // Add the parameters to the ensemble, starting with the coolest chain
	for (int i=0; i<nwalkers; i++)
    {
		// Add this walker to the ensemble
        CarEnsemble.AddObject(new CAR1(false, "CAR(1) Parameters", time, y, yerr, temp_ladder(i)));
		// Set the prior parameters
        CarEnsemble[i].SetPrior(max_stdev);
	}
    
    // Report average acceptance rates at end of sampler
    int report_iter = burnin + thin * sample_size;
    
    // Setup initial covariance matrix for RAM proposals. This
	// is just a diagonal matrix with the diagonal elements equal to 0.01^2.
	//
	// TODO: Get a better guess from maximum-likelihood fit
	//
	arma::mat prop_covar(2+p,2+p);
	prop_covar.eye();
	prop_covar.diag() = prop_covar.diag() * 0.01 * 0.01;
    prop_covar(0,0) = 2.0 * arma::var(y) * arma::var(y) / y.n_elem;
    
	// Instantiate base proposal object
    StudentProposal RAMProp(8.0, 1.0);
	//NormalProposal RAMProp(1.0);
    double target_rate = 0.25;

    // Instantiate MCMC Sampler object for CAR process
	Sampler CarModel(sample_size, burnin, thin);
    
    // Add the steps to the sampler, starting with the hottest chain first
    for (int i=nwalkers-1; i>0; i--) {
        // First add Robust Adaptive Metropolis Step
        CarModel.AddStep( new AdaptiveMetro(CarEnsemble[i], RAMProp, prop_covar, 
                                            target_rate, burnin) );
        // Now add Exchange steps
        CarModel.AddStep( new ExchangeStep<arma::vec, CAR1>(CarEnsemble[i], i, CarEnsemble, report_iter) );
    }
    
    // Make sure we set this parameter to be tracked
    CarEnsemble[0].SetTracking(true);
    // Add in coolest chain. This is the chain that is actually moving in the posterior.
    CarModel.AddStep( new AdaptiveMetro(CarEnsemble[0], RAMProp, prop_covar, target_rate, burnin) );

    // Now run the MCMC sampler. The samples will be dumped in the 
    // output file provided by the user.
    
	CarModel.Run();
    
    std::vector<arma::vec> car_samples = CarEnsemble[0].GetSamples();
    std::vector<double> car_likes = CarEnsemble[0].GetLogLikes();

    return std::make_pair(car_samples, car_likes);
}

std::pair<std::vector<arma::vec>, std::vector<double> >
RunCarmaSampler(int sample_size, int burnin, arma::vec time, arma::vec y,
                arma::vec yerr, int nwalkers, int p, int q, bool do_zcarma,
                int thin)
{
    double max_stdev = 10.0 * arma::stddev(y); // For prior: maximum standard-deviation of CAR(1) process
    
    // Set the temperature ladder. The logarithms of the temperatures are on a linear grid
    double max_temperature = 100.0;
    
    arma::vec temp_ladder = arma::linspace<arma::vec>(0.0, log(max_temperature), nwalkers);
    temp_ladder = arma::exp(temp_ladder);
    
    Ensemble<CARp> CarEnsemble;
    // Add the parameters to the ensemble, starting with the coolest chain
	for (int i=0; i<nwalkers; i++)
    {
		// Add this walker to the ensemble
        if (!do_zcarma) {
            if (q == 0) {
                // just doing a CAR(p) model
                CarEnsemble.AddObject(new CARp(false, "CAR(p) Parameters", time, y, yerr, p, temp_ladder(i)));
            } else {
                // doing a CARMA(p,q) model
                CarEnsemble.AddObject(new CARMA(false, "CARMA(p,q) Parameters", time, y, yerr, p, q, temp_ladder(i)));
            }
        } else {
            // doing a ZCARMA(p) model
            CarEnsemble.AddObject(new ZCARMA(false, "ZCARMA(p) Parameters", time, y, yerr, p, temp_ladder(i)));
        }
		// Set the prior parameters
        CarEnsemble[i].SetPrior(max_stdev);
	}
    
    // Report average acceptance rates at end of sampler
    int report_iter = burnin + thin * sample_size;
    
    // Setup initial covariance matrix for RAM proposals. This
	// is just a diagonal matrix with the diagonal elements equal to 0.01^2.
	//
	// TODO: Get a better guess from maximum-likelihood fit
	//
    
    int nparams;
    if (do_zcarma) {
        nparams = 3 + p;
    } else {
        nparams = 2 + p + q;
    }
    
	arma::mat prop_covar(nparams,nparams);
	prop_covar.eye();
	prop_covar.diag() = prop_covar.diag() * 0.01 * 0.01;
    prop_covar(0,0) = 2.0 * arma::var(y) * arma::var(y) / y.n_elem;
    
	// Instantiate base proposal object
    StudentProposal RAMProp(8.0, 1.0);
	//NormalProposal RAMProp(1.0);
    double target_rate = 0.25;
    
    // Instantiate MCMC Sampler object for CAR process
	Sampler CarModel(sample_size, burnin, thin);
    
    // Add the steps to the sampler, starting with the hottest chain first
    for (int i=nwalkers-1; i>0; i--) {
        // First add Robust Adaptive Metropolis Step
        CarModel.AddStep( new AdaptiveMetro(CarEnsemble[i], RAMProp, prop_covar,
                                            target_rate, burnin) );
        // Now add Exchange steps
        CarModel.AddStep( new ExchangeStep<arma::vec, CARp>(CarEnsemble[i], i, CarEnsemble, report_iter) );
    }
    
    // Make sure we set this parameter to be tracked
    CarEnsemble[0].SetTracking(true);
    // Add in coolest chain. This is the chain that is actually moving in the posterior.
    CarModel.AddStep( new AdaptiveMetro(CarEnsemble[0], RAMProp, prop_covar, target_rate, burnin) );
    
    // Now run the MCMC sampler. The samples will be dumped in the
    // output file provided by the user.
    
	CarModel.Run();
    
    std::vector<arma::vec> car_samples = CarEnsemble[0].GetSamples();
    std::vector<double> car_likes = CarEnsemble[0].GetLogLikes();
    
    return std::make_pair(car_samples, car_likes);
}
