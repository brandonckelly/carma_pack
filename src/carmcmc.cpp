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
#include <numeric>
// Include the MCMC sampler header files
#include <random.hpp>
#include <proposals.hpp>
#include <samplers.hpp>
#include <steps.hpp>
#include <parameters.hpp>
#include "include/carpack.hpp"
#include "include/carmcmc.hpp"

std::shared_ptr<CAR1>
RunCar1Sampler(int sample_size, int burnin, std::vector<double> time, std::vector<double> y, std::vector<double> yerr, 
	       int thin, const std::vector<double>& init)
{
    int p = 1;    
    double sum = std::accumulate(y.begin(), y.end(), 0.0);
    double mean = sum / y.size();
    double sq_sum = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
    double var = (sq_sum / y.size() - mean * mean);

	double max_stdev = 10.0 * std::sqrt(var); // For prior: maximum standard-deviation of CAR(1) process
        
    // Report average acceptance rates at end of sampler
    int report_iter = burnin + thin * sample_size;
    
    // Setup initial covariance matrix for RAM proposals. This
	// is just a diagonal matrix with the diagonal elements equal to 0.01^2.
	//
	// TODO: Get a better guess from maximum-likelihood fit
	//
	arma::mat prop_covar(3+p,3+p);
	prop_covar.eye();
	prop_covar.diag() = prop_covar.diag() * 0.01 * 0.01;
    prop_covar(0,0) = 2.0 * var * var / y.size();
    prop_covar(2,2) = var / y.size();
    
	// Instantiate base proposal object
    StudentProposal RAMProp(8.0, 1.0);
	//NormalProposal RAMProp(1.0);
    double target_rate = 0.25;

    // Instantiate MCMC Sampler object for CAR process
	Sampler CarModel(sample_size, burnin, thin);
    
	// Construct the parameter object
    CAR1 Car1Par(true, "CAR(1)", time, y, yerr);
    Car1Par.SetPrior(max_stdev);

    // Add Robust Adaptive Metropolis Step
    CarModel.AddStep( new AdaptiveMetro(Car1Par, RAMProp, prop_covar, target_rate, burnin) );

    // Now run the MCMC sampler.
    arma::vec armaInit = arma::conv_to<arma::vec>::from(init);
    CarModel.Run(armaInit);

    std::shared_ptr<CAR1> retObject = std::make_shared<CAR1>(Car1Par);
    return retObject;
}

std::shared_ptr<CARp>
RunCarmaSampler(int sample_size, int burnin, std::vector<double> time, std::vector<double> y,
                std::vector<double> yerr, int p, int q, int nwalkers, bool do_zcarma,
                int thin, const std::vector<double>& init)
{
    assert(p > 1);
    double sum = std::accumulate(y.begin(), y.end(), 0.0);
    double mean = sum / y.size();
    double sq_sum = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
    double var = (sq_sum / y.size() - mean * mean);
    double max_stdev = 10.0 * sqrt(var);
    
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
            CarEnsemble.AddObject(new ZCAR(false, "ZCAR(p) Parameters", time, y, yerr, p, temp_ladder(i)));
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
    
    int nparams = 3 + p + q;
    if (do_zcarma) {
        nparams = 3 + p;
    }
    
	arma::mat prop_covar(nparams,nparams);
	prop_covar.eye();
	prop_covar.diag() = prop_covar.diag() * 0.01 * 0.01;
    prop_covar(0,0) = 2.0 * var * var / y.size();
    prop_covar(2,2) = var / y.size();
    
	// Instantiate base proposal object
    StudentProposal RAMProp(8.0, 1.0);
	//NormalProposal RAMProp(1.0);
    double target_rate = 0.25;
    
    // Instantiate MCMC Sampler object for CAR process
    Sampler CarModel(sample_size, burnin, thin);
    
    // Add the steps to the sampler, starting with the hottest chain first
    for (int i=nwalkers-1; i>0; i--) {
        // First add Robust Adaptive Metropolis Step
        CarModel.AddStep( new AdaptiveMetro(CarEnsemble[i], RAMProp, prop_covar, target_rate, burnin) );
        // Now add Exchange steps
        CarModel.AddStep( new ExchangeStep<arma::vec, CARp>(CarEnsemble[i], i, CarEnsemble, report_iter) );
    }
    
    // Make sure we set this parameter to be tracked
    CarEnsemble[0].SetTracking(true);
    // Add in coolest chain. This is the chain that is actually moving in the posterior.
    CarModel.AddStep( new AdaptiveMetro(CarEnsemble[0], RAMProp, prop_covar, target_rate, burnin) );
    
    // Now run the MCMC sampler. The samples will be dumped in the
    // output file provided by the user.
    arma::vec armaInit = arma::conv_to<arma::vec>::from(init);
    CarModel.Run(armaInit);
    
    std::shared_ptr<CARp> retObject;

    if (do_zcarma) {
        retObject = std::make_shared<ZCAR>(*(dynamic_cast<ZCAR*>(&CarEnsemble[0])));
    } else {
        if (q == 0) {
            retObject = std::make_shared<CARp>(CarEnsemble[0]);
        } else {
            retObject = std::make_shared<CARMA>(*(dynamic_cast<CARMA*>(&CarEnsemble[0])));
        }
    }
    
    return retObject;
}
