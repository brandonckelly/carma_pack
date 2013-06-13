/*
 MCMC SAMPLER FOR A CAR(p) PROCESS.
 
 AUTHOR: DR. BRANDON C. KELLY, DEC 2012
		 DEPT. OF PHYSICS
		 UNIV. OF CALIFORNIA, SANTA BARBARA
 
 
 TESTING HISTORY:
 
 ************ CAR(1) ************
 
 PASS: Test loading in data and instantiating CAR1 class, including
	   when the time values are not sorted and contain duplicates (12/20/2012)
 PASS: CAR1::StartingValue (12/20/2012)
 PASS: CAR1::Value (12/20/2012)
 PASS: CAR1::StringValue (12/20/2012)
 PASS: CAR1::Save (12/20/2012)
 PASS: CAR1::KalmanFilter (12/21/2012)
 PASS: CAR1::LogDensity
 PASS: Test MCMC sampler without measurement errors, starting at true values. (12/21/2012)
 PASS: Test MCMC sampler without meas. errors, with random starts. (12/21/2012)
 PASS: Test MCMC sampler with measurement errors, starting at true values. (12/21/2012)
 PASS: Test MCMC sampler with measurement errors, with random startings values. (12/21/2012)
 PASS: Test MCMC sampler with all of above for time series with long gaps (12/24/2012)
 PASS: Test MCMC sampler on real AGN lightcurve, compare with IDL results (1/3/2013)
  
 ************* CAR(p) ************
 
 PASS: Test loading in data and instantiating CARp object, including
	   when the time values are not sorted and contain duplicates (12/31/2012)
 PASS: Test root finding algorithm and polycoefs function (12/31/2012)
 PASS: Test unique_roots function (12/31/2012)
 PASS: Within StartingValue, test alpha_roots --> phi and phi --> alpha_roots (12/31/2012)
 PASS: CARp::StartingValue (12/31/2012)
 PASS: CARp::Value (1/1/2013)
 PASS: CARp::StringValue (1/1/2013)
 PASS: CARp::SetKappa (1/1/2013)
 PASS: CARp::Save (1/1/2013)
 PASS: CARp::Variance (1/1/2013)
 PASS: CARp::KalmanFilter (1/2/2013)
 PASS: CARp::LogDensity (1/3/2013)

 Generated a CAR(1) process without measurement errors in IDL as follows:
 
 ny = 1000
 dt = randomu(seed, ny)
 time = total(dt, /cum)
 tau = 25.0
 omega = 1.0 / tau
 sigma = 1.0
 mu = 6.0
 kappa = 1.0 / median(time[1:*] - time)
 phi = -1.0 * (1.0 + omega / kappa) / (1.0 - omega / kappa)
 y = ouprocess(seed, time, tau, sigma) + mu
 
 Test fitting this data with a CAR(2) process. Did the following tests to compare
 the model powerspectrum with the true powerspectrum:
 
 PASS: Test MCMC sampler without measurement errors, starting at true values. (1/7/2013)
 PASS: Test MCMC sampler without meas. errors, with random starts. (1/7/2013)
 PASS: Test MCMC sampler with 10% measurement errors, with random startings values. (1/7/2013)
 
 Generated a CAR(2) process without measurement errors in IDL as follows:
 
 ny = 1000
 dt = randomu(seed, ny)
 time = total(dt, /cum)
 lorentz_cent = 1d / 25.0
 lorentz_width = 1d / 100.0
 roots = dcomplexarr(2)
 roots[0] = dcomplex(-1.0 * lorentz_width, lorentz_cent)
 roots[1] = dcomplex(-1.0 * lorentz_width, -1.0 * lorentz_cent)
 roots = roots * 2.0 * !pi
 sigma = 1.0
 mu = 6.0
 alpha = polycoefs(roots)
 alpha = alpha[1:*]
 kappa = 1.0 / median(time[1:*]-time)
 y = carma_process(seed, time, sigma^2, reverse(alpha), kappa=kappa) + mu
 
 phi_roots = (1.0 + roots / kappa) / (1.0 - roots / kappa)
 phi = polycoefs(phi_roots)
 phi = phi[1:*]
 print, phi
 -1.9530404      0.95999374

 PASS: Test MCMC sampler without measurement errors, starting at true values. (1/8/2013)
 PASS: Test MCMC sampler without meas. errors, with random starts. (1/8/2013)
 PASS: Test MCMC sampler with measurement errors, with random startings values. (1/8/2013)
 PASS, but slow convergence: Test MCMC sampler with meas. err., random starts, and using a CAR(3) process. (1/9/2013)
 
 Test MCMC sampler with all of above but for a CAR(5) process
 Test MCMC sampler with all of above for time series with long gaps
 Test MCMC sampler on real AGN lightcurve, compare with IDL results
 
 TODO: 
	- Include optimizer to start the sampler off at the maximum-likelihood estimate,
	  use Fisher information matrix as initial guess for proposal scale matrix.
 
 */

// Standard includes
#include <iostream>
#include <fstream>
#include <vector>
// Include the MCMC sampler header files
#include "yamcmc++.hpp"
// Local include
#include "carpack.hpp"
#include "carmcmc.hpp"

// Run the MCMC sampler for a CAR(p) process
std::vector<arma::vec> RunEnsembleCarSampler(MCMCOptions mcmc_options, arma::vec time, arma::vec y,
                                             arma::vec yerr, int p, int nwalkers)
{
    // Instantiate MCMC Sampler object for CAR process
	Sampler CarModel(mcmc_options.sample_size, mcmc_options.burnin, mcmc_options.thin);
	
	// Construct the parameter ensemble
    Ensemble<CAR1> CarEnsemble;
    
	double max_stdev = 10.0 * arma::stddev(y); // For prior: maximum standard-deviation of CAR(1) process
    
    // Set the temperature ladder. The logarithms of the temperatures are on a linear grid
    double max_temperature = 100.0;
    
    arma::vec temp_ladder = arma::linspace<arma::vec>(0.0, log(max_temperature), nwalkers);
    temp_ladder = arma::exp(temp_ladder);
    
    // Add the parameters to the ensemble, starting with the coolest chain
	for (int i=0; i<nwalkers; i++) {
		// Add this walker to the ensemble
		if (p == 1) {
			// Doing a CAR(1) model
			CarEnsemble.AddObject(new CAR1(false, "CAR(1) Parameters", time, y, yerr, temp_ladder(i)));
		} else {
			// Doing a CAR(p) model
			CarEnsemble.AddObject(new CARp(false, "CAR(p) Parameters", time, y, yerr, p, temp_ladder(i)));
		}
		
		// Set the prior parameters
        CarEnsemble[i].SetPrior(max_stdev);
	}
    
    // Report average acceptance rates at end of sampler
    int report_iter = mcmc_options.burnin + mcmc_options.thin * mcmc_options.sample_size;
    
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

    //test_ptemp(CarEnsemble, RAMProp, prop_covar);
    
    // Add the steps to the sampler, starting with the hottest chain first
    for (int i=nwalkers-1; i>0; i--) {
        // First add Robust Adaptive Metropolis Step
        CarModel.AddStep( new AdaptiveMetro(CarEnsemble[i], RAMProp, prop_covar, 
                                            target_rate, mcmc_options.burnin) );
        // Now add Exchange steps
        CarModel.AddStep( new ExchangeStep<arma::vec, CAR1>(CarEnsemble[i], i, CarEnsemble, report_iter) );
    }
    
    // Make sure we set this parameter to be tracked
    CarEnsemble[0].SetTracking(true);
    // Add in coolest chain. This is the chain that is actually moving in the posterior.
    CarModel.AddStep( new AdaptiveMetro(CarEnsemble[0], RAMProp, prop_covar, target_rate, mcmc_options.burnin) );

    // Now run the MCMC sampler. The samples will be dumped in the 
    // output file provided by the user.
    
	CarModel.Run();
    
    std::vector<arma::vec> car_samples = CarEnsemble[0].GetSamples();
    
    return car_samples;
}
