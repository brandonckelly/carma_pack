//
//  steps.cpp
//  yamcmc++
//
//  Created by Brandon Kelly on 3/2/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#include <boost/timer.hpp>
// Local includes
#include "include/steps.hpp"

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
RandomGenerator RandGen;

/* ****** Methods of AdaptiveMetro class ********* */

// Constructor, requires a parameter object, a proposal object, an initial
// covariance matrix for the multivariate proposals, a target acceptance rate,
// and the maximum number of iterations to perform the adaptations for.
AdaptiveMetro::AdaptiveMetro(Parameter<arma::vec>& parameter, Proposal<double>& proposal,
                             arma::mat proposal_covar, double target_rate, int maxiter) :
parameter_(parameter), proposal_(proposal),
target_rate_(target_rate), maxiter_(maxiter)
{
	gamma_ = 2.0 / 3.0;
	niter_ = 0;
	naccept_ = 0;
	chol_factor_ = arma::chol(proposal_covar);
}

// Method to calculate whether the proposal is accepted
bool AdaptiveMetro::Accept(arma::vec new_value, arma::vec old_value) {
	
	// MH accept/reject criteria: Proposal must be symmetric!!
	alpha_ = (parameter_.LogDensity(new_value) - parameter_.GetLogDensity()) / parameter_.GetTemperature();
    
	if (!arma::is_finite(alpha_)) {
		// New value of the log-posterior is not finite, so reject this
		// proposal
        alpha_ = 0.0;
		return false;
	}
	
	double unif = uniform_(rng);
	alpha_ = std::min(exp(alpha_), 1.0);
	if (unif < alpha_) {
		naccept_++;
		return true;
	} else {
		return false;
	}
}

// Method to perform the RAM step. This involves a standard Metropolis-Hastings update, followed
// by an update to the proposal scale matrix so long as niter < maxiter
void AdaptiveMetro::DoStep()
{
	arma::vec old_value = parameter_.Value();
    
	// Draw a new parameter vector
	arma::vec unit_proposal(old_value.n_rows);
	for (int i=0; i<old_value.n_rows; i++) {
		// Unscaled proposal
		unit_proposal(i) = proposal_.Draw(0.0);
	}
	
	// Scaled proposal vector
	arma::vec scaled_proposal = chol_factor_.t() * unit_proposal;
	arma::vec new_value = old_value + scaled_proposal;
	
	// MH accept/reject criteria
	if (Accept(new_value, old_value)) {
		parameter_.Save(new_value);
	}
    
    double step_size, unit_norm;
    
	if ((niter_ < maxiter_) && arma::is_finite(alpha_)) {
		// Still in the adaptive stage, so update the scale matrix cholesky factor
		
		// The step size sequence for the scale matrix update. This is eta_n in the
		// notation of Vihola (2012)
		step_size = std::min(1.0, new_value.n_rows / pow(niter_, gamma_));
        
		unit_norm = arma::norm(unit_proposal, 2);
        
		// Rescale the proposal vector for updating the scale matrix cholesky factor
		scaled_proposal = sqrt(step_size * fabs(alpha_ - target_rate_)) / unit_norm * scaled_proposal;
        
		// Update or downdate the Cholesky factor?
		bool downdate = (alpha_ < target_rate_);
        
		// Perform the rank-1 update (downdate) of the scale matrix Cholesky factor
		CholUpdateR1(chol_factor_, scaled_proposal, downdate);
	}
    
	niter_++;
	
	if (niter_ == maxiter_) {
		double arate = ((double)(naccept_)) / ((double)(niter_));
		std::cout << "Average RAM Acceptance Rate is " << arate << std::endl;
	}
}

// Function to perform the rank-1 Cholesky update, needed for updating the
// proposal covariance matrix
void CholUpdateR1(arma::mat& L, arma::vec& v, bool downdate)
{
	double sign = 1.0;
	if (downdate) {
		// Perform the downdate instead
		sign = -1.0;
	}
	for (int k=0; k<L.n_rows; k++) {
		double r = sqrt( L(k,k) * L(k,k) + sign * v(k) * v(k) );
		double c = r / L(k,k);
		double s = v(k) / L(k,k);
		L(k,k) = r;
		if (k < L.n_rows-1) {
			L(k,arma::span(k+1,L.n_rows-1)) = (L(k,arma::span(k+1,L.n_rows-1)) +
                                               sign * s * v(arma::span(k+1,v.n_elem-1)).t()) / c;
			v(arma::span(k+1,v.n_elem-1)) = c * v(arma::span(k+1,v.n_elem-1)) -
			s * L(k,arma::span(k+1,L.n_rows-1)).t();
		}
		
	}
}

