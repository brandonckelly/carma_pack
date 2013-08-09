/*
 *  carpack.cpp
 *  carpack
 *
 *  Created by Brandon Kelly on 12/19/12.
 *
 *  Method definitions of classes for MCMC sampling from continuous time
 *  autoregressive (CAR) models.
 *
 */

// Standard includes
#include <complex>
#include <iostream>

// Boost includes
#include <boost/math/special_functions/binomial.hpp>

// Local includes
#include "include/carpack.hpp"

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

/********************************************************************
						METHODS OF CAR1 CLASS
 *******************************************************************/

// Method of CAR1 class to generate the starting values of the
// parameters theta = (mu, sigma, measerr_scale, log(omega)).

arma::vec CAR1::StartingValue()
{
	double log_omega_start, car1_stdev_start, sigma;

	// Initialize the standard deviation of the CAR(1) process
	// by drawing from its prior
	car1_stdev_start = RandGen.scaled_inverse_chisqr(y_.size()-1, arma::var(y_));
	car1_stdev_start = sqrt(car1_stdev_start);
	
	// Initialize log(omega) to log( 1 / (a * median(dt)) ), where
	// a ~ Uniform(1,50) , under the constraint that 
	// tau = 1 / omega < max(time)
	
    arma::vec dt = time_(arma::span(1,time_.n_elem-1)) - time_(arma::span(0,time_.n_elem-2));
	log_omega_start = -1.0 * log(arma::median(dt) * RandGen.uniform( 1.0, 50.0 ));
	log_omega_start = std::min(log_omega_start, max_freq_);
	
	sigma = car1_stdev_start * sqrt(2.0 * exp(log_omega_start));
	
	// Get initial value of the measurement error scaling parameter by
	// drawing from its prior.
	
	double measerr_scale = RandGen.scaled_inverse_chisqr(measerr_dof_, 1.0);
    measerr_scale = std::min(measerr_scale, 1.99);
    measerr_scale = std::max(measerr_scale, 0.51);
	
	arma::vec theta(3);
	
	theta << car1_stdev_start << measerr_scale << log_omega_start << arma::endr;
	
	// Initialize the Kalman filter
    pKFilter_->SetOmega(exp(log_omega_start));
    pKFilter_->SetSigsqr(sigma * sigma);
    arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
    pKFilter_->SetTimeSeriesErr(proposed_yerr);
    pKFilter_->Filter();
	
	return theta;
}

bool CAR1::CheckPriorBounds(arma::vec theta)
{
    double ysigma = theta(0);
    double measerr_scale = theta(1);
    double omega = exp(theta(2));
    
    bool prior_satisfied = true;
    if ( (omega > max_freq_) || (omega < min_freq_) ||
        (ysigma > max_stdev_) || (ysigma < 0) ||
        (measerr_scale < 0.5) || (measerr_scale > 2.0) ) {
		// prior bounds not satisfied
		prior_satisfied = false;
	}
    return prior_satisfied;
}

/********************************************************************
                        METHODS OF CARp CLASS
 *******************************************************************/

// Calculate the roots of the AR(p) polynomial from the parameters
arma::cx_vec CARp::ARRoots(arma::vec theta)
{
    arma::cx_vec ar_roots(p_);
    
    // Construct the complex vector of roots of the characteristic polynomial:
    // alpha(s) = s^p + alpha_1 s^{p-1} + ... + alpha_{p-1} s + alpha_p
    for (int i=0; i<p_/2; i++) {
        double lorentz_cent = exp(theta(2+2*i)); // PSD is a sum of Lorentzian functions
        double lorentz_width = exp(theta(2+2*i+1));
        ar_roots(2*i) = std::complex<double> (-lorentz_width,lorentz_cent);
        ar_roots(2*i+1) = std::conj(ar_roots(2*i));
    }
	
    if ((p_ % 2) == 1) {
        // p is odd, so add in additional low-frequency component
        double lorentz_width = exp(theta(2+p_-1));
        ar_roots(p_-1) = std::complex<double> (-lorentz_width, 0.0);
    }
    
    ar_roots *= 2.0 * arma::datum::pi;
    
    return ar_roots;
}


// Return the starting value and set log_posterior_
arma::vec CARp::StartingValue()
{
	double min_freq = 1.0 / (time_.max() - time_.min());
    // Create the parameter vector, theta
    arma::vec theta(p_+2);
    
    bool good_initials = false;
    while (!good_initials) {
        
        // Obtain initial values for Lorentzian centroids (= system frequencies) and 
        // widths (= break frequencies)
        arma::vec lorentz_cent((p_+1)/2);
        lorentz_cent.randu();
        lorentz_cent = log(max_freq_ / min_freq) * lorentz_cent + log(min_freq);
        lorentz_cent = arma::exp(lorentz_cent);
        
        // Force system frequencies to be in descending order to make the model identifiable
        lorentz_cent = arma::sort(lorentz_cent, 1);
        
        arma::vec lorentz_width((p_+1)/2);
        lorentz_width.randu();
        lorentz_width = log(max_freq_ / min_freq) * lorentz_width + log(min_freq);
        lorentz_width = arma::exp(lorentz_width);

        if ((p_ % 2) == 1) {
            // p is odd, so add additional low-frequency component
            lorentz_cent(p_/2) = 0.0;
            // make initial break frequency of low-frequency component less than minimum
            // value of the system frequencies
            lorentz_width(p_/2) = exp(RandGen.uniform(log(min_freq), log(lorentz_cent(p_/2-1))));
        }

        for (int i=0; i<p_/2; i++) {
            theta(2+2*i) = log(lorentz_cent(i));
            theta(3+2*i) = log(lorentz_width(i));
        }
        if ((p_ % 2) == 1) {
            // p is odd, so add in additional value of lorentz_width
            theta(p_+1) = log(lorentz_width(p_/2));
        }

        // Initial guess for model standard deviation is randomly distributed
        // around measured standard deviation of the time series
        
        double yvar = RandGen.scaled_inverse_chisqr(y_.size()-1, arma::var(y_));
        arma::cx_vec alpha_roots = ARRoots(theta);
        double sigsqr = yvar / Variance(alpha_roots, ma_coefs_, 1.0);
        // Get initial value of the measurement error scaling parameter by
        // drawing from its prior.
        
        double measerr_scale = RandGen.scaled_inverse_chisqr(measerr_dof_, 1.0);
        measerr_scale = std::min(measerr_scale, 1.99);
        measerr_scale = std::max(measerr_scale, 0.51);
        
        theta(0) = sqrt(yvar);
        theta(1) = measerr_scale;
        
        // set the Kalman filter parameters
        pKFilter_->SetSigsqr(sigsqr);
        pKFilter_->SetOmega(ExtractAR(theta));
        arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
        pKFilter_->SetTimeSeriesErr(proposed_yerr);
        
        // run the kalman filter
        pKFilter_->Filter();
        
        double logpost = LogDensity(theta);
        good_initials = arma::is_finite(logpost);
    } // continue loop until the starting values give us a finite posterior
    
    return theta;
}

// check prior bounds
bool CARp::CheckPriorBounds(arma::vec theta)
{
    double ysigma = theta(0);
    double measerr_scale = theta(1);
    arma::vec lorentz_params = ExtractAR(theta);
    
    // Find the set of Frequencies satisfying the prior bounds
    arma::uvec valid_frequencies1 = arma::find(lorentz_params < max_freq_);
	arma::uvec valid_frequencies2 = arma::find(lorentz_params > min_freq_);
    
    bool prior_satisfied = true;
    
    if ( (valid_frequencies1.n_elem != lorentz_params.n_elem) ||
        (valid_frequencies2.n_elem != lorentz_params.n_elem) ||
        (ysigma > max_stdev_) || (ysigma < 0) ||
        (measerr_scale < 0.5) || (measerr_scale > 2.0) ) {
        // Value are outside of prior bounds
        prior_satisfied = false;
    }
	// Make sure the Lorentzian centroids are still in decreasing order
	for (int i=1; i<p_/2; i++) {
		double lorentz_cent_difference = exp(theta(2+2*(i-1))) - exp(theta(2+2*i));
		if (lorentz_cent_difference < 0) {
			// Lorentzians are not in decreasing order, reject this proposal
			prior_satisfied = false;
		}
    }
    
    return prior_satisfied;
}

// Calculate the variance of the CAR(p) process
double CARp::Variance(arma::cx_vec alpha_roots, arma::vec ma_coefs, double sigma, double dt)
{
    std::complex<double> car_var(0.0,0.0);
    std::complex<double> denom(0.0,0.0);
    std::complex<double> numer(0.0,0.0);
    
	// Calculate the variance of a CAR(p) process
	for (int k=0; k<alpha_roots.n_elem; k++) {
		
		std::complex<double> denom_product(1.0,0.0);
		
		for (int l=0; l<alpha_roots.n_elem; l++) {
			if (l != k) {
				denom_product *= (alpha_roots(l) - alpha_roots(k)) * 
                (std::conj(alpha_roots(l)) + alpha_roots(k));
			}
		}
        denom = -2.0 * std::real(alpha_roots(k)) * denom_product;
        
        int q = ma_coefs.n_elem;
        std::complex<double> ma_sum1(0.0,0.0);
        std::complex<double> ma_sum2(0.0,0.0);
        for (int l=0; l<q; l++) {
            ma_sum1 += ma_coefs(l) * std::pow(alpha_roots(k),l);
            ma_sum2 += ma_coefs(l) * std::pow(-alpha_roots(k),l);
        }
        numer = ma_sum1 * ma_sum2 * std::exp(alpha_roots(k) * dt);
        
        car_var += numer / denom;
	}
	
	// Variance is real-valued, so only return the real part of CARMA_var.
    return sigma * sigma * car_var.real();
}

/*******************************************************************
                        METHODS OF CARMA CLASS
 ******************************************************************/

// Return the starting value and set log_posterior_
arma::vec CARMA::StartingValue()
{
    double min_freq = 1.0 / (time_.max() - time_.min());
    // Create the parameter vector, theta
    arma::vec theta(p_+q_+2);
    
    bool good_initials = false;
    while (!good_initials) {
        
        // Obtain initial values for Lorentzian centroids (= system frequencies) and
        // widths (= break frequencies)
        arma::vec lorentz_cent((p_+1)/2);
        lorentz_cent.randu();
        lorentz_cent = log(max_freq_ / min_freq) * lorentz_cent + log(min_freq);
        lorentz_cent = arma::exp(lorentz_cent);
        
        // Force system frequencies to be in descending order to make the model identifiable
        lorentz_cent = arma::sort(lorentz_cent, 1);
        
        arma::vec lorentz_width((p_+1)/2);
        lorentz_width.randu();
        lorentz_width = log(max_freq_ / min_freq) * lorentz_width + log(min_freq);
        lorentz_width = arma::exp(lorentz_width);
        
        if ((p_ % 2) == 1) {
            // p is odd, so add additional low-frequency component
            lorentz_cent(p_/2) = 0.0;
            // make initial break frequency of low-frequency component less than minimum
            // value of the system frequencies
            lorentz_width(p_/2) = exp(RandGen.uniform(log(min_freq), log(lorentz_cent(p_/2-1))));
        }
        
        for (int i=0; i<p_/2; i++) {
            theta(2+2*i) = log(lorentz_cent(i));
            theta(3+2*i) = log(lorentz_width(i));
        }
        if ((p_ % 2) == 1) {
            // p is odd, so add in additional value of lorentz_width
            theta(p_+1) = log(lorentz_width(p_/2));
        }

        // get initial guess for the moving average polynomial roots
        arma::vec ma_real((q_+1)/2);
        arma::vec ma_imag((q_+1)/2);
        ma_real.randn();
        ma_real = arma::exp(ma_real);
        ma_imag.randn();
        ma_imag = arma::exp(ma_imag);
        if ((q_ % 2) == 1) {
            ma_imag(q_/2) = 0.0;
        }
        // Order imaginary components of MA roots to be in descending order to make the model identifiable
        ma_imag = arma::sort(ma_imag, 1);

        for (int j=0; j<q_/2; j++) {
            theta(2+p_+2*j) = log(ma_imag(j));
            theta(3+p_+2*j) = log(ma_real(j));
        }
        if ((q_ % 2) == 1) {
            theta(q_+p_+1) = log(ma_real(q_/2));
        }
        
        // compute the coefficients of the MA polynomial
        arma::vec ma_coefs(p_);
        ma_coefs = ExtractMA(theta);
        
        // Initial guess for model standard deviation is randomly distributed
        // around measured standard deviation of the time series
        
        double yvar = RandGen.scaled_inverse_chisqr(y_.size()-1, arma::var(y_));
        arma::cx_vec alpha_roots = ARRoots(theta);
        double sigsqr = yvar / Variance(alpha_roots, ma_coefs, 1.0);
        
        // Get initial value of the measurement error scaling parameter by
        // drawing from its prior.
        
        double measerr_scale = RandGen.scaled_inverse_chisqr(measerr_dof_, 1.0);
        measerr_scale = std::min(measerr_scale, 1.99);
        measerr_scale = std::max(measerr_scale, 0.51);
        
        theta(0) = sqrt(yvar);
        theta(1) = measerr_scale;
        
        // set the Kalman filter parameters
        pKFilter_->SetSigsqr(sigsqr);
        pKFilter_->SetOmega(ExtractAR(theta));
        pKFilter_->SetMA(ma_coefs);
        arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
        pKFilter_->SetTimeSeriesErr(proposed_yerr);
        
        // run the kalman filter
        pKFilter_->Filter();
        
        double logpost = LogDensity(theta);
        good_initials = arma::is_finite(logpost);
    } // continue loop until the starting values give us a finite posterior
    
    return theta;
}

// extract the moving-average coefficients from the CARMA parameter vector
arma::vec CARMA::ExtractMA(arma::vec theta)
{
    arma::cx_vec ma_roots(q_);
    
    // Construct the complex vector of roots of the MA polynomial
    for (int i=0; i<q_/2; i++) {
        double ma_imag = exp(theta(2+p_+2*i));
        double ma_real = exp(theta(2+p_+2*i+1));
        ma_roots(2*i) = std::complex<double> (-ma_real,ma_imag);
        ma_roots(2*i+1) = std::conj(ma_roots(2*i));
    }
	
    if ((q_ % 2) == 1) {
        // p is odd, so add in additional low-frequency component
        double ma_real = theta(q_+p_+1);
        ma_roots(q_-1) = std::complex<double> (-ma_real, 0.0);
    }
    // compute the coefficients of the MA polynomial
    arma::vec ma_coefs = polycoefs(ma_roots);
    ma_coefs = arma::flipud(ma_coefs);
    ma_coefs = ma_coefs / ma_coefs(0);
    ma_coefs.resize(p_);
    for (int j=q_+1; j<p_; j++) {
        ma_coefs(j) = 0.0;
    }
    return ma_coefs;
}

// Set the bounds on the uniform prior.
bool CARMA::CheckPriorBounds(arma::vec theta)
{
    bool prior_satisfied = CARp::CheckPriorBounds(theta);

    // Make sure the imaginary components of MA roots are still in decreasing order
	for (int i=1; i<q_/2; i++) {
		double ma_imag_difference = exp(theta(2+p_+2*(i-1))) - exp(theta(2+p_+2*i));
		if (ma_imag_difference < 0) {
			prior_satisfied = false;
		}
    }
    return prior_satisfied;
}


/*******************************************************************
                        METHODS OF ZCARMA CLASS
 *******************************************************************/

arma::vec ZCARMA::StartingValue()
{
    double min_freq = 1.0 / (time_.max() - time_.min());
    // Create the parameter vector, theta
    arma::vec theta(p_+3);
    
    bool good_initials = false;
    while (!good_initials) {
        
        // Obtain initial values for Lorentzian centroids (= system frequencies) and
        // widths (= break frequencies)
        arma::vec lorentz_cent((p_+1)/2);
        lorentz_cent.randu();
        lorentz_cent = log(max_freq_ / min_freq) * lorentz_cent + log(min_freq);
        lorentz_cent = arma::exp(lorentz_cent);
        
        // Force system frequencies to be in descending order to make the model identifiable
        lorentz_cent = arma::sort(lorentz_cent, 1);
        
        arma::vec lorentz_width((p_+1)/2);
        lorentz_width.randu();
        lorentz_width = log(max_freq_ / min_freq) * lorentz_width + log(min_freq);
        lorentz_width = arma::exp(lorentz_width);
        
        if ((p_ % 2) == 1) {
            // p is odd, so add additional low-frequency component
            lorentz_cent(p_/2) = 0.0;
            // make initial break frequency of low-frequency component less than minimum
            // value of the system frequencies
            lorentz_width(p_/2) = exp(RandGen.uniform(log(min_freq), log(lorentz_cent(p_/2-1))));
        }
        
        for (int i=0; i<p_/2; i++) {
            theta(2+2*i) = log(lorentz_cent(i));
            theta(3+2*i) = log(lorentz_width(i));
        }
        if ((p_ % 2) == 1) {
            // p is odd, so add in additional value of lorentz_width
            theta(p_+1) = log(lorentz_width(p_/2));
        }
        
        // get initial guess for the moving average polynomial coefficients, parameterized by kappa
        double kappa_normed = RandGen.uniform();
        theta(2+p_) = logit(kappa_normed);
        
        // compute the coefficients of the MA polynomial
        arma::vec ma_coefs = ExtractMA(theta);
        
        // Initial guess for model standard deviation is randomly distributed
        // around measured standard deviation of the time series
        
        double yvar = RandGen.scaled_inverse_chisqr(y_.size()-1, arma::var(y_));
        arma::cx_vec alpha_roots = ARRoots(theta);
        double sigsqr = yvar / Variance(alpha_roots, ma_coefs, 1.0);
        
        // Get initial value of the measurement error scaling parameter by
        // drawing from its prior.
        
        double measerr_scale = RandGen.scaled_inverse_chisqr(measerr_dof_, 1.0);
        measerr_scale = std::min(measerr_scale, 1.99);
        measerr_scale = std::max(measerr_scale, 0.51);
        
        theta(0) = sqrt(yvar);
        theta(1) = measerr_scale;
        
        
        /****************
         *
         * USE TRUE VALUES FOR DEBUGGING
         *
         ****************/
        // True ZCARMA(5) process parameters
        double qpo_width[3] = {0.01, 0.01, 0.002};
        double qpo_cent[2] = {0.2, 0.02};
        double sigmay = 2.3;
        measerr_scale = 1.0;
        double kappa = 0.7;

        // Create the parameter vector, theta
        theta(0) = log(sigmay);
        theta(1) = measerr_scale;
        for (int i=0; i<p_/2; i++) {
            theta(2+2*i) = log(qpo_cent[i]);
            theta(3+2*i) = log(qpo_width[i]);
        }
        // p is odd, so add in additional value of lorentz_width
        theta(p_+1) = log(qpo_width[p_/2]);

        kappa_low_ = 0.9 * kappa;
        kappa_high_ = 1.1 * kappa;
        
        double kappa_norm = (kappa - kappa_low_) / (kappa_high_ - kappa_low_);
        theta(p_+2) = logit(kappa_norm);

        
        // set the Kalman filter parameters
        pKFilter_->SetSigsqr(sigsqr);
        pKFilter_->SetOmega(ExtractAR(theta));
        pKFilter_->SetMA(ma_coefs);
        arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
        pKFilter_->SetTimeSeriesErr(proposed_yerr);
        
        // run the kalman filter
        pKFilter_->Filter();
        
        double logpost = LogDensity(theta);
        good_initials = arma::is_finite(logpost);
    } // continue loop until the starting values give us a finite posterior
    
    return theta;
}

// extract the moving average coefficients from the parameter vector
arma::vec ZCARMA::ExtractMA(arma::vec theta)
{
    double kappa_normed = inv_logit(theta(2 + p_));
    double kappa = (kappa_high_ - kappa_low_) * kappa_normed + kappa_low_;
    // Set the moving average terms
    arma::vec ma_coefs(p_);
	ma_coefs(0) = 1.0;
	for (int i=1; i<p_; i++) {
		ma_coefs(i) = boost::math::binomial_coefficient<double>(p_-1, i) / pow(kappa,i);
	}
    return ma_coefs;
}

/*********************************************************************
                                FUNCTIONS
 ********************************************************************/

double logit(double x) { return log(x / (1.0 - x)); }
double inv_logit(double x) { return exp(x) / (1.0 + exp(x)); }

// Check that all of the roots are unique to within a specified fractional
// tolerance.
bool unique_roots(arma::cx_vec roots, double tolerance)
{
    // Initialize the smallest fractional difference
    double min_frac_diff = 100.0 * tolerance;

    int p = roots.n_elem;
    
    for (int i=0; i<(p-1); i++) {
        for (int j=i+1; j<p; j++) {
            // Calculate fractional difference between roots(i) and roots(j)
            double frac_diff = std::abs( (roots(i) - roots(j)) / 
                                        (roots(i) + roots(j)) );
            if (frac_diff < min_frac_diff) {
                // Found new minimum fractional difference, record it
                min_frac_diff = frac_diff;
            }
        }
    }
    
    // Test if the roots unique to within the specified tolerance
    bool unique = (min_frac_diff > tolerance);
    
    return unique;
}
    
// Return the coefficients of a polynomial given its roots. The polynomial
// is assumed to be of the form:
//
//      p(x) = x^n + c_1 * x^{n-1} + ... + c_{n-1} * x + c_n
//
// where {c_i ; i=1,...,n} are the coefficients. Note that this function
// returns a (n+1)-element column vector, where c_0 = 1.0.

arma::vec polycoefs(arma::cx_vec roots)
{    
    arma::cx_vec coefs(roots.n_elem+1);
    coefs.zeros(); // Initialize all values to zero
    
    coefs(0) = 1.0; // Coefficient for highest order term is set to one
    
    for (int i=0; i<roots.n_elem; i++) {
        // Calculate the coefficients using a recursion formula
        coefs(arma::span(1,i+1)) = coefs(arma::span(1,i+1)) - roots(i) * coefs(arma::span(0,i));
    }
    
    // The coefficients must be real, so only return the real part
    return arma::real(coefs);
}
