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

// Armadillo includes
#include <armadillo>

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
	car1_stdev_start = RandGen.scaled_inverse_chisqr(y_.n_elem-1, arma::var(y_));
	car1_stdev_start = sqrt(car1_stdev_start);
    
    // Get initial value of the time series mean
    double mu = RandGen.normal(arma::mean(y_), car1_stdev_start / y_.n_elem);

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
	
	arma::vec theta(4);
	
	theta << car1_stdev_start << measerr_scale << mu << log_omega_start << arma::endr;
	
	// Initialize the Kalman filter
    pKFilter_->SetOmega(exp(log_omega_start));
    pKFilter_->SetSigsqr(sigma * sigma);
    arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
    pKFilter_->SetTimeSeriesErr(proposed_yerr);
    arma::vec ycent = y_ - mu;
    pKFilter_->SetTimeSeries(ycent);
    pKFilter_->Filter();
	
	return theta;
}

arma::vec CAR1::SetStartingValue(arma::vec init)
{
   if (init.n_elem != 4) {
      std::cout << "WARNING: initial guess not length 4, initializing with prior" << std::endl;
      return StartingValue();
   }

   double logpost = LogDensity(init);
   bool good_initials = arma::is_finite(logpost);
   if (good_initials == false) {
      std::cout << "WARNING: initial guess yields non-finite likelihood, initializing with prior" << std::endl;
      return StartingValue();
   }
      
   double mu, log_omega_start, car1_stdev_start, sigma, measerr_scale;
   car1_stdev_start = init[0];
   measerr_scale    = init[1];
   mu               = init[2];
   log_omega_start  = init[3];
   sigma            = car1_stdev_start * sqrt(2.0 * exp(log_omega_start));

   pKFilter_->SetOmega(exp(log_omega_start));
   pKFilter_->SetSigsqr(sigma * sigma);
   arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
   pKFilter_->SetTimeSeriesErr(proposed_yerr);
   arma::vec ycent = y_ - mu;
   pKFilter_->SetTimeSeries(ycent);
   pKFilter_->Filter();

   return init;
}


bool CAR1::CheckPriorBounds(arma::vec theta)
{
    double ysigma = theta(0);
    double measerr_scale = theta(1);
    double omega = exp(theta(3));
    
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
        // alpha(s) decomposed into its quadratic terms:
        //   alpha(s) = (quad_term1 + quad_term2 * s + s^2) * ...
        double quad_term1 = exp(theta(3+2*i));
        double quad_term2 = exp(theta(3+2*i+1));

        double discriminant = quad_term2 * quad_term2 - 4.0 * quad_term1;
        
        if (discriminant > 0) {
            // two real roots
            double root1 = -0.5 * (quad_term2 + sqrt(discriminant));
            double root2 = -0.5 * (quad_term2 - sqrt(discriminant));
            ar_roots(2*i) = std::complex<double> (root1, 0.0);
            ar_roots(2*i+1) = std::complex<double> (root2, 0.0);
        } else {
            double real_part = -0.5 * quad_term2;
            double imag_part = -0.5 * sqrt(-discriminant);
            ar_roots(2*i) = std::complex<double> (real_part, imag_part);
            ar_roots(2*i+1) = std::complex<double> (real_part, -imag_part);
        }
    }
	
    if ((p_ % 2) == 1) {
        // p is odd, so add in additional low-frequency component
        double real_root = -exp(theta(3+p_-1));
        ar_roots(p_-1) = std::complex<double> (real_root, 0.0);
    }
        
    return ar_roots;
}

// Return the starting value and set log_posterior_
arma::vec CARp::StartingValue()
{
    // Create the parameter vector, theta
    arma::vec theta(p_+3);
    
    bool good_initials = false;
    int iguess_count = 0;
    while (!good_initials) {

        arma::vec loga = StartingAR();
        for (int i=0; i<p_; i++) {
            theta(3+i) = loga(i);
        }
        
        // Initial guess for model standard deviation is randomly distributed
        // around measured standard deviation of the time series
        double yvar = RandGen.scaled_inverse_chisqr(y_.n_elem-1, arma::var(y_));
        
        // Get initial value of the time series mean
        double mu = RandGen.normal(arma::mean(y_), sqrt(yvar) / y_.n_elem);

        arma::cx_vec alpha_roots = ARRoots(theta);
        double sigsqr = yvar / Variance(alpha_roots, ma_coefs_, 1.0);
        
        // Get initial value of the measurement error scaling parameter by
        // drawing from its prior.
        double measerr_scale = RandGen.scaled_inverse_chisqr(measerr_dof_, 1.0);
        measerr_scale = std::min(measerr_scale, 1.99);
        measerr_scale = std::max(measerr_scale, 0.51);
        
        theta(0) = sqrt(yvar);
        theta(1) = measerr_scale;
        theta(2) = mu;
        
        // set the Kalman filter parameters
        pKFilter_->SetSigsqr(sigsqr);
        pKFilter_->SetOmega(ExtractAR(theta));
        arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
        pKFilter_->SetTimeSeriesErr(proposed_yerr);
        arma::vec ycent = y_ - mu;
        pKFilter_->SetTimeSeries(ycent);
        
        // run the kalman filter
        pKFilter_->Filter();
        
        double logpost = LogDensity(theta);
        good_initials = arma::is_finite(logpost);
        
        iguess_count++;
        if (iguess_count > 200) {
            std::cout << "Tried 200 initial guesses, still trying..." << std::endl;
        }
    } // continue loop until the starting values give us a finite posterior
    
    return theta;
}

// Return the starting value and set log_posterior_
arma::vec CARp::SetStartingValue(arma::vec init)
{

   if (init.n_elem != (p_+3)) {
      std::cout << "WARNING: initial guess wrong length, initializing with prior" << std::endl;
      return StartingValue();
   }

   double logpost = LogDensity(init);
   bool good_initials = arma::is_finite(logpost);
   if (good_initials == false) {
      std::cout << "WARNING: initial guess yields non-finite likelihood, initializing with prior" << std::endl;
      return StartingValue();
   }

   double yvar = init(0)*init(0);
   double measerr_scale = init(1);
   double mu = init(2);

   arma::cx_vec alpha_roots = ARRoots(init);
   double sigsqr = yvar / Variance(alpha_roots, ma_coefs_, 1.0);
   
   // set the Kalman filter parameters
   pKFilter_->SetSigsqr(sigsqr);
   pKFilter_->SetOmega(ExtractAR(init));
   arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
   pKFilter_->SetTimeSeriesErr(proposed_yerr);
   arma::vec ycent = y_ - mu;
   pKFilter_->SetTimeSeries(ycent);
   pKFilter_->Filter();

   return init;
}

// return the starting values for the autoregressive polynomial paramters
arma::vec CARp::StartingAR() {
    double min_freq = 1.0 / (time_.max() - time_.min());
    
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

    arma::vec loga(p_);
    
    // convert the PSD lorentzian parameters to quadratic terms in the AR polynomial decomposition
    for (int i=0; i<p_/2; i++) {
        double real_part = -2.0 * arma::datum::pi * lorentz_width(i);
        double imag_part = 2.0 * arma::datum::pi * lorentz_cent(i);
        double quad_term1 = real_part * real_part + imag_part * imag_part;
        double quad_term2 = -2.0 * real_part;
        loga(2*i) = log(quad_term1);
        loga(1+2*i) = log(quad_term2);
    }
    if ((p_ % 2) == 1) {
        // p is odd, so add in additional value of lorentz_width
        double real_part = -2.0 * arma::datum::pi * lorentz_width(p_/2);
        loga(p_-1) = log(-real_part);
    }
    return loga;
}

// check prior bounds
bool CARp::CheckPriorBounds(arma::vec theta)
{
    if (ignore_prior_) {return true;}

    double ysigma = theta(0);
    double measerr_scale = theta(1);
    arma::cx_vec ar_roots = ExtractAR(theta);
    
    arma::vec lorentz_cent = arma::abs(arma::imag(ar_roots)) / 2.0 / arma::datum::pi;
    arma::vec lorentz_width = -arma::real(ar_roots) / 2.0 / arma::datum::pi;
    
    // Find the set of Frequencies satisfying the prior bounds
    arma::uvec valid_frequencies1 = arma::find(lorentz_cent < max_freq_);
	arma::uvec valid_frequencies2 = arma::find(lorentz_width < max_freq_);
    arma::uvec valid_frequencies3 = arma::find(lorentz_width > min_freq_);
    
    double tol = 1e-4;
    bool prior_satisfied = unique_roots(ar_roots, tol); // are the roots unique?
    
    if ( (valid_frequencies1.n_elem != lorentz_cent.n_elem) ||
        (valid_frequencies2.n_elem != lorentz_width.n_elem) ||
        (valid_frequencies3.n_elem != lorentz_width.n_elem) ||
        (ysigma > max_stdev_) || (ysigma < 0) ||
        (measerr_scale < 0.5) || (measerr_scale > 2.0) ) {
        // Value are outside of prior bounds
        
//        std::cout << "prior bounds violated" << std::endl;
//        std::cout << "# of valid centroids: " << valid_frequencies1.n_elem << std::endl;
//        std::cout << "# of valid widths: " << valid_frequencies2.n_elem << std::endl;
//        std::cout << "max_freq: " << max_freq_ << std::endl;
//        std::cout << "min_freq: " << min_freq_ << std::endl;
//        lorentz_cent.print("centroid");
//        lorentz_width.print("width");
//        std::cout << "ysigma: " << ysigma << ", max_stdev: " << max_stdev_ << std::endl;
//        std::cout << "measerr_scale: " << measerr_scale << std::endl;
//        
        prior_satisfied = false;
    }
    
    if (order_lorentzians_) {
        // Make sure the Lorentzian centroids are still in decreasing order
        for (int i=1; i<lorentz_cent.n_elem; i++) {
            double lorentz_cent_difference = lorentz_cent(i) - lorentz_cent(i-1);
            if (lorentz_cent_difference > 1e-8) {
                // Lorentzians are not in decreasing order, reject this proposal
                prior_satisfied = false;
            }
        }
    }

//    // Make sure Lorentzian widths are greater than minimum frequency for those Lorentzians with centroids
//    // less than the minimum frequency.
//    for (int i=0; i<lorentz_cent.n_elem; i++) {
//        if (lorentz_cent(i) < min_freq_) {
//            if (lorentz_width(i) < min_freq_) {
//                prior_satisfied = false;
//            }
//        }
//    }    
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
    // Create the parameter vector, theta
    arma::vec theta(p_+q_+3);
    
    bool good_initials = false;
    while (!good_initials) {
        
        // Initial guess for model standard deviation is randomly distributed
        // around measured standard deviation of the time series
        arma::vec loga = StartingAR();
        for (int i=0; i<p_; i++) {
            theta(3+i) = loga(i);
        }
        
        arma::vec log_ma_quad = StartingMA();
        theta(arma::span(p_+3,theta.n_elem-1)) = log_ma_quad;
        arma::vec ma_coefs = ExtractMA(theta);

        // Initial guess for model standard deviation is randomly distributed
        // around measured standard deviation of the time series
        double yvar = RandGen.scaled_inverse_chisqr(y_.n_elem-1, arma::var(y_));
        
        // Get initial value of the time series mean
        double mu = RandGen.normal(arma::mean(y_), sqrt(yvar) / y_.n_elem);
        
        arma::cx_vec alpha_roots = ARRoots(theta);
        double sigsqr = yvar / Variance(alpha_roots, ma_coefs, 1.0);
        good_initials = arma::is_finite(sigsqr);
        // Don't run kalman filter if we're already bad_initials

        if (good_initials) {
            // Get initial value of the measurement error scaling parameter by
            // drawing from its prior.
        
            double measerr_scale = RandGen.scaled_inverse_chisqr(measerr_dof_, 1.0);
            measerr_scale = std::min(measerr_scale, 1.99);
            measerr_scale = std::max(measerr_scale, 0.51);
        
            theta(0) = sqrt(yvar);
            theta(1) = measerr_scale;
            theta(2) = mu;
                
            // set the Kalman filter parameters
            pKFilter_->SetSigsqr(sigsqr);
            pKFilter_->SetOmega(ExtractAR(theta));
            pKFilter_->SetMA(ExtractMA(theta));
            arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
            pKFilter_->SetTimeSeriesErr(proposed_yerr);
            arma::vec ycent = y_ - mu;
            pKFilter_->SetTimeSeries(ycent);
        
            // run the kalman filter
            pKFilter_->Filter();
        
            double logpost = LogDensity(theta);
            good_initials = arma::is_finite(logpost);
        }
    } // continue loop until the starting values give us a finite posterior
    
    return theta;
}

arma::vec CARMA::SetStartingValue(arma::vec init)
{
   if (init.n_elem != (p_+q_+3)) {
      std::cout << "WARNING: initial guess wrong length, initializing with prior" << std::endl;
      return StartingValue();
   }

   double logpost = LogDensity(init);
   bool good_initials = arma::is_finite(logpost);
   if (good_initials == false) {
      std::cout << "WARNING: initial guess yields non-finite likelihood, initializing with prior" << std::endl;
      return StartingValue();
   }
   
   double yvar          = init(0)*init(0);        
   double measerr_scale = init(1);
   double mu            = init(2);
   
   arma::cx_vec alpha_roots = ARRoots(init);
   arma::vec ma_coefs       = ExtractMA(init);
   double sigsqr            = yvar / Variance(alpha_roots, ma_coefs, 1.0);
   
   // set the Kalman filter parameters
   pKFilter_->SetSigsqr(sigsqr);
   pKFilter_->SetOmega(ExtractAR(init));
   pKFilter_->SetMA(ExtractMA(init));
   arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
   pKFilter_->SetTimeSeriesErr(proposed_yerr);
   arma::vec ycent = y_ - mu;
   pKFilter_->SetTimeSeries(ycent);
   pKFilter_->Filter();
   
   return init;
}

// get initial guess for the moving average polynomial coefficients
arma::vec CARMA::StartingMA() {
    arma::vec ma_quad(q_);
    ma_quad.randn();
    return arma::abs(ma_quad);
}

// extract the moving-average coefficients from the CARMA parameter vector
arma::vec CARMA::ExtractMA(arma::vec theta)
{
    arma::cx_vec ma_roots(q_);
    
    // Construct the complex vector of roots of the characteristic polynomial:
    // alpha(s) = s^p + alpha_1 s^{p-1} + ... + alpha_{p-1} s + alpha_p
    for (int i=0; i<q_/2; i++) {
        // alpha(s) decomposed into its quadratic terms:
        //   alpha(s) = (quad_term1 + quad_term2 * s + s^2) * ...
        double quad_term1 = exp(theta(3+p_+2*i));
        double quad_term2 = exp(theta(3+p_+2*i+1));
        
        double discriminant = quad_term2 * quad_term2 - 4.0 * quad_term1;
        
        if (discriminant > 0) {
            // two real roots
            double root1 = -0.5 * (quad_term2 + sqrt(discriminant));
            double root2 = -0.5 * (quad_term2 - sqrt(discriminant));
            ma_roots(2*i) = std::complex<double> (root1, 0.0);
            ma_roots(2*i+1) = std::complex<double> (root2, 0.0);
        } else {
            double real_part = -0.5 * quad_term2;
            double imag_part = -0.5 * sqrt(-discriminant);
            ma_roots(2*i) = std::complex<double> (real_part, imag_part);
            ma_roots(2*i+1) = std::complex<double> (real_part, -imag_part);
        }
    }
	
    if ((q_ % 2) == 1) {
        // p is odd, so add in additional low-frequency component
        double real_root = -exp(theta(3+p_+q_-1));
        ma_roots(q_-1) = std::complex<double> (real_root, 0.0);
    }

    // calculate the coefficients of the polynomial
    //
    //    p(x) = x^q + c_1 * x^{q-1} + ... + c_{q-1} * x + c_q
    //
    // from it roots. note that poly_coefs[0] = 1.0 = c_0.
    arma::vec poly_coefs = polycoefs(ma_roots);

    // convert coefficients to MA polynomial representation:
    //
    //   beta(s) = beta_q * x^q + beta_{q-1} * x^{q-1} + ... + beta_1 x + beta_0,
    //
    // where beta_0 = 1.0.
    //
    poly_coefs = poly_coefs / poly_coefs(q_); // standardize so c_q = 1 instead of c_0;
    arma::vec ma_coefs = arma::zeros(p_);
    
    // poly_coefs[0]   poly_coefs[1]   ...   poly_coefs[q] = 1.0
    //    ||                ||                     ||
    // ma_coefs[q]    ma_coefs[q-1]    ...    ma_coefs[0]
    for (int i=0; i<q_+1; i++) {
        ma_coefs(i) = poly_coefs(q_-i);
    }

    return ma_coefs;
}

/*******************************************************************
                        METHODS OF ZCARMA CLASS
 *******************************************************************/

arma::vec ZCARMA::StartingValue()
{
    // Create the parameter vector, theta
    arma::vec theta(p_+4);
    
    bool good_initials = false;
    while (!good_initials) {
        
        // Initial guess for model standard deviation is randomly distributed
        // around measured standard deviation of the time series
        arma::vec loga = StartingAR();
        for (int i=0; i<p_; i++) {
            theta(3+i) = loga(i);
        }
        
        theta(3+p_) = logit(StartingKappa());
        
        // compute the coefficients of the MA polynomial
        arma::vec ma_coefs = ExtractMA(theta);
        
        // Initial guess for model standard deviation is randomly distributed
        // around measured standard deviation of the time series
        double yvar = RandGen.scaled_inverse_chisqr(y_.n_elem-1, arma::var(y_));
        
        // Get initial value of the time series mean
        double mu = RandGen.normal(arma::mean(y_), sqrt(yvar) / y_.n_elem);
        
        arma::cx_vec alpha_roots = ARRoots(theta);
        double sigsqr = yvar / Variance(alpha_roots, ma_coefs, 1.0);
        
        // Get initial value of the measurement error scaling parameter by
        // drawing from its prior.
        
        double measerr_scale = RandGen.scaled_inverse_chisqr(measerr_dof_, 1.0);
        measerr_scale = std::min(measerr_scale, 1.99);
        measerr_scale = std::max(measerr_scale, 0.51);
        
        theta(0) = sqrt(yvar);
        theta(1) = measerr_scale;
        theta(2) = mu;
        
        // set the Kalman filter parameters
        pKFilter_->SetSigsqr(sigsqr);
        pKFilter_->SetOmega(ExtractAR(theta));
        pKFilter_->SetMA(ma_coefs);
        arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
        pKFilter_->SetTimeSeriesErr(proposed_yerr);
        arma::vec ycent = y_ - mu;
        pKFilter_->SetTimeSeries(ycent);
        
        // run the kalman filter
        pKFilter_->Filter();
        
        double logpost = LogDensity(theta);
        good_initials = arma::is_finite(logpost);
    } // continue loop until the starting values give us a finite posterior
    
    return theta;
}

arma::vec ZCARMA::SetStartingValue(arma::vec init)
{
   if (init.n_elem != (p_+4)) {
      std::cout << "WARNING: initial guess wrong length, initializing with prior" << std::endl;
      return StartingValue();
   }

   double logpost = LogDensity(init);
   bool good_initials = arma::is_finite(logpost);
   if (good_initials == false) {
      std::cout << "WARNING: initial guess yields non-finite likelihood, initializing with prior" << std::endl;
      return StartingValue();
   }

   double yvar          = init(0)*init(0);        
   double measerr_scale = init(1);
   double mu            = init(2);

   arma::cx_vec alpha_roots = ARRoots(init);
   arma::vec ma_coefs = ExtractMA(init);
   double sigsqr = yvar / Variance(alpha_roots, ma_coefs, 1.0);

   pKFilter_->SetSigsqr(sigsqr);
   pKFilter_->SetOmega(ExtractAR(init));
   pKFilter_->SetMA(ma_coefs);
   arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
   pKFilter_->SetTimeSeriesErr(proposed_yerr);
   arma::vec ycent = y_ - mu;
   pKFilter_->SetTimeSeries(ycent);
   pKFilter_->Filter();

   return init;
}

// get initial guess for the moving average polynomial coefficients, parameterized by kappa
double ZCARMA::StartingKappa() {
    double kappa_normed = RandGen.uniform();
    return kappa_normed;
}

// extract the moving average coefficients from the parameter vector
arma::vec ZCARMA::ExtractMA(arma::vec theta)
{
    double kappa_normed = inv_logit(theta(3 + p_));
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
