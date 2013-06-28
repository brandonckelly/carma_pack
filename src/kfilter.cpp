//
//  kfilter.cpp
//  carma_pack
//
//  Created by Brandon Kelly on 6/27/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#include <random.hpp>
#include "include/kfilter.hpp"

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

// Reset the Kalman Filter for a CAR(1) process
void KalmanFilter1::Reset() {
    
    mean_(0) = 0.0;
    var_(0) = sigsqr_ / (2.0 * omega_) + yerr_(0) * yerr_(0);
    current_index_ = 1;
}

// Perform one iteration of the Kalman Filter for a CAR(1) process to update it
void KalmanFilter1::Update() {
    
    double rho, var_ratio, previous_var;
    rho = exp(-1.0 * omega_ * dt_(current_index_-1));
    previous_var = var_(current_index_-1) - yerr_(current_index_-1) * yerr_(current_index_-1);
    var_ratio = previous_var / var_(current_index_-1);
		
    // Update the Kalman filter mean
    mean_(current_index_) = rho * mean_(current_index_-1) +
        rho * var_ratio * (y_(current_index_-1) - mean_(current_index_-1));
		
    // Update the Kalman filter variance
    var_(current_index_) = sigsqr_ / (2.0 * omega_) * (1.0 - rho * rho) +
        rho * rho * previous_var * (1.0 - var_ratio);
    
    // add in contribution to variance from measurement errors
    var_(current_index_) += yerr_(current_index_) * yerr_(current_index_);
}

// Return the predicted value and its variance at time assuming a CAR(1) process
std::pair<double, double> KalmanFilter1::Predict(double time) {
    double rho, var_ratio, previous_var;
    double back_mean, back_var, forward_mean, forward_var;
    unsigned int ny = y_.n_elem;
    
    if (time < time_(0)) {
        // backcast the value of the time series
        double dt = time_(0) - time;
        back_mean = 0.0;
        back_var = sigsqr_ / (2.0 * omega_);
        rho = exp(-dt * omega_);
        forward_mean = y_(0) / rho;
        forward_var = var_(0) / rho / rho;
    } else if (time > time_(ny-1)) {
        // forecast the value of the time series
        double dt = time - time_(ny-1);
        rho = exp(-dt * omega_);
        previous_var = var_(ny-1) - yerr_(ny-1) * yerr_(ny-1);
        var_ratio = previous_var / var_(ny-1);
        back_mean = rho * mean_(ny-1) + rho * var_ratio * (y_(ny-1) - mean_(ny-1));
        back_var = sigsqr_ / (2.0 * omega_) * (1.0 - rho * rho) + rho * rho * previous_var * (1.0 - var_ratio);
        forward_mean = 0.0;
        forward_var = 1E300;
    } else {
        // interpolate the value of the time series
        double time_i = time_(0);
        unsigned int i = 0;
        while (time > time_i) {
            // find the index where time_ > time for the first time
            i++;
            time_i = time_(i);
        }
        double dt = time - time_(i-1);
        rho = exp(-dt * omega_);
        previous_var = var_(i-1) - yerr_(i-1) * yerr_(i-1);
        var_ratio = previous_var / var_(i-1);
        back_mean = rho * mean_(i-1) + rho * var_ratio * (y_(i-1) - mean_(i-1));
        back_var = sigsqr_ / (2.0 * omega_) * (1.0 - rho * rho) + rho * rho * previous_var * (1.0 - var_ratio);
        dt = time_i - time;
        rho = exp(-dt * omega_);
        forward_mean = y_(i) / rho;
        forward_var = var_(i) / rho / rho;
    }
    
    double ypredict_var = 1.0 / (1.0 / back_var + 1.0 / forward_var);
    double ypredict_mean = ypredict_var * (back_mean / back_var + forward_mean / forward_var);
    
    std::pair<double, double> ypredict(ypredict_mean, ypredict_var);
    return ypredict;
}

// Return a simulated time series conditional on the measured time series, assuming the CAR(1) model.
arma::vec KalmanFilter1::Simulate(arma::vec time) {
    unsigned int ntime = time.n_elem;
    arma::vec ysimulated(ntime);
    for (int i=0; i<ntime; i++) {
        std::pair<double, double> ypredict = Predict(time(i));
        double ymean = ypredict.first;
        double ysigma = sqrt(ypredict.second);
        ysimulated(i) = RandGen.normal(ymean, ysigma);
    }
    return ysimulated;
}

// Calculate the roots of the AR(p) polynomial from the PSD parameters
arma::cx_vec KalmanFilterp::ARRoots(arma::vec omega) {
    
    arma::cx_vec ar_roots;
    int p = omega.n_elem;
    
    // Construct the complex vector of roots of the characteristic polynomial:
    // alpha(s) = s^p + alpha_1 s^{p-1} + ... + alpha_{p-1} s + alpha_p
    for (int i=0; i<p/2; i++) {
        double lorentz_cent = omega(2*i); // PSD is a sum of Lorentzian functions
        double lorentz_width = omega(2*i+1);
        ar_roots(2*i) = std::complex<double> (-lorentz_width,lorentz_cent);
        ar_roots(2*i+1) = std::conj(ar_roots(2*i));
    }
	
    if ((p % 2) == 1) {
        // p is odd, so add in additional low-frequency component
        double lorentz_width = omega(p+1);
        ar_roots(p-1) = std::complex<double> (-lorentz_width, 0.0);
    }
    
    ar_roots *= 2.0 * arma::datum::pi;
    
    return ar_roots;
}

// Reset the Kalman Filter for a CARMA(p,q) process
void KalmanFilterp::Reset() {
    
}

// Perform one iteration of the Kalman Filter for a CARMA(p,q) process to update it
void KalmanFilterp::Update() {
    
}

// Predict the time series at the input time given the measured time series, assuming a CARMA(p,q) process
std::pair<double, double> KalmanFilterp::Predict(double time) {
    
    std::pair<double, double> ypredict;
    
    return ypredict;
}

// Simulate a CARMA(p,q) process at the input time values given the measured time series
arma::vec KalmanFilterp::Simulate(arma::vec time) {
    
    arma::vec ysimulated;
    
    return ysimulated;
}