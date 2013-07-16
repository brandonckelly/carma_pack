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
    
    current_index_++;
}

// Initialize the coefficients used for interpolation and backcasting assuming a CAR(1) process
void KalmanFilter1::InitializeCoefs(double time, unsigned int itime, double ymean, double yvar) {
    yconst_ = 0.0;
    yslope_ = exp(-std::abs(time_(itime) - time) * omega_);
    current_index_ = itime + 1;
}

// Update the coefficients used for interpolation and backcasting, assuming a CAR(1) process
void KalmanFilter1::UpdateCoefs() {
    double rho = exp(-1.0 * dt_(current_index_-1) * omega_);
    double previous_var = var_(current_index_-1) - yerr_(current_index_-1) * yerr_(current_index_-1);
    double var_ratio = previous_var / var_(current_index_-1);
    
    yslope_ *= rho * (1.0 - var_ratio);
    yconst_ = yconst_ * rho * (1.0 - var_ratio) + rho * var_ratio * y_[current_index_-1];
    var_(current_index_) = sigsqr_ / (2.0 * omega_) * (1.0 - rho * rho) +
        rho * rho * previous_var * (1.0 - var_ratio) + yerr_(current_index_) * yerr_(current_index_);
}

// Return the predicted value and its variance at time assuming a CAR(1) process
std::pair<double, double> KalmanFilter1::Predict(double time) {
    double rho, var_ratio, previous_var;
    double ypredict_mean, ypredict_var, yprecision;
    
    unsigned int ipredict = 0;
    while (time > time_(ipredict)) {
        // find the index where time_ > time for the first time
        ipredict++;
        if (ipredict == time_.n_elem) {
            // time is greater than last element of time_, so do forecasting
            break;
        }
    }
    
    // Run the Kalman filter up to the point time_[ipredict-1]
    Reset();
    for (int i=1; i<ipredict; i++) {
        Update();
    }
        
    if (ipredict == 0) {
        // backcasting, so initialize the conditional mean and variance to the stationary values
        ypredict_mean = 0.0;
        ypredict_var = sigsqr_ / (2.0 * omega_);
    } else {
        // predict the value of the time series at time, given the earlier values
        double dt = time - time_[ipredict-1];
        rho = exp(-dt * omega_);
        previous_var = var_(ipredict-1) - yerr_(ipredict-1) * yerr_(ipredict-1);
        var_ratio = previous_var / var_(ipredict-1);
		
        // Update the Kalman filter mean
        mean_(current_index_) = rho * mean_(current_index_-1) +
        rho * var_ratio * (y_(current_index_-1) - mean_(current_index_-1));
		
        // Update the Kalman filter variance
        var_(current_index_) = sigsqr_ / (2.0 * omega_) * (1.0 - rho * rho) +
        rho * rho * previous_var * (1.0 - var_ratio);
        
        // initialize the conditional mean and variance
        ypredict_mean = rho * mean_(ipredict-1) + rho * var_ratio * (y_(ipredict-1) - mean_(ipredict-1));
        ypredict_var = sigsqr_ / (2.0 * omega_) * (1.0 - rho * rho) + rho * rho * previous_var * (1.0 - var_ratio);
    }
    
    if (ipredict == time_.n_elem) {
        // Forecasting, so we're done: no need to run interpolation steps
        std::pair<double, double> ypredict(ypredict_mean, ypredict_var);
        return ypredict;
    }
    
    yprecision = 1.0 / ypredict_var;
    
    // Either backcasting or interpolating, so need to calculate coefficients of linear filter as a function of
    // the predicted time series value, then update the running conditional mean and variance of the predicted
    // time series value
    
    InitializeCoefs(time, ipredict, ypredict_mean, ypredict_var);
    
    yprecision += yslope_ * yslope_ / var_(ipredict);
    ypredict_mean += yslope_ * (y_(ipredict) - yconst_) / var_(ipredict);
    
    for (int i=ipredict+1; i<time_.n_elem; i++) {
        UpdateCoefs();
        yprecision += yslope_ * yslope_ / var_(i);
        ypredict_mean += yslope_ * (y_(i) - yconst_) / var_(i);
    }
    
    ypredict_var = 1.0 / yprecision;
    ypredict_mean *= ypredict_var;

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

    int p = omega.n_elem;
    arma::cx_vec ar_roots(p);
    
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
        double lorentz_width = omega(p-1);
        ar_roots(p-1) = std::complex<double> (-lorentz_width, 0.0);
    }
    
    ar_roots *= 2.0 * arma::datum::pi;
    
    return ar_roots;
}

// Reset the Kalman Filter for a CARMA(p,q) process
void KalmanFilterp::Reset() {
    
    // Initialize the matrix of Eigenvectors. We will work with the state vector
	// in the space spanned by the Eigenvectors because in this space the state
	// transition matrix is diagonal, so the calculation of the matrix exponential
	// is fast.
    arma::cx_mat EigenMat(p_,p_);
	EigenMat.row(0) = arma::ones<arma::cx_rowvec>(p_);
	EigenMat.row(1) = ar_roots_.st();
	for (int i=2; i<p_; i++) {
		EigenMat.row(i) = strans(arma::pow(ar_roots_, i));
	}
    
	// Input vector under original state space representation
	arma::cx_vec Rvector = arma::zeros<arma::cx_vec>(p_);
	Rvector(p_-1) = 1.0;
    
	// Transform the input vector to the rotated state space representation.
	// The notation R and J comes from Belcher et al. (1994).
	arma::cx_vec Jvector(p_);
	Jvector = arma::solve(EigenMat, Rvector);
	
	// Transform the moving average coefficients to the space spanned by EigenMat.

    rotated_ma_coefs_ = ma_coefs_ * EigenMat;
	
	// Calculate the stationary covariance matrix of the state vector.
	for (int i=0; i<p_; i++) {
		for (int j=i; j<p_; j++) {
			// Only fill in upper triangle of StateVar because of symmetry
			StateVar_(i,j) = -sigsqr_ * Jvector(i) * std::conj(Jvector(j)) /
            (ar_roots_(i) + std::conj(ar_roots_(j)));
		}
	}
	StateVar_ = arma::symmatu(StateVar_); // StateVar is symmetric
	PredictionVar_ = StateVar_; // One-step state prediction error
	
	state_vector_.zeros(); // Initial state is set to zero
	
	// Initialize the Kalman mean and variance. These are the forecasted value
	// for the measured time series values and its variance, conditional on the
	// previous measurements
	mean_(0) = 0.0;
    var_(0) = std::real( arma::as_scalar(rotated_ma_coefs_ * StateVar_ * rotated_ma_coefs_.t()) );
    var_(0) += yerr_(0) * yerr_(0); // Add in measurement error contribution

	innovation_ = y_(0); // The innovation
    current_index_ = 1;
}

// Perform one iteration of the Kalman Filter for a CARMA(p,q) process to update it
void KalmanFilterp::Update() {
    // First compute the Kalman Gain
    kalman_gain_ = PredictionVar_ * rotated_ma_coefs_.t() / var_(current_index_-1);
    
    // Now update the state vector
    state_vector_ += kalman_gain_ * innovation_;
    
    // Update the state one-step prediction error variance
    PredictionVar_ -= var_(current_index_-1) * (kalman_gain_ * kalman_gain_.t());
    
    // Predict the next state
    rho_ = arma::exp(ar_roots_ * dt_(current_index_-1));
    state_vector_ = rho_ % state_vector_;
    
    // Update the predicted state variance matrix
    PredictionVar_ = (rho_ * rho_.t()) % (PredictionVar_ - StateVar_) + StateVar_;
    
    // Now predict the observation and its variance.
    mean_(current_index_) = std::real( arma::as_scalar(rotated_ma_coefs_ * state_vector_) );
    
    var_(current_index_) = std::real( arma::as_scalar(rotated_ma_coefs_ * PredictionVar_ * rotated_ma_coefs_.t()) );
    var_(current_index_) += yerr_(current_index_) * yerr_(current_index_); // Add in measurement error contribution
    
    // Finally, update the innovation
    innovation_ = y_(current_index_) - mean_(current_index_);
    current_index_++;
}

// Predict the time series at the input time given the measured time series, assuming a CARMA(p,q) process
std::pair<double, double> KalmanFilterp::Predict(double time) {
    
    unsigned int ipredict = 0;
    while (time > time_(ipredict)) {
        // find the index where time_ > time for the first time
        ipredict++;
        if (ipredict == time_.n_elem) {
            // time is greater than last element of time_, so do forecasting
            break;
        }
    }
    
    // Run the Kalman filter up to the point time_[ipredict-1]
    Reset();
    for (int i=1; i<ipredict; i++) {
        Update();
    }
    
    double ypredict_mean, ypredict_var, yprecision;
    
    if (ipredict == 0) {
        // backcasting, so initialize the conditional mean and variance to the stationary values
        ypredict_mean = 0.0;
        ypredict_var = std::real( arma::as_scalar(rotated_ma_coefs_ * StateVar_ * rotated_ma_coefs_.t()) );
    } else {
        // predict the value of the time series at time, given the earlier values
        kalman_gain_ = PredictionVar_ * rotated_ma_coefs_.t() / var_(ipredict-1);
        state_vector_ += kalman_gain_ * innovation_;
        PredictionVar_ -= var_(ipredict-1) * (kalman_gain_ * kalman_gain_.t());
        double dt = time - time_[ipredict-1];
        rho_ = arma::exp(ar_roots_ * dt);
        state_vector_ = rho_ % state_vector_;
        state_vector_ = state_vector_ % rho_;
        PredictionVar_ = (rho_ * rho_.t()) % (PredictionVar_ - StateVar_) + StateVar_;
        
        // initialize the conditional mean and variance
        ypredict_mean = std::real( arma::as_scalar(rotated_ma_coefs_ * state_vector_) );
        ypredict_var = std::real( arma::as_scalar(rotated_ma_coefs_ * PredictionVar_ * rotated_ma_coefs_.t()) );
    }

    if (ipredict == time_.n_elem) {
        // Forecasting, so we're done: no need to run interpolation steps
        std::pair<double, double> ypredict(ypredict_mean, ypredict_var);
        return ypredict;
    }

    yprecision = 1.0 / ypredict_var;

    // Either backcasting or interpolating, so need to calculate coefficients of linear filter as a function of
    // the predicted time series value, then update the running conditional mean and variance of the predicted
    // time series value
    
    InitializeCoefs(time, ipredict, ypredict_mean, ypredict_var);

    yprecision += yslope_ * yslope_ / var_(ipredict);
    ypredict_mean += yslope_ * (y_(ipredict) - yconst_) / var_(ipredict);
    
    for (int i=ipredict+1; i<time_.n_elem; i++) {
        UpdateCoefs();
        yprecision += yslope_ * yslope_ / var_(i);
        ypredict_mean += yslope_ * (y_(i) - yconst_) / var_(i);
    }
    
    ypredict_var = 1.0 / yprecision;
    ypredict_mean *= ypredict_var;
    
    std::pair<double, double> ypredict(ypredict_mean, ypredict_var);
    return ypredict;
}

// Initialize the coefficients needed for computing the Kalman Filter at future times as a function of
// the time series at time, where time_(itime-1) < time < time_(itime)
void KalmanFilterp::InitializeCoefs(double time, unsigned int itime, double ymean, double yvar) {
    
    kalman_gain_ = PredictionVar_ * rotated_ma_coefs_.t() / yvar;
    // initialize the coefficients for predicting the state vector at coefs(time_predict|time_predict)
    state_const_ = state_vector_ - kalman_gain_ * ymean;
    state_slope_ = kalman_gain_;
    // update the state one-step prediction error variance
    PredictionVar_ -= yvar * (kalman_gain_ * kalman_gain_.t());
    // coefs(time_predict|time_predict) --> coefs(time[i+1]|time_predict)
    double dt = time_[itime] - time;
    rho_ = arma::exp(ar_roots_ * dt);
    state_const_ = rho_ % state_const_;
    state_slope_ = rho_ % state_slope_;
    // update the predicted state covariance matrix
    PredictionVar_ = (rho_ * rho_.t()) % (PredictionVar_ - StateVar_) + StateVar_;
    // compute the coefficients for the linear filter at time[ipredict], and compute the variance in the predicted
    // y[ipredict]
    yconst_ = std::real( arma::as_scalar(rotated_ma_coefs_ * state_const_) );
    yslope_ = std::real( arma::as_scalar(rotated_ma_coefs_ * state_slope_) );
    var_[itime] = std::real( arma::as_scalar(rotated_ma_coefs_ * PredictionVar_ * rotated_ma_coefs_.t()) )
        + yerr_[itime] * yerr_[itime];
    current_index_ = itime + 1;
}

// Update the coefficients need for computing the Kalman Filter at future times as a function of the
// time series value at some earlier time
void KalmanFilterp::UpdateCoefs() {
    
    kalman_gain_ = PredictionVar_ * rotated_ma_coefs_.t() / var_[current_index_-1];
    // update the coefficients for predicting the state vector at coefs(i|i-1) --> coefs(i|i)
    state_const_ += kalman_gain_ * (y_[current_index_-1] - yconst_);
    state_slope_ -= kalman_gain_ * yslope_;
    // update the state one-step prediction error variance
    PredictionVar_ -= var_[current_index_-1] * (kalman_gain_ * kalman_gain_.t());
    // compute the one-step state prediction coefficients: coefs(i|i) --> coefs(i+1|i)
    rho_ = arma::exp(ar_roots_ * dt_[current_index_-1]);
    state_const_ = rho_ % state_const_;
    state_slope_ = rho_ % state_slope_;
    // update the predicted state covariance matrix
    PredictionVar_ = (rho_ * rho_.t()) % (PredictionVar_ - StateVar_) + StateVar_;
    // compute the coefficients for the linear filter at time[ipredict], and compute the variance in the predicted
    // y[ipredict]
    yconst_ = std::real( arma::as_scalar(rotated_ma_coefs_ * state_const_) );
    yslope_ = std::real( arma::as_scalar(rotated_ma_coefs_ * state_slope_) );
    var_[current_index_] = std::real( arma::as_scalar(rotated_ma_coefs_ * PredictionVar_ * rotated_ma_coefs_.t()) )
        + yerr_[current_index_] * yerr_[current_index_];
    current_index_++;
}

// Simulate a CARMA(p,q) process at the input time values given the measured time series
arma::vec KalmanFilterp::Simulate(arma::vec time) {
    
    // first save old values since we will overwrite them later
    arma::vec time0 = time_;
    arma::vec y0 = y_;
    arma::vec yerr0 = yerr_;
    
    arma::vec ysimulated(time.n_elem);
    
    time = arma::sort(time);
    unsigned int insert_idx = 0;
    for (int i=0; i<time.n_elem; i++) {
        // first simulate the value at time(i)
        std::pair<double, double> ypredict = Predict(time(i));
        ysimulated(i) = RandGen.normal(ypredict.first, sqrt(ypredict.second));
        // find the index where time_[insert_idx-1] < time(i) < time_[insert_idx]
        while (time_(insert_idx) < time(i)) {
            insert_idx++;
        }
        // insert the simulated value into the measured time series array. these values are used in subsequent
        // calls to Predict(time(i)).
        time_.insert_rows(insert_idx, time(i));
        y_.insert_rows(insert_idx, ysimulated(i));
        yerr_.insert_rows(insert_idx, 0.0);
    }
        
    // restore values of measured time series
    time_ = time0;
    y_ = y0;
    yerr_ = yerr0;
    
    return ysimulated;
}