//
//  kfilter.h
//  carma_pack
//
//  Created by Brandon Kelly on 6/27/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef __carma_pack__kfilter__
#define __carma_pack__kfilter__

#include <armadillo>
#include <utility>

/*
 Abstract base class for the Kalman Filter.
 */

template <class OmegaType>
class KalmanFilter {
public:
    // Constructor
    KalmanFilter(arma::vec& time, arma::vec& y, arma::vec& yerr, double sigsqr, OmegaType omega) :
    time_(time), y_(y), yerr_(yerr), sigsqr_(sigsqr), omega_(omega)
    {
        dt_ = time_(arma::span(1,time_.n_elem-1)) - time_(arma::span(0,time.n_elem-2));
        mean_.set_size(time.n_elem);
        var_.set_size(time.n_elem);
    }
    
    // Methods to set and get the data
    void SetTime(arma::vec& time) {
        time_ = time;
    }
    arma::vec GetTime() {
        return time_;
    }
    void SetTimeSeries(arma::vec& y) {
        y_ = y;
    }
    arma::vec GetTimeSeries() {
        return y_;
    }
    void SetTimeSeriesErr(arma::vec& yerr) {
        yerr_ = yerr;
    }
    arma::vec GetTimeSeriesErr() {
        return yerr_;
    }
    
    // Methods to set and get the parameter values
    void SetSigsqr(double sigsqr) {
        sigsqr_ = sigsqr;
    }
    double GetSigsqr() {
        return sigsqr_;
    }
    void SetOmega(OmegaType omega) {
        omega_ = omega;
    }
    OmegaType GetOmega() {
        return omega_;
    }
    
    // Methods to get the kalman filter mean and variance
    arma::vec GetMean() {
        return mean_;
    }
    arma::vec GetVariance() {
        return var_;
    }
    
    /* 
     Methods to perform the Kalman Filter operations 
     */
    virtual void Reset() = 0;
    virtual void Update() = 0;
    
    void Filter() {
        // Run the Kalman Filter
        Reset();
        for (int i=1; i<time_.n_elem; i++) {
            Update();
        }
    }
    
    virtual std::pair<double, double> Predict(double time) = 0;
    virtual arma::vec Simulate(arma::vec time) = 0;
    
protected:
    // Data
    arma::vec& time_;
    arma::vec dt_;
    arma::vec& y_;
    arma::vec& yerr_;
    // Kalman mean and variance
    arma::vec mean_;
    arma::vec var_;
    // The Kalman Filter parameters
    double sigsqr_;
    OmegaType omega_;
    unsigned int current_index_;
};

/*
 Class to perform the Kalman Filter and related operations for a zero-mean CAR(1) process (see below).
 This class will calculate the Kalman Filter, needed to calculate the log-likelihood in the CAR1
 parameter class. It will also provide interpolated, extrapolated, and simulated values given a
 CAR(1) model and a measured time series.
 */

class KalmanFilter1 : public KalmanFilter<double> {
public:
    // Constructor
    KalmanFilter1(arma::vec& time, arma::vec& y, arma::vec& yerr, double sigsqr, double omega) :
        KalmanFilter(time, y, yerr, sigsqr, omega) {}
    
    // Methods to perform the Kalman Filter operations
    virtual void Reset();
    virtual void Update();
    virtual std::pair<double, double> Predict(double time);
    virtual arma::vec Simulate(arma::vec time);
};

/*
 Same as KalmanFilter1 but for a CARMA(p,q) process
 */

class KalmanFilterp : public KalmanFilter<arma::vec> {
public:
    // Constructor
    KalmanFilterp(arma::vec& time, arma::vec& y, arma::vec& yerr, double sigsqr, arma::vec& omega, arma::vec& ma_coefs) :
        KalmanFilter(time, y, yerr, sigsqr, omega), ma_coefs_(ma_coefs)
    {
        ar_roots_ = ARRoots(omega_);
        
        p_ = omega_.n_elem;
        state_vector_.set_size(p_);
        StateVar_.set_size(p_,p_);
        PredictionVar_.set_size(p_,p_);
        
        q_ = ma_coefs_.n_elem;
        rotated_ma_coefs_.set_size(q_);
    }
    
    // Set and get and moving average terms
    void SetMA(arma::vec ma_coefs) {
        ma_coefs_ = ma_coefs.st();
    }
    arma::vec GetMA() {
        return ma_coefs_.st();
    }
    // Compute the roots of the AR(p) polynomial from the PSD parameters, omega
    arma::cx_vec ARRoots(arma::vec omega);
    
    // Methods to perform the Kalman Filter operations
    virtual void Reset();
    virtual void Update();
    virtual std::pair<double, double> Predict(double time);
    virtual arma::vec Simulate(arma::vec time);
    
private:
    // parameters
    arma::cx_vec ar_roots_; // ar_roots are derived from the values of omega_
    arma::rowvec ma_coefs_; // moving average terms
    unsigned int p_, q_; // the orders of the CARMA process
    // quantities updated in the kalman filter
    arma::cx_vec state_vector_; // current value of the rotated state vector
    arma::cx_mat StateVar_; // stationary covariance matrix of the rotated state vector
    arma::cx_vec rotated_ma_coefs_; // rotated moving average coefficients
    arma::cx_mat PredictionVar_; // covariance matrix of the predicted rotated state vector
    arma::cx_vec kalman_gain_;
    arma::cx_vec rho_;
    double innovation_;
};


#endif /* defined(__carma_pack__kfilter__) */
