//
//  kfilter.hpp
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
 Abstract base class for the Kalman Filter of a CARMA(p,q) process.
 */

template <class OmegaType>
class KalmanFilter {
public:
    // Constructor
    KalmanFilter(arma::vec& time, arma::vec& y, arma::vec& yerr, double sigsqr, OmegaType omega) :
    time_(time), y_(y), yerr_(yerr), sigsqr_(sigsqr), omega_(omega)
    {
        int ndata = time_.n_elem;
        dt_ = time_(arma::span(1,ndata-1)) - time_(arma::span(0,ndata-2));
        
        // Make sure the time vector is strictly increasing
        if (dt_.min() < 0) {
            std::cout << "Time vector is not sorted in increasing order. Sorting the data vectors..."
            << std::endl;
            // Sort the time values such that dt > 0
            arma::uvec sorted_indices = arma::sort_index(time_);
            time_ = time.elem(sorted_indices);
            y_ = y_.elem(sorted_indices);
            yerr_ = yerr.elem(sorted_indices);
            dt_ = time_.rows(1,ndata-1) - time_.rows(0,ndata-2);
        }
        // Make sure there are no duplicate values of time
        if (dt_.min() == 0) {
            std::cout << "Found duplicate values of time, removing them..." << std::endl;
            // Find the unique values of time
            arma::uvec unique_values = 1 + arma::find(dt_ != 0);
            // Add extra row in to keep time(0), y(0), yerr(0)
            unique_values.insert_rows(0, arma::zeros<arma::uvec>(1));
            time_ = time_.elem(unique_values);
            y_ = y_.elem(unique_values);
            yerr_ = yerr_.elem(unique_values);
            ndata = time.n_elem;
            dt_.set_size(ndata-1);
            dt_ = time_.rows(1,time_.n_elem-1) - time_.rows(0,time_.n_elem-2);
        }
        
        // Set the size of the Kalman Filter mean and variance vectors
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
    void Reset();
    void Update();
    std::pair<double, double> Predict(double time);
    arma::vec Simulate(arma::vec time);
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
        kalman_gain_.set_size(p_,p_);
        rho_.set_size(p_,p_);
        state_const_.set_size(p_);
        state_slope_.set_size(p_);
        
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
    void Reset();
    void Update();
    std::pair<double, double> Predict(double time);
    arma::vec Simulate(arma::vec time);
    
    // Methods needed for interpolation and backcasting
    void InitializeCoefs(double time, unsigned int itime, double ymean, double yvar);
    void UpdateCoefs();
    
private:
    // parameters
    arma::cx_vec ar_roots_; // ar_roots are derived from the values of omega_
    arma::rowvec ma_coefs_; // moving average terms
    unsigned int p_, q_; // the orders of the CARMA process
    // quantities defining the current state of the kalman filter
    arma::cx_vec state_vector_; // current value of the rotated state vector
    arma::cx_mat StateVar_; // stationary covariance matrix of the rotated state vector
    arma::cx_vec rotated_ma_coefs_; // rotated moving average coefficients
    arma::cx_mat PredictionVar_; // covariance matrix of the predicted rotated state vector
    arma::cx_vec kalman_gain_;
    arma::cx_vec rho_;
    double innovation_;
    // linear coefficients needed for doing interpolation or backcasting
    arma::cx_vec state_const_;
    arma::cx_vec state_slope_;
    double yconst_, yslope_;
    
};


#endif /* defined(__carma_pack__kfilter__) */
