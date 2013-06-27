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
        Filter();
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
    KalmanFilterp(arma::vec& time, arma::vec& y, arma::vec& yerr, double sigsqr, arma::vec omega, arma::vec ma_terms) :
    KalmanFilter(time, y, yerr, sigsqr, omega), ma_terms_(ma_terms)
    {
        ar_roots_ = ARRoots(omega_);
    }
    
    // Set and get and moving average terms
    void SetMA(arma::vec ma_terms) {
        ma_terms_ = ma_terms;
    }
    arma::vec GetMA() {
        return ma_terms_;
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
    arma::vec ma_terms_; // moving average terms
};


#endif /* defined(__carma_pack__kfilter__) */
