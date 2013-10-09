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
#include <boost/assert.hpp>

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;


/*
 Abstract base class for the Kalman Filter of a CARMA(p,q) process.
 */

template <class OmegaType>
class KalmanFilter {
public:
    // Kalman mean and variance
    arma::vec mean;
    arma::vec var;

    // Constructor
    KalmanFilter() {};
    KalmanFilter(arma::vec& time, arma::vec& y, arma::vec& yerr) :
    time_(time), y_(y), yerr_(yerr)
    {
        init();
    }

    // Initialize arrays
    void init() {
        int ndata = time_.n_elem;
        dt_ = time_(arma::span(1,ndata-1)) - time_(arma::span(0,ndata-2));

        // Make sure the time vector is strictly increasing
        if (dt_.min() < 0) {
            std::cout << "Time vector is not sorted in increasing order. Sorting the data vectors..."
            << std::endl;
            // Sort the time values such that dt > 0
            arma::uvec sorted_indices = arma::sort_index(time_);
            time_ = time_.elem(sorted_indices);
            y_ = y_.elem(sorted_indices);
            yerr_ = yerr_.elem(sorted_indices);
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
            ndata = time_.n_elem;
            dt_.set_size(ndata-1);
            dt_ = time_.rows(1,time_.n_elem-1) - time_.rows(0,time_.n_elem-2);
        }
        
        // Set the size of the Kalman Filter mean and variance vectors
        mean.zeros(time_.n_elem);
        var.zeros(time_.n_elem);
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
    virtual void SetOmega(OmegaType omega) {
        omega_ = omega;
    }
    OmegaType GetOmega() {
        return omega_;
    }
    virtual void SetMA(arma::vec ma_coefs) {}
        
    double GetConst() { return yconst_; }
    double GetSlope() { return yslope_; }

    std::vector<double> GetMeanSvec() { return arma::conv_to<std::vector<double> >::from(mean); }
    std::vector<double> GetVarSvec() { return arma::conv_to<std::vector<double> >::from(var); }

    /*
     Methods to perform the Kalman Filter operations 
     */
    virtual void Reset() = 0;
    virtual void Update() = 0;
    virtual std::pair<double, double> Predict(double time) = 0;

    void Filter() {
        // Run the Kalman Filter
        Reset();
        for (int i=1; i<time_.n_elem; i++) {
            Update();
        }
    }
    
    // simulate a CARMA process, conditional on the measured time series
    std::vector<double> Simulate(arma::vec time) {
        // first save old values since we will overwrite them later
        arma::vec time0 = time_;
        arma::vec y0 = y_;
        arma::vec yerr0 = yerr_;
        
        arma::vec ysimulated(time.n_elem);
        
        time = arma::sort(time);
        unsigned int insert_idx = 0;
        arma::vec tinsert(1);
        arma::vec dt_insert(1);
        arma::vec yinsert(1);
        arma::vec yerr_insert(1);
        yerr_insert(0) = 0.0;
        for (int i=0; i<time.n_elem; i++) {
            insert_idx = 0;
            // first simulate the value at time(i)
            std::pair<double, double> ypredict = this->Predict(time(i));
            ysimulated(i) = RandGen.normal(ypredict.first, sqrt(ypredict.second));
            // find the index where time_[insert_idx-1] < time(i) < time_[insert_idx]
            while (time_(insert_idx) < time(i)) {
                insert_idx++;
                if (insert_idx == (time_.n_elem)) {
                    break;
                }
            }
            // insert the simulated value into the measured time series array. these values are used in subsequent
            // calls to Predict(time(i)).
            tinsert(0) = time(i);
            time_.insert_rows(insert_idx, tinsert);
            dt_ = time_(arma::span(1,time_.n_elem-1)) - time_(arma::span(0,time_.n_elem-2));
            yinsert(0) = ysimulated(i);
            y_.insert_rows(insert_idx, yinsert);
            yerr_.insert_rows(insert_idx, yerr_insert);
            mean.zeros(time_.n_elem);
            var.zeros(time_.n_elem);
        }
        
        // restore values of measured time series
        time_ = time0;
        dt_ = time_(arma::span(1,time_.n_elem-1)) - time_(arma::span(0,time_.n_elem-2));
        y_ = y0;
        yerr_ = yerr0;
        // restore the original sizes of the kalman mean and variance arrays
        mean.zeros(time_.n_elem);
        var.zeros(time_.n_elem);
        
        return arma::conv_to<std::vector<double> >::from(ysimulated);
    }
    
    // Methods needed for interpolation and backcasting
    virtual void InitializeCoefs(double time, unsigned int itime, double ymean, double yvar) = 0;
    virtual void UpdateCoefs() = 0;

protected:
    // Data
    arma::vec time_;
    arma::vec dt_;
    arma::vec y_;
    arma::vec yerr_;
    // The Kalman Filter parameters
    double sigsqr_;
    OmegaType omega_;
    unsigned int current_index_;
    // linear coefficients needed for doing interpolation or backcasting
    double yconst_, yslope_;
};

/*
 Class to perform the Kalman Filter and related operations for a zero-mean CAR(1) process (see below).
 This class will calculate the Kalman Filter, needed to calculate the log-likelihood in the CAR1
 parameter class. It will also provide interpolated, extrapolated, and simulated values given a
 CAR(1) model and a measured time series.
 */

class KalmanFilter1 : public KalmanFilter<double> {
public:
    // Constructors
    KalmanFilter1() : KalmanFilter<double>() {}
    KalmanFilter1(arma::vec& time, arma::vec& y, arma::vec& yerr) : KalmanFilter<double>(time, y, yerr) {}
    KalmanFilter1(arma::vec& time, arma::vec& y, arma::vec& yerr, double sigsqr, double omega) :
        KalmanFilter<double>(time, y, yerr)
    {
        sigsqr_ = sigsqr;
        omega_ = omega;
    }
    KalmanFilter1(std::vector<double> time, std::vector<double> y, std::vector<double> yerr) : 
        KalmanFilter<double>() 
    {
        arma::vec armatime = arma::conv_to<arma::vec>::from(time);
        arma::vec armay    = arma::conv_to<arma::vec>::from(y);
        arma::vec armady   = arma::conv_to<arma::vec>::from(yerr);
        time_ = armatime;
        y_ = armay;
        yerr_ = armady;
        init();
    }
    KalmanFilter1(std::vector<double> time, std::vector<double> y, std::vector<double> yerr, double sigsqr, double omega) :
        KalmanFilter<double>()
    {
        arma::vec armatime = arma::conv_to<arma::vec>::from(time);
        arma::vec armay    = arma::conv_to<arma::vec>::from(y);
        arma::vec armady   = arma::conv_to<arma::vec>::from(yerr);
        time_ = armatime;
        y_ = armay;
        yerr_ = armady;
        sigsqr_ = sigsqr;
        omega_ = omega;
        init();
    }

    // Methods to perform the Kalman Filter operations
    void Reset();
    void Update();
    std::pair<double, double> Predict(double time);
    void InitializeCoefs(double time, unsigned int itime, double ymean, double yvar);
    void UpdateCoefs();

    std::vector<double> Simulate(std::vector<double> time) {
        arma::vec armatime = arma::conv_to<arma::vec>::from(time);
        arma::vec armasimulate = KalmanFilter<double>::Simulate(armatime);
        std::vector<double> vecsimulate = arma::conv_to<std::vector<double> >::from(armasimulate);
        return vecsimulate;
    }
};

/*
 Same as KalmanFilter1 but for a CARMA(p,q) process
 */

class KalmanFilterp : public KalmanFilter<arma::cx_vec> {
public:
    // Constructors
    KalmanFilterp() : KalmanFilter<arma::cx_vec>() {}
    KalmanFilterp(arma::vec& time, arma::vec& y, arma::vec& yerr) : KalmanFilter<arma::cx_vec>(time, y, yerr) {init();}
    KalmanFilterp(arma::vec& time, arma::vec& y, arma::vec& yerr, double sigsqr, arma::cx_vec& omega, arma::vec& ma_coefs) :
        KalmanFilter<arma::cx_vec>(time, y, yerr)
    {
        sigsqr_ = sigsqr;
        omega_ = omega; // omega are the roots of the AR characteristic polynomial
        p_ = omega_.n_elem;
        if (p_ > ma_coefs.n_elem) {
            ma_coefs.resize(p_);
        }
        SetMA(ma_coefs);
        
        // set sizes of arrays
        state_vector_.zeros(p_);
        StateVar_.zeros(p_,p_);
        PredictionVar_.zeros(p_,p_);
        kalman_gain_.zeros(p_);
        rho_.zeros(p_);
        state_const_.zeros(p_);
        state_slope_.zeros(p_);        
        rotated_ma_coefs_.zeros(p_);
    }
    KalmanFilterp(std::vector<double> time, std::vector<double> y, std::vector<double> yerr) :
        KalmanFilter<arma::cx_vec>()
    {
        arma::vec armatime = arma::conv_to<arma::vec>::from(time);
        arma::vec armay    = arma::conv_to<arma::vec>::from(y);
        arma::vec armady   = arma::conv_to<arma::vec>::from(yerr);
        time_ = armatime;
        y_ = armay;
        yerr_ = armady;
        init();
    }
   KalmanFilterp(std::vector<double> time, std::vector<double> y, std::vector<double> yerr, double sigsqr, 
                 std::vector<std::complex<double> > omega, std::vector<double> ma_coefs) :
        KalmanFilter<arma::cx_vec>()
    {
        arma::vec armatime  = arma::conv_to<arma::vec>::from(time);
        arma::vec armay     = arma::conv_to<arma::vec>::from(y);
        arma::vec armady    = arma::conv_to<arma::vec>::from(yerr);
        arma::cx_vec armaomega = arma::conv_to<arma::cx_vec>::from(omega);
        arma::vec armacoefs = arma::conv_to<arma::vec>::from(ma_coefs);
        time_ = armatime;
        y_ = armay;
        yerr_ = armady;
        sigsqr_ = sigsqr;
        omega_ = armaomega;
        p_ = omega_.n_elem;
        if (p_ > armacoefs.n_elem) {
            armacoefs.resize(p_);
        }
        SetMA(armacoefs);
        
        // set sizes of arrays
        state_vector_.zeros(p_);
        StateVar_.zeros(p_,p_);
        PredictionVar_.zeros(p_,p_);
        kalman_gain_.zeros(p_);
        rho_.zeros(p_);
        state_const_.zeros(p_);
        state_slope_.zeros(p_);
        rotated_ma_coefs_.zeros(p_);

        init();
    }

    // Set and get and moving average terms
    void SetMA(arma::vec ma_coefs) {
        ma_coefs_ = ma_coefs.st();
    }
    arma::vec GetMA() {
        return ma_coefs_.st();
    }

    // set the AR parameters
    void SetOmega(arma::cx_vec omega) {
        omega_ = omega;
        // resize arrays
        p_ = omega_.n_elem;
        state_vector_.zeros(p_);
        StateVar_.zeros(p_,p_);
        PredictionVar_.zeros(p_,p_);
        kalman_gain_.zeros(p_);
        rho_.zeros(p_);
        state_const_.zeros(p_);
        state_slope_.zeros(p_);
        rotated_ma_coefs_.zeros(p_);
    }

    
    // Methods to perform the Kalman Filter operations
    void Reset();
    void Update();
    std::pair<double, double> Predict(double time);
    void InitializeCoefs(double time, unsigned int itime, double ymean, double yvar);
    void UpdateCoefs();
    
    std::vector<double> Simulate(std::vector<double> time) {
        arma::vec armatime = arma::conv_to<arma::vec>::from(time);
        arma::vec armasimulate = KalmanFilter<arma::cx_vec>::Simulate(armatime);
        std::vector<double> vecsimulate = arma::conv_to<std::vector<double> >::from(armasimulate);
        return vecsimulate;
    }
    
private:
    // parameters
    arma::rowvec ma_coefs_; // moving average terms
    unsigned int p_; // the orders of the CARMA process
    // quantities defining the current state of the kalman filter
    arma::cx_vec state_vector_; // current value of the rotated state vector
    arma::cx_mat StateVar_; // stationary covariance matrix of the rotated state vector
    arma::cx_rowvec rotated_ma_coefs_; // rotated moving average coefficients
    arma::cx_mat PredictionVar_; // covariance matrix of the predicted rotated state vector
    arma::cx_vec kalman_gain_;
    arma::cx_vec rho_;
    double innovation_;
    // linear coefficients needed for doing interpolation or backcasting
    arma::cx_vec state_const_;
    arma::cx_vec state_slope_;    
};


#endif /* defined(__carma_pack__kfilter__) */
