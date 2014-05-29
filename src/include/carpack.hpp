/*
 *  carpack.hpp
 *  carpack
 *
 *  Created by Dr. Brandon Kelly on 12/19/12.
 *
 *  Header file containing the class definitions and function protoypes
 *  for CARMCMC.
 *
 */

#ifndef __CARPACK_HDEF__
#define __CARPACK_HDEF__

#include <boost/math/special_functions/binomial.hpp>
#include <stdexcept>
#include <string>
#include <memory>
#include <random.hpp>
#include <proposals.hpp>
#include <samplers.hpp>
#include <steps.hpp>
#include <parameters.hpp>
#include "kfilter.hpp"

/*
 First-order continuous time autoregressive process (CAR(1)) class. Note that this is the same
 as an Ornstein-Uhlenbeck process. A CAR(1) process, Y(t), is defined as
 
 dY(t) = -omega * (Y(t) - mu) * dt + sigma * dW(t),
 
 where tau = 1 / omega is the "characteristic time scale" of the process, mu is the mean
 of the process, sigma is the amplitude of the driving noise, and the driving noise dW(t)
 is the derivative of the Weiner process (i.e., a white noise process). The variance of the
 process is Var(Y(t)) = sigma^2 / (2 * omega).
 
 The data member of this class include the time series values (y), the 1-sigma uncertainties 
 on y (yerr), and the time values (time). Note that it is assumed that the CAR(1) process is 
 Gaussian, and that the uncertainties on y are normally distributed with mean zero. The member 
 functions of this class include methods to calculate the Kalman filter and the logarithms of the 
 posterior probability distribution. The parameters of the CAR(1) process are held in
 the value_ private member, where value_ = (mu, log(omega), sigma) and tau = 1 / omega.
 
 The prior on theta is assumed to be uniform on theta, subject to the an upper bound
 on Var(Y(t)) and omega. The default value of the upper bound on Var(Y(t)) was chosen
 to be 6.9 (i.e., three orders of magnitude when Y(t) is the logarithm of some quantity), 
 but this may be overriden through the use of the SetPrior method. The upper bound on
 omega is fixed to be 1 / min(dt), where dt is the vector of time steps.
*/

template <class OmegaType>
class CARMA_Base : public Parameter<arma::vec> {
public:
    // Constructors
    CARMA_Base() {}
    CARMA_Base(bool track, std::string name, std::vector<double> time, std::vector<double> y, std::vector<double> yerr,
               double temperature=1.0) : Parameter<arma::vec>(track, name, temperature)
    {
        // default is to do Bayesian inference
        ignore_prior_ = false;
        
        // Set the degrees of freedom for the prior on the measurement error scaling parameter
        measerr_dof_ = 50;
        
        // convert input data to armadillo vectors
        y_  = arma::conv_to<arma::vec>::from(y);
        time_ = arma::conv_to<arma::vec>::from(time);
        yerr_ = arma::conv_to<arma::vec>::from(yerr);
        
        // default prior bounds on the standard deviation of the time series
        SetPrior(10.0 * sqrt(arma::var(y_)));
    }
    
    virtual arma::vec StartingValue() = 0;
    
    std::string StringValue()
    {
        std::stringstream ss;
        
        ss << log_posterior_;
        for (int i=0; i<value_.n_elem; i++) {
            ss << " " << value_(i);
        }
                
        std::string theta_str = ss.str();
        return theta_str;
    }

    
    void Save(arma::vec& new_value)
    {
        // new carma value ---> value_
        value_ = new_value;
        
        // Update the log-posterior using this new value of theta.
        //
        // IMPORTANT: This assumes that the Kalman filter was calculated
        // using the value of new_value.
        //
        log_posterior_ = 0.0;
        double mu = new_value(2);
        for (int i=0; i<time_.n_elem; i++) {
            double ycent = y_(i) - pKFilter_->mean(i) - mu;
            log_posterior_ += -0.5 * log(pKFilter_->var(i)) - 0.5 * ycent * ycent / pKFilter_->var(i);
        }
        log_posterior_ += LogPrior(new_value);

    }
    
    // extract the lorentzian parameters from the CARMA parameter vector
    virtual OmegaType ExtractAR(arma::vec theta) = 0;
    // extract the moving-average parameters from the CARMA parameter vector
    virtual arma::vec ExtractMA(arma::vec theta) = 0;
    // extract the variance in the driving noise from the CARMA parameter vector
    virtual double ExtractSigsqr(arma::vec theta) = 0;
        
    // compute the log-prior of the CARMA parameters
    virtual double LogPrior(arma::vec theta)
    {
        double measerr_scale = theta(1);
        
        double logprior = -0.5 * measerr_dof_ / measerr_scale -
        (1.0 + measerr_dof_ / 2.0) * log(measerr_scale);
        
        return logprior;
    }
    
    virtual void PrintOmega(OmegaType omega) {};
    
    // compute the log-posterior
    double LogDensity(arma::vec theta)
    {
        // Prior bounds satisfied?
        bool prior_satisfied = CheckPriorBounds(theta);
        if (!prior_satisfied) {
            double logpost = -1.0 * arma::datum::inf;
            return logpost;
        }
        
        OmegaType omega = ExtractAR(theta);
        arma::vec ma_coefs = ExtractMA(theta);
        double sigsqr = ExtractSigsqr(theta);
        double measerr_scale = theta(1);
        double mu = theta(2);
        
        // Run the Kalman filter
        pKFilter_->SetSigsqr(sigsqr);
        pKFilter_->SetOmega(omega);
        pKFilter_->SetMA(ma_coefs);
        arma::vec proposed_yerr = sqrt(measerr_scale) * yerr_;
        pKFilter_->SetTimeSeriesErr(proposed_yerr);
        arma::vec ycent = y_ - mu;
        pKFilter_->SetTimeSeries(ycent);
        try {
            pKFilter_->Filter();
        } catch (std::runtime_error& e) {
            std::cout << "Caught a runtime error when trying to run the Kalman Filter: " << e.what() << std::endl;
            std::cout << "Rejecting this proposal..." << std::endl;
            PrintOmega(omega);
            bool prior_satisfied = CheckPriorBounds(theta);
            std::cout << "Prior satisfied: " << prior_satisfied << std::endl;
            double logpost = -1.0 * arma::datum::inf;
            return logpost;
        }
        
        // calculate the log-likelihood
        double logpost = 0.0;
        for (int i=0; i<time_.n_elem; i++) {
            double ycent = y_(i) - pKFilter_->mean(i) - mu;
            logpost += -0.5 * log(pKFilter_->var(i)) - 0.5 * ycent * ycent / pKFilter_->var(i);
        }

        logpost += LogPrior(theta);
        
        return logpost;
    }
    
    bool virtual CheckPriorBounds(arma::vec theta)
    {
        if (ignore_prior_) {return true;}
        
        double ysigma = theta(0);
        double measerr_scale = theta(1);
        bool prior_satisfied = true;
        if ( (ysigma > max_stdev_) || (ysigma < 0) ||
            (measerr_scale < 0.5) || (measerr_scale > 2.0) )
        {
            prior_satisfied = false;
        }
        return prior_satisfied;
    }
    
    // Setters and Getters
    arma::vec GetTime() { return time_; }
    arma::vec GetTimeSeries() { return y_; }
    arma::vec GetTimeSeriesErr() { return yerr_; }
    arma::vec GetKalmanMean() { return value_(2) + pKFilter_->mean; }
    arma::vec GetKalmanVar() { return pKFilter_->var; }
    std::shared_ptr<KalmanFilter<OmegaType> > GetKalmanPtr() { return pKFilter_; }
    
    virtual void SetPrior(double max_stdev) // set the bounds on the uniform prior
    {
        max_stdev_ = max_stdev;
        arma::vec dt = time_(arma::span(1,time_.n_elem-1)) - time_(arma::span(0,time_.n_elem-2));
        max_freq_ = 1.0 / dt.min();
        min_freq_ = 1.0 / (time_.max() - time_.min());
    }
    
    // Return a copy of the MCMC samples
    std::vector<std::vector<double> > getSamples() {
        int nx = samples_.size();
        int ny = samples_[0].n_elem;
        std::vector<std::vector<double> > samples(nx,std::vector<double>(ny));
        for (int i = 0; i < nx; i++) {
            samples[i] = arma::conv_to<std::vector<double> >::from(samples_[i]);
        }
        return samples;
    }

    // grab the log-prior and log-posterior for a std::vector input
    double getLogPrior(std::vector<double> theta)
    {
        arma::vec armaVec = arma::conv_to<arma::vec>::from(theta);
        return LogPrior(armaVec);
    }
    double getLogDensity(std::vector<double> theta)
    {
        arma::vec armaVec = arma::conv_to<arma::vec>::from(theta);
        return LogDensity(armaVec);
    }
    
    // set flag for maximum-likelihood estimation
    void SetMLE(bool ignore_prior) {ignore_prior_ = ignore_prior;}
    
protected:
    // time series data
    arma::vec time_;
    arma::vec y_;
    arma::vec yerr_;
    // pointer to Kalman Filter object. The Kalman filter is the workhorse behind the likelihood calculations.
    std::shared_ptr<KalmanFilter<OmegaType> > pKFilter_;
    // prior parameters
    double max_stdev_; // Maximum value of the standard deviation of the CAR(1) process
	double max_freq_; // Maximum value of omega = 1 / tau
	double min_freq_; // Minimum value of omega = 1 / tau
	int measerr_dof_; // Degrees of freedom for prior on measurement error scaling parameter
    bool ignore_prior_; // If true, then do maximum-likelihood estimation
};

// class for a CAR(1) process
class CAR1 : public CARMA_Base<double> {
	
public:
	// Constructors //
    CAR1() {}
	CAR1(bool track, std::string name, std::vector<double> time, std::vector<double> y, std::vector<double> yerr,
         double temperature=1.0) : CARMA_Base<double>(track, name, time, y, yerr, temperature)
    {
        pKFilter_ = std::make_shared<KalmanFilter1>(time_, y_, yerr_);
        // Set the size of the parameter vector theta=(mu,sigma,measerr_scale,log(omega))
        value_.set_size(4);
    }
    
    // extract the AR parameters from the parameter vector
    double ExtractAR(arma::vec theta) { return exp(theta(3)); }
    arma::vec ExtractMA(arma::vec theta) { return arma::zeros<arma::vec>(1); }
    
    // generate starting values of the CAR(1) parameters
	arma::vec StartingValue();
    arma::vec SetStartingValue(arma::vec init);
    
    // return the variance of a CAR(1) process
    double ExtractSigsqr(arma::vec theta) {
        return 2.0 * theta(0) * theta(0) * exp(theta(3));
    }

	// Set the bounds on the uniform prior.
    bool CheckPriorBounds(arma::vec theta);
};

/*
 Continuous time autoregressive process of order p.
*/

class CARp : public CARMA_Base<arma::cx_vec> {
public:
    // Constructor
    CARp() {}
    CARp(bool track, std::string name, std::vector<double> time, std::vector<double> y, std::vector<double> yerr, int p,
         double temperature=1.0): CARMA_Base<arma::cx_vec>(track, name, time, y, yerr, temperature), p_(p)
	{
        pKFilter_ = std::make_shared<KalmanFilterp>(time_, y_, yerr_);
		value_.set_size(p_+3);
        ma_coefs_ = arma::zeros(p);
        ma_coefs_(0) = 1.0;
        order_lorentzians_ = true;
        pKFilter_->SetMA(ma_coefs_);
	}
    
    // calculate the roots of the AR(p) polynomial from the CAR(p) process parameters
    arma::cx_vec ARRoots(arma::vec theta);
    
    // Return the starting value and set log_posterior_
	arma::vec StartingValue();
    arma::vec SetStartingValue(arma::vec init);
     // return the starting values for the AR and MA parameters
    arma::vec StartingAR();

    // extract the lorentzian parameters from the CARMA parameter vector
    arma::cx_vec ExtractAR(arma::vec theta) {
        return ARRoots(theta);
    }
    // extract the moving-average parameters from the CARMA parameter vector
    arma::vec ExtractMA(arma::vec theta) { return ma_coefs_; }
    
    double ExtractSigsqr(arma::vec theta) {
        arma::cx_vec ar_roots = ARRoots(theta);
        return theta(0) * theta(0) / Variance(ar_roots, ma_coefs_, 1.0);
    }
    
    // Calculate the variance of the CAR(p) process
    double Variance(arma::cx_vec alpha_roots, arma::vec ma_coefs, double sigma, double dt=0.0);
	
    // Set the bounds on the uniform prior.
    bool CheckPriorBounds(arma::vec theta);
    
    void PrintOmega(arma::cx_vec omega) {
        omega.print("AR Roots:");
    }
    
protected:
    int p_; // Order of the CAR(p) process
    bool order_lorentzians_; // force the lorentzian centroids to be in order?
private:
    arma::vec ma_coefs_;
};

/*
 Same as CARp class, but using the Belcher et al. (1994) parameterization for the moving average coefficients.
 */
class ZCAR : public CARp
{
public:
    // constructor //
    ZCAR() {}
    ZCAR(bool track, std::string name, std::vector<double> time, std::vector<double> y, std::vector<double> yerr, int p,
         double temperature=1.0) : CARp(track, name, time, y, yerr, p, temperature)
    {
        // set value of kappa
        arma::vec dt = time_(arma::span(1,time_.n_elem-1)) - time_(arma::span(0,time_.n_elem-2));
        kappa_ = 1.0 / dt.min();
        // set the moving average coefficients
        ma_coefs_ = arma::zeros(p_);
        ma_coefs_(0) = 1.0;
        for (int i=1; i<p_; i++) {
            ma_coefs_(i) = boost::math::binomial_coefficient<double>(p_-1, i) / pow(kappa_,i);
        }
        pKFilter_->SetMA(ma_coefs_);
    }
private:
    double kappa_; // minimum frequency resolved by the observation times
    arma::vec ma_coefs_;
};

/*
 Continuous time autoregressive moving average process of order (p,q)
*/

class CARMA : public CARp
{	
public:
	// Constructor //
    CARMA() {}
	CARMA(bool track, std::string name, std::vector<double> time, std::vector<double> y, std::vector<double> yerr, int p, int q,
          double temperature=1.0) : CARp(track, name, time, y, yerr, p, temperature), q_(q)
    {
        BOOST_ASSERT_MSG(q < p, "Order of moving average polynomial must be less than order of autoregressive polynomial");
        value_.set_size(p_+q_+3);
    }

    // Return the starting value and set log_posterior_
	arma::vec StartingValue();
    arma::vec SetStartingValue(arma::vec init);
 
    // return the starting value for the MA coefficients
    arma::vec StartingMA();
    
    // extract the moving-average parameters from the CARMA parameter vector
    arma::vec ExtractMA(arma::vec theta);
    
    double ExtractSigsqr(arma::vec theta) {
        arma::cx_vec ar_roots = ARRoots(theta);
        arma::vec ma_coefs = ExtractMA(theta);
        return theta(0) * theta(0) / Variance(ar_roots, ma_coefs, 1.0);
    }
    
private:
    int q_; // order of moving average polynomial
};

/*
 CARMA(p,p-1) model using the z-transformed parameterization (Belcher et al. 1994). This is the same as the ZCAR model, except
 that kappa is a free parameter for the ZCARMA model.
 */

class ZCARMA : public CARp
{    
public:
    ZCARMA() {}
    ZCARMA(bool track, std::string name, std::vector<double> time, std::vector<double> y, std::vector<double> yerr, int p,
           double temperature=1.0) : CARp(track, name, time, y, yerr, p, temperature)
    {
        value_.set_size(p_+4);
        // set default boundaries on kappa
        arma::vec dt = time_(arma::span(1,time_.n_elem-1)) - time_(arma::span(0,time_.n_elem-2));
        kappa_high_ = 1.0 / dt.min();
        // kappa_low_ = 0.9 / dt.min();
        // kappa_low_ = 1.0 / (time_.max() - time_.min());
        kappa_low_ = std::max(1.0 / (time_.max() - time_.min()), 1.0 / (10.0 * arma::median(dt)));
    }
    
    // Return the starting value and set log_posterior_
	arma::vec StartingValue();
    arma::vec SetStartingValue(arma::vec init);
     // Return the starting value for the kappa parameter
    double StartingKappa();
    
    // extract the moving-average parameters from the CARMA parameter vector
    arma::vec ExtractMA(arma::vec theta);
    
    double ExtractSigsqr(arma::vec theta) {
        arma::cx_vec ar_roots = ARRoots(theta);
        arma::vec ma_coefs = ExtractMA(theta);
        return theta(0) * theta(0) / Variance(ar_roots, ma_coefs, 1.0);
    }

    // Set bounds on kappa
    void SetKappaBounds(double kappa_low, double kappa_high) {
        kappa_low_ = kappa_low;
        kappa_high_ = kappa_high;
    }
    
    // compute the log-prior of the ZCARMA parameters
    double LogPrior(arma::vec theta)
    {
        // first compute prior for measurement error scaling parameter
        double measerr_scale = theta(1);
        double logprior = -0.5 * measerr_dof_ / measerr_scale -
        (1.0 + measerr_dof_ / 2.0) * log(measerr_scale);
        
        // now compute prior on x = logit(kappa_norm), assuming a uniform prior on kappa
        double logit_kappa = theta(p_+3);
        logprior += -logit_kappa - 2.0 * log(1.0 + exp(-logit_kappa));
                
        return logprior;
    }

    
private:
    double kappa_low_, kappa_high_; // prior bounds on the kappa parameter
};

/********************************
	FUNCTION PROTOTYPES
********************************/

double logit(double x);
double inv_logit(double x);

// Check if all of the roots are unique within some fractional tolerance
bool unique_roots(arma::cx_vec roots, double tolerance);

// Return the coefficients of a polynomial given its roots.
arma::vec polycoefs(arma::cx_vec roots);

#endif
