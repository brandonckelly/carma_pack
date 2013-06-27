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

#include <string>
#include <random.hpp>
#include <proposals.hpp>
#include <samplers.hpp>
#include <steps.hpp>
#include <parameters.hpp>

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

class CAR1 : public Parameter<arma::vec> {
	
public:
	// Constructor //
	CAR1(bool track, std::string name, arma::vec& time, arma::vec& y, arma::vec& yerr, double temperature=1.0);

	virtual arma::vec StartingValue();
	virtual std::string StringValue();
	
    void Save(arma::vec new_value);

    // Methods to get the data vectors //
    arma::vec GetTime() {
        return time_;
    }
    
    arma::vec GetTimeSeries() {
        return y_;
    }
    
    arma::vec GetTimeSeriesErr() {
        return yerr_;
    }
    
	// Methods for the Kalman filter //
	
	// Calculate the kalman filter mean and variance
	virtual void KalmanFilter(arma::vec car1_value);
	// Return the Kalman Filter Mean
	arma::vec GetKalmanMean();
	// Return the Kalman Filter Variance
	arma::vec GetKalmanVariance();
	
	// Methods for related to the log-posterior //
	
	// Set the bounds on the uniform prior.
	virtual void SetPrior(double max_stdev); 
	virtual void PrintPrior();
	
	// Compute the log-posterior
    virtual double LogPrior(arma::vec car1_value);
	virtual double LogDensity(arma::vec car1_value);
    
protected:
	// Data vectors
	arma::vec time_;
	arma::vec y_;
	arma::vec yerr_;
	arma::vec dt_;
	// Vectors for the Kalman filter. Make these protected members so we don't
	// have to initialize them every time we evaluate the likelihood function.
	arma::vec kalman_mean_;
	arma::vec kalman_var_;
	
	// Prior parameters
	double max_stdev_; // Maximum value of the standard deviation of the CAR(1) process
	double max_freq_; // Maximum value of omega = 1 / tau
	double min_freq_; // Minimum value of omega = 1 / tau
	int measerr_dof_; // Degrees of freedom for prior on measurement error scaling parameter
};

/*
 Continuous time autoregressive process of order p.
 
*/

class CARp : public CAR1 {
public:
    // Constructor
    CARp(bool track, std::string name, arma::vec& time, arma::vec& y, arma::vec& yerr, int p, double temperature=1.0):
    CAR1(track, name, time, y, yerr, temperature), p_(p)
	{ 
		value_.set_size(p_+2);
	}
    
    // Return the starting value and set log_posterior_
	arma::vec StartingValue();
	
	// Calculate the kalman filter mean and variance
	void KalmanFilter(arma::vec theta);
	
	// Calculate the logarithm of the posterior
	double LogDensity(arma::vec theta);
    
    // Calculate the variance of the CAR(p) process
    double Variance(arma::cx_vec alpha_roots, double sigma);
	
	// Print out useful info
	void PrintInfo();
	
private:
    int p_; // Order of the CAR(p) process
};

/* 
 Continuous time autoregressive process of order p with a moving average term,
 defined by kappa.
 
*/

class CARMA : public CAR1 {
	
public:
	// Constructor //
	CARMA(bool track, std::string name, arma::vec& time, arma::vec& y, arma::vec& yerr, int p, double temperature=1.0);

	// Return the starting value and set log_posterior_
	arma::vec StartingValue();
    
    // Save the value
    void Save(arma::vec new_value);
	
	// Calculate the kalman filter mean and variance
	void KalmanFilter(double sigma, double measerr_scale, arma::cx_vec alpha_roots);

	// Set the bounds on the prior.
	void SetPrior(double max_stdev, arma::vec phi_var); 
	
	// Calculate the logarithm of the posterior
	double LogDensity(arma::vec car_value);
    
    // Calculate the variance of the CAR(p) process
    double Variance(arma::cx_vec alpha_roots, double sigma);
    
    // Set the value of kappa, the prescribed moving average term.
    void SetKappa(double kappa);
	
	// Print out useful info
	void PrintInfo(bool print_data=true);
	
private:
	int p_; // Order of CARMA(p) process
	double kappa_; // Width of additional moving average factor for converting between phi and alpha
	arma::vec ma_terms_; // The moving average terms implied by the alpha --> phi transform
	// Prior parameters
	arma::vec phi_var_; // Vector of prior variances on phi
	
	double tol_; // Tolerance for testing equality of roots.
};

/********************************
	FUNCTION PROTOTYPES
********************************/

// Check if all of the roots are unique within some fractional tolerance
bool unique_roots(arma::cx_vec roots, double tolerance);

// Return the coefficients of a polynomial given its roots.
arma::vec polycoefs(arma::cx_vec roots);

#endif
