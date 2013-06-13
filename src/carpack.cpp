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
#include "carpack.hpp"

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

/********************************************************************
						METHODS OF CAR1 CLASS
 *******************************************************************/

// CAR1 constructor. In addition to the standard parameter constructor,
// this constructor initializes the data vectors, makes sure that the
// time data vector is strictly increasing, and sets the prior parameters
// do the default values.

CAR1::CAR1(bool track, std::string name, arma::vec& time, arma::vec& y, arma::vec& yerr, double temperature) : 
Parameter<arma::vec>(track, name, temperature)
{
	// Set the size of the parameter vector theta=(mu,sigma,measerr_scale,log(omega))
	value_.set_size(3);
	// Set the degrees of freedom for the prior on the measurement error scaling parameter
	measerr_dof_ = 100;
	
    y_ = y - arma::mean(y); // center the time series
    time_ = time;
    yerr_ = yerr;
	int ndata = time.n_rows;
	
	dt_ = time(arma::span(1,ndata-1)) - time(arma::span(0,ndata-2));
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

	// Set the size of the Kalman filter vectors
	kalman_mean_.set_size(ndata);
	kalman_var_.set_size(ndata);
}

// Method of CAR1 class to set the bounds on the uniform prior.
//
// The prior on the parameters is uniform on log(omega) and the CAR(1)
// process standard deviation, subject to the following constraints:
//
//	omega < 1 / min(dt)
//  CAR(1) standard deviation < max_stdev
//
// where CAR1_stdev = sigma / sqrt(2 * omega). Therefore, the upper bound
// on CAR1_stdev implies an upper bound on the factor sigma / sqrt(omega), 
// and thus a lower bound on omega|sigma. The default value of max_stdev=6.9
// was chosen because it is assumed that the time series should not show
// a dispersion greater than 3 orders of magnitude.

void CAR1::SetPrior(double max_stdev)
{
	max_stdev_ = max_stdev;
	max_freq_ = 10.0 / dt_.min();
	min_freq_ = 1.0 / (10.0 * (time_.max() - time_.min()));
}

// Method of CAR1 class to print the prior parameters.
void CAR1::PrintPrior()
{
	std::cout << "Maximum Standard Deviation: " << max_stdev_ << std::endl;
	std::cout << "Maximum Frequency: " << max_freq_ << std::endl;
	std::cout << "Minimum Frequency: " << min_freq_ << std::endl;
	std::cout << "Degrees of freedom for measurement error scaling parameter: " <<
	measerr_dof_ << std::endl;
}

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
	
	log_omega_start = -1.0 * log(arma::median(dt_) * RandGen.uniform( 1.0, 50.0 ));
	log_omega_start = std::min(log_omega_start, max_freq_);
	
	sigma = car1_stdev_start * sqrt(2.0 * exp(log_omega_start));
	
	// Get initial value of the measurement error scaling parameter by
	// drawing from its prior.
	
	double measerr_scale = RandGen.scaled_inverse_chisqr(measerr_dof_, 1.0);
    measerr_scale = std::min(measerr_scale, 1.99);
    measerr_scale = std::max(measerr_scale, 0.51);
	
	arma::vec theta(3);
	
	theta << sigma << measerr_scale << log_omega_start << arma::endr;
	
	// Initialize the Kalman filter
	KalmanFilter(theta);
	
	return theta;
}

// Method of CAR1 class to return the value of the parameter vector
// as a string
std::string CAR1::StringValue()
{
	std::stringstream ss;
    
    ss << log_posterior_;
    for (int i=0; i<value_.n_elem; i++) {
        ss << " " << value_(i);
    }

	//(theta_.t()).raw_print(ss);

	std::string theta_str = ss.str();
	return theta_str;
}

// Method of CAR1 class to save a new parameter vector. Also
// save the values of the log-posterior.
void CAR1::Save(arma::vec new_car1)
{
	// new_car1 ---> value_
	value_ = new_car1;
	
	double measerr_scale = value_(1);
	
	// Update the log-posterior using this new value of theta.
	//
	// IMPORTANT: This assumes that the Kalman filter was calculated
	// using the value of new_car1.
	//
	log_posterior_ = 0.0;
	for (int i=0; i<time_.n_elem; i++) {
		log_posterior_ += -0.5 * log(kalman_var_(i)) - 
		0.5 * (y_(i) - kalman_mean_(i)) * (y_(i) - kalman_mean_(i)) / kalman_var_(i);
	}
	
	log_posterior_ += LogPrior(new_car1);
}

// Method of CAR1 to compute the Kalman filter. This is needed for the likelihood 
// calculation, and is useful for assessing the quality of the fit.
void CAR1::KalmanFilter(arma::vec car1_value)
{
	// The CAR(1) parameters
	double sigma = car1_value(0); // Amplitude of driving noise
	double measerr_scale = car1_value(1); // Scaling parameter for measurement errors
	double omega = exp(car1_value(2)); // Damping frequency of time series
	
	kalman_mean_(0) = 0.0;
	kalman_var_(0) = sigma * sigma / (2.0 * omega);
	
	// TODO: SEE IF I CAN SPEED THIS UP USING ITERATORS
	double rho, var_ratio;
	for (int i=1; i<time_.n_elem; i++) 
	{
		rho = exp(-1.0 * omega * dt_(i-1));
		var_ratio = kalman_var_(i-1) / (kalman_var_(i-1) + measerr_scale * yerr_(i-1) * yerr_(i-1));
		
		// Update the Kalman filter mean
		kalman_mean_(i) = rho * kalman_mean_(i-1)
			+ rho * var_ratio * (y_(i-1) - kalman_mean_(i-1));
		
		// Update the Kalman filter variance
		kalman_var_(i) = kalman_var_(0) * (1.0 - rho * rho)
			+ rho * rho * kalman_var_(i-1) * (1.0 - var_ratio);
	}
    kalman_var_ += measerr_scale * yerr_ % yerr_; // add in contribution to variance from measurement errors
}

// Return the log-prior for a CAR(1) process
double CAR1::LogPrior(arma::vec car1_value)
{
    double measerr_scale = car1_value(1);
    
    double logprior = -0.5 * measerr_dof_ / measerr_scale -
     (1.0 + measerr_dof_ / 2.0) * log(measerr_scale);
    
    return logprior;
}

// Method of CAR1 to compute the log-posterior as a function of the 
// parameter vector
double CAR1::LogDensity(arma::vec car1_value)
{
	double omega = exp(car1_value(2));
	double car1_stdev = car1_value(0) / sqrt(2.0 * omega);
	double measerr_scale = car1_value(1);
	double logpost = 0.0;
	
	// Run the Kalman filter
	KalmanFilter(car1_value);
	
	// TODO: SEE IF I CAN GET A SPEED INCREASE USING ITERATORS
	for (int i=0; i<time_.n_elem; i++) {
		logpost += -0.5 * log(kalman_var_(i)) -
		0.5 * (y_(i) - kalman_mean_(i)) * (y_(i) - kalman_mean_(i)) / kalman_var_(i);
	}
	
	// Prior bounds satisfied?
	if ( (omega > max_freq_) || (omega < min_freq_) || 
		 (car1_stdev > max_stdev_) || (car1_stdev < 0) ||
		 (measerr_scale < 0.5) || (measerr_scale > 2.0) ) {
		// Value of either omega or model standard deviation are above the
		// prior bounds, so set logpost to be negative infinity
		logpost = -1.0 * arma::datum::inf;
        return logpost;
	}
	
    logpost += LogPrior(car1_value);
    
	return logpost;
}

// Return the Kalman Filter Mean
arma::vec CAR1::GetKalmanMean()
{
	return kalman_mean_;
}

// Return the Kalman Filter Variance
arma::vec CAR1::GetKalmanVariance()
{
	return kalman_var_;
}

/********************************************************************
                        METHODS OF CARp CLASS
 *******************************************************************/

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
        
        // Initial guess for model standard deviation is randomly distributed
        // around measured standard deviation of the time series
        
        double yvar = RandGen.scaled_inverse_chisqr(y_.size()-1, arma::var(y_));
        
        // Get initial value of the measurement error scaling parameter by
        // drawing from its prior.
        
        double measerr_scale = RandGen.scaled_inverse_chisqr(measerr_dof_, 1.0);
        measerr_scale = std::min(measerr_scale, 1.99);
        measerr_scale = std::max(measerr_scale, 0.51);
        
        theta(0) = sqrt(yvar);
        theta(1) = measerr_scale;
        for (int i=0; i<p_/2; i++) {
            theta(2+2*i) = log(lorentz_cent(i));
            theta(3+2*i) = log(lorentz_width(i));
        }
        if ((p_ % 2) == 1) {
            // p is odd, so add in additional value of lorentz_width
            theta(p_+1) = log(lorentz_width(p_/2));
        }
        
        // Initialize the Kalman filter
        KalmanFilter(theta);
        
        double logpost = LogDensity(theta);
        good_initials = arma::is_finite(logpost);
    } // continue loop until the starting values give us a finite posterior
    
    return theta;
}

// Calculate the kalman filter mean and variance
void CARp::KalmanFilter(arma::vec theta)
{
    // Construct the complex vector of roots of the characteristic polynomial:
    // alpha(s) = s^p + alpha_1 s^{p-1} + ... + alpha_{p-1} s + alpha_p
    arma::cx_vec alpha_roots(p_);
    for (int i=0; i<p_/2; i++) {
        double lorentz_cent = exp(theta(2+2*i));
        double lorentz_width = exp(theta(3+2*i));
        alpha_roots(2*i) = std::complex<double> (-lorentz_width,lorentz_cent);
        alpha_roots(2*i+1) = std::conj(alpha_roots(2*i));
    }
	
    if ((p_ % 2) == 1) {
        // p is odd, so add in additional low-frequency component
        double lorentz_width = exp(theta(p_+1));
        alpha_roots(p_-1) = std::complex<double> (-lorentz_width, 0.0); 
    }
    
    alpha_roots *= 2.0 * arma::datum::pi;
    
	// Initialize the matrix of Eigenvectors. We will work with the state vector
	// in the space spanned by the Eigenvectors because in this space the state
	// transition matrix is diagonal, so the calculation of the matrix exponential
	// is fast.
	arma::cx_mat EigenMat(p_,p_);
	EigenMat.row(0) = arma::ones<arma::cx_rowvec>(p_);
	EigenMat.row(1) = alpha_roots.st();
	for (int i=2; i<p_; i++) {
		EigenMat.row(i) = strans(arma::pow(alpha_roots, i));
	}
    
	// Input vector under original state space representation
	arma::cx_vec Rvector = arma::zeros<arma::cx_vec>(p_);
	Rvector(p_-1) = 1.0;
    
	// Transform the input vector to the rotated state space representation. 
	// The notation R and J comes from Belcher et al. (1994).
	arma::cx_vec Jvector(p_);
	Jvector = arma::solve(EigenMat, Rvector);
	
	// Transform the moving average coefficients to the space spanned by EigenMat.
    // For a CAR(p) model this is just a row vector of ones.
    arma::cx_rowvec rotated_ma_terms = EigenMat.row(0);
    
	// Get the amplitude of the driving noise
	double normalized_variance = Variance(alpha_roots, 1.0);
    double ysigma = theta(0); // The model standard deviation for the time series
	double sigma = ysigma / sqrt(normalized_variance);
	
	// Calculate the stationary covariance matrix of the state vector.
	arma::cx_mat StateVar(p_,p_);
	for (int i=0; i<p_; i++) {
		for (int j=i; j<p_; j++) {
			// Only fill in upper triangle of StateVar because of symmetry
			StateVar(i,j) = -sigma * sigma * Jvector(i) * std::conj(Jvector(j)) / 
            (alpha_roots(i) + std::conj(alpha_roots(j)));
		}
	}
	StateVar = arma::symmatu(StateVar); // StateVar is symmetric
	arma::cx_mat PredictionVar = StateVar; // One-step state prediction error
	
	arma::cx_vec state_vector(p_);
	state_vector.zeros(); // Initial state is set to zero
	
    
	// Initialize the Kalman mean and variance. These are the forecasted value
	// for the measured time series values and its variance, conditional on the
	// previous measurements
	kalman_mean_(0) = 0.0;
    kalman_var_(0) = std::real( arma::accu(PredictionVar) );
    
	// Get the scaling parameter for the measurement error variance
	double measerr_scale = theta(1);
	
	double innovation = y_(0); // The innovations
	kalman_var_(0) += measerr_scale * yerr_(0) * yerr_(0); // Add in measurement error contribution
    
	// Run the Kalman Filter
	// 
	// CAN I MAKE THIS FASTER USING ITERATORS?
	//
	arma::cx_vec kalman_gain(p_);
	arma::cx_vec state_transition(p_);
	
	for (int i=1; i<time_.n_elem; i++) {
		// First compute the Kalman Gain
		kalman_gain = arma::sum(PredictionVar, 1) / kalman_var_(i-1);
        
		// Now update the state vector
		state_vector += kalman_gain * innovation;

		// Update the state one-step prediction error variance
		PredictionVar -= kalman_var_(i-1) * (kalman_gain * kalman_gain.t());
        
		// Predict the next state
		state_transition = arma::exp(alpha_roots * dt_(i-1));
		state_vector = state_vector % state_transition;

		// Update the predicted state variance matrix
		PredictionVar = (state_transition * state_transition.t()) % (PredictionVar - StateVar) 
        + StateVar;
        
		// Now predict the observation and its variance. Note that for a CARMA(p,q) model we need to include
        // the rotated MA terms here, which we currently ignore because they are just a vector of ones.
        kalman_mean_(i) = std::real( arma::accu(state_vector) );

        kalman_var_(i) = std::real( arma::accu(PredictionVar) );
		kalman_var_(i) += measerr_scale * yerr_(i) * yerr_(i); // Add in measurement error contribution
        
		// Finally, update the innovation
		innovation = y_(i) - kalman_mean_(i);
	}
}

// Calculate the logarithm of the posterior
double CARp::LogDensity(arma::vec theta)
{
 
    double logpost = 0.0;
    
    // Run the Kalman filter
    KalmanFilter(theta);
    
    // Calculate the log-likelihood
    double ysigma = theta(0);
	double measerr_scale = theta(1);
    
    // TODO: SEE IF I CAN GET A SPEED INCREASE USING ITERATORS
    for (int i=0; i<time_.n_elem; i++) {
        logpost += -0.5 * log(kalman_var_(i)) -
        0.5 * (y_(i) - kalman_mean_(i)) * (y_(i) - kalman_mean_(i)) / kalman_var_(i);
    }
	
    // Prior bounds satisfied?

    arma::vec lorentz_params = arma::exp(theta(arma::span(2,theta.n_elem-1)));
    // Find the set of Frequencies satisfying the prior bounds
    arma::uvec valid_frequencies1 = arma::find(lorentz_params < max_freq_);
	arma::uvec valid_frequencies2 = arma::find(lorentz_params > min_freq_);
    
    if ( (valid_frequencies1.n_elem != lorentz_params.n_elem) || 
		 (valid_frequencies2.n_elem != lorentz_params.n_elem) ||
		 (ysigma > max_stdev_) || (ysigma < 0) || 
         (measerr_scale < 0.5) || (measerr_scale > 2.0) ) {
        // Value of either the frequencies or the model standard deviation are outside
        // of the prior bounds, so set logpost to be negative infinity
        return logpost = -1.0 * arma::datum::inf;
    }
	// Make sure the Lorentzian centroids are still in decreasing order
	for (int i=1; i<p_/2; i++) {
		double lorentz_cent_difference = exp(theta(2+2*(i-1))) - exp(theta(2+2*i));
		if (lorentz_cent_difference < 0) {
			// Lorentzians are not in decreasing order, reject this proposal
			return logpost = -1.0 * arma::datum::inf;
		}
    }
    
    // Add the log-prior to the log-likelihood
    logpost += LogPrior(theta);
    
    return logpost;
}

// Calculate the variance of the CAR(p) process
double CARp::Variance(arma::cx_vec alpha_roots, double sigma)
{
    std::complex<double> car_var(0.0,0.0);
    
	// Calculate the variance of a CAR(p) process
	for (int k=0; k<alpha_roots.n_elem; k++) {
		
		std::complex<double> denom_product(1.0,0.0);
		
		for (int l=0; l<alpha_roots.n_elem; l++) {
			if (l != k) {
				denom_product *= (alpha_roots(l) - alpha_roots(k)) * 
                (std::conj(alpha_roots(l)) + alpha_roots(k));
			}
		}
		
		car_var += 1.0 / (-2.0 * std::real(alpha_roots(k)) * denom_product);
	}
	
	// Variance is real-valued, so only return the real part of CARMA_var.
    return sigma * sigma * car_var.real();
}

void CARp::PrintInfo()
{	
	std::cout << "****************** PARAMETERS **************" << std::endl;
	std::cout << "ysigma: " << value_(0) << std::endl;
	std::cout << "measerr scaling parameter: " << value_(1) << std::endl;
	arma::vec lorentz_params = value_(arma::span(2,value_.n_elem-1));
	lorentz_params.print("log(lorentz_params):");
	std::cout << "logpost: " << log_posterior_ << std::endl;
	std::cout << "Prior upper bound on CAR(p) standard deviation: " << max_stdev_ << std::endl;
	std::cout << "Prior upper bound on CAR(p) frequencies: " << max_freq_ << std::endl;
	kalman_mean_.print("kalman mean:");
	kalman_var_.print("kalman_var:");
}

/*******************************************************************
                        METHODS OF CARMA CLASS
 ******************************************************************/

// Constructor
CARMA::CARMA(bool track, std::string name, arma::vec& time, arma::vec& y, arma::vec& yerr, int p, double temperature) :
	CAR1(track, name, time, y, yerr, temperature), p_(p) 
{
	value_.set_size(p_+2);
	phi_var_.set_size(p_);
	// tol_ = 10.0 * arma::datum::eps;
	tol_ = 1e-6;
	kappa_ = 1.0 / arma::median(dt_);
	// convert kappa from angular to regular frequency
	kappa_ = kappa_ * 2.0 * arma::datum::pi;
	
	// Set the moving average terms
	ma_terms_.set_size(p_);
	ma_terms_(0) = 1.0;
	for (int i=1; i<p_; i++) {
		ma_terms_(i) = boost::math::binomial_coefficient<double>(p_-1, i) / pow(kappa_,i);
	}
}

/*
 Return the starting value and set log_posterior_. Starting values are
 calculated by randomly drawing the imaginary and real parts of the roots
 from Uniform(min_freq, max_freq), where min_freq corresponds to the
 length of the time series, and max_freq corresponds to the median time
 spacing. I do this because the PSD of the CAR(p) process can be 
 expressed as a mixture of Lorentzians, with the imaginary parts of the
 roots being the centroid frequencies, and the real parts being the
 Lorentzian widths. The starting values of phi are then calculated from
 these roots. In addition, the starting value of sigma is drawn from a 
 uniform distribution, ensuring that the variance of the CAR(p) process
 is less than the prior upper limit. The mean is just set to the mean
 of the time series.
*/

arma::vec CARMA::StartingValue() {
	
	double max_freq = kappa_;
	double min_freq = 1.0 / (time_.max() - time_.min());
	
	arma::vec sys_freq(p_); // Imaginary part of roots of characteristic polynomial
	arma::vec break_freq(p_); // Real part of roots
	
	// Roots of the characteristic polynomial :
	//		alpha(s) = s^p + alpha_1 * s^{p-1} + ... + alpha_{p-1} * s + alpha_p
	arma::cx_vec alpha_roots(p_);

    bool roots_are_unique = false;
    
	do {
		// Get initial values for phi by first getting initial values for the
		// roots of the characteristic polynomial, and then transforming to the
		// values of phi. If initial values of the roots result in non-unique
		// roots within tol_, then try again.
		
        // Obtain initial values for Lorentzian centroids (= system frequencies) and 
        // widths (= break frequencies)
        arma::vec lorentz_cent((p_+1)/2);
        lorentz_cent.randu();
        lorentz_cent = log(max_freq / min_freq) * lorentz_cent + log(min_freq);
        lorentz_cent = arma::exp(lorentz_cent);
        
        // Force system frequencies to be in descending order
        lorentz_cent = arma::sort(lorentz_cent, 1);
                
        arma::vec lorentz_width((p_+1)/2);
        lorentz_width.randu();
        lorentz_width = log(max_freq / min_freq) * lorentz_width + log(min_freq);
        lorentz_width = arma::exp(lorentz_width);
                
        if ((p_ % 2) == 1) {
            // p is odd, so add additional low-frequency component
            lorentz_cent(p_/2) = 0.0;
            // make initial break frequency of low-frequency component less than minimum
            // value of the system frequencies
            lorentz_width(p_/2) = exp(RandGen.uniform(log(min_freq), log(lorentz_cent(p_/2-1))));
        }
        
        for (int i=0; i<p_/2; i++) {
            alpha_roots(2*i) = std::complex<double> (-lorentz_width(i),lorentz_cent(i));
            alpha_roots(2*i+1) = std::conj(alpha_roots(2*i));
        }
        
        if ((p_ % 2) == 1) {
            // p is odd, so add in additional low-frequency component
            alpha_roots(p_-1) = std::complex<double> (-lorentz_width(p_/2), 0.0); 
        }
        
        alpha_roots *= 2.0 * arma::datum::pi;
        
		// Check that roots are unique within specified tolerance
        roots_are_unique = unique_roots(alpha_roots, tol_);
        
	} while (!roots_are_unique); // Repeat until we have unique roots
	
    // Transform the roots of alpha(s) to the roots of z^p * phi(1/z)
    arma::cx_vec phi_roots = (1.0 + alpha_roots / kappa_) / (1.0 - alpha_roots / kappa_);

    // Now that we have the roots of the polynomial z^p * phi(1/z), calculate the
    // coefficients {phi_j ; j = 1,...,p}
    arma::vec phi(p_+1);
    phi = polycoefs(phi_roots);
    
    // Remove coefficient for constant term in phi(1/z), since it is fixed to one
    // and therefore not a free parameter
    phi.shed_row(0);

	// Initialize the standard deviation of the CARMA(p) process
	// by drawing from its prior
	double carma_stdev_start = RandGen.scaled_inverse_chisqr(y_.size()-1, arma::var(y_));
	carma_stdev_start = sqrt(carma_stdev_start);
	
	// Get initial value of the measurement error scaling parameter by
	// drawing from its prior.
	
	double measerr_scale = RandGen.scaled_inverse_chisqr(measerr_dof_, 1.0);
    measerr_scale = std::min(measerr_scale, 1.99);
    measerr_scale = std::max(measerr_scale, 0.51);
	

    // Create the parameter vector, theta
	arma::vec theta(p_+2);
	
    theta(0) = carma_stdev_start;
    theta(1) = measerr_scale;
    theta(arma::span(2,p_+1)) = phi;
    
	// Initialize the Kalman filter
	KalmanFilter(theta(0), measerr_scale, alpha_roots);
	
    if (temperature_ == 1.0) {
        std::cout << "Kappa: " << kappa_ << std::endl;
    }
    
	return theta;
}

// Method of CARMA class to save a new parameter vector and its
// log-posterior.
void CARMA::Save(arma::vec new_car)
{
	// new_car ---> value_
	value_ = new_car;
    double measerr_scale = value_(1);
	
    
	// Update the log-posterior using this new value of theta.
	//
	// IMPORTANT: This assumes that the Kalman filter was calculated
	// using the value of new_car.
	//
	// TODO: SEE IF I CAN GET A SPEED INCREASE USING ITERATORS
	log_posterior_ = 0.0;
	for (int i=0; i<time_.n_elem; i++) {
		log_posterior_ += -0.5 * log(measerr_scale * yerr_(i) * yerr_(i) + kalman_var_(i)) - 
		0.5 * (y_(i) - kalman_mean_(i)) * (y_(i) - kalman_mean_(i)) / 
		(measerr_scale * yerr_(i) * yerr_(i) + kalman_var_(i));
	}

    // Compute the prior on the measurement error scaling parameter. This is a scaled
	// inverse chi-square distribution with scaling parameter 1.0.
	double logprior = -0.5 * measerr_dof_ / measerr_scale - 
    (1.0 + measerr_dof_ / 2.0) * log(measerr_scale);
	
	arma::vec phi = value_(arma::span(2,value_.n_elem-1));

	// Add in the prior on phi to the log-likelihood
	log_posterior_ += -0.5 * arma::sum(phi % phi / phi_var_) + logprior;
	log_posterior_ = log_posterior_;
}

// Calculate the kalman filter mean and variance
void CARMA::KalmanFilter(double ysigma, double measerr_scale, arma::cx_vec alpha_roots) {
    
	// Initialize the matrix of Eigenvectors. We will work with the state vector
	// in the space spanned by the Eigenvectors because in this space the state
	// transition matrix is diagonal, so we calculation of the matrix exponential
	// is fast.
	arma::cx_mat EigenMat(p_,p_);
	EigenMat.row(0) = arma::ones<arma::cx_rowvec>(p_);
	EigenMat.row(1) = alpha_roots.st();
	for (int i=2; i<p_; i++) {
		EigenMat.row(i) = strans(arma::pow(alpha_roots, i));
	}

	// Input vector under original state space representation
	arma::cx_vec Rvector = arma::zeros<arma::cx_vec>(p_);
	Rvector(p_-1) = 1.0;
    
	// Transform the input vector to the rotated state space representation. 
	// The notation R and J comes from Belcher et al. (1994).
	arma::cx_vec Jvector(p_);
	Jvector = arma::solve(EigenMat, Rvector);
    
	// Transform the moving average coefficients to the space spanned by EigenMat
    arma::cx_rowvec rotated_ma_terms = ma_terms_.t() * EigenMat;
    
	// Get the amplitude of the driving noise
	double normalized_variance = Variance(alpha_roots, 1.0);
	double sigma = ysigma / sqrt(normalized_variance);
	
	// Calculate the stationary covariance matrix of the state vector.
	arma::cx_mat StateVar(p_,p_);
	for (int i=0; i<p_; i++) {
		for (int j=i; j<p_; j++) {
			// Only fill in upper triangle of StateVar because of symmetry
			StateVar(i,j) = -sigma * sigma * Jvector(i) * std::conj(Jvector(j)) / 
				(alpha_roots(i) + std::conj(alpha_roots(j)));
		}
	}
	StateVar = arma::symmatu(StateVar); // StateVar is symmetric
	arma::cx_mat PredictionVar = StateVar; // One-step state prediction error
	
	arma::cx_vec state_vector(p_);
	state_vector.zeros(); // Initial state is set to zero
	
    
	// Initialize the Kalman mean and variance. These are the forecasted value
	// for the measured time series values and its variance, conditional on the
	// previous measurements
	kalman_mean_(0) = 0.0;
	kalman_var_(0) = std::real( arma::as_scalar(rotated_ma_terms * PredictionVar * 
                                                rotated_ma_terms.t()) );
	
	double innovation = y_(0); // The innovations
	kalman_var_(0) += measerr_scale * yerr_(0) * yerr_(0); // Add in measurement error contribution
		
	// Run the Kalman Filter
	// 
	// CAN I MAKE THIS FASTER USING ITERATORS?
	//
	arma::cx_vec kalman_gain(p_);
	arma::cx_vec state_transition(p_);
    
	for (int i=1; i<time_.n_elem; i++) {
		
		// First compute the Kalman Gain
		kalman_gain = PredictionVar * rotated_ma_terms.t() / kalman_var_(i-1);
        
		// Now update the state vector
		state_vector += kalman_gain * innovation;
        
		// Update the state one-step prediction error variance
		PredictionVar -= kalman_var_(i-1) * (kalman_gain * kalman_gain.t());
        
		// Predict the next state
		state_transition = arma::exp(alpha_roots * dt_(i-1));
		state_vector = state_vector % state_transition;
        
		// Update the predicted state variance matrix
		PredictionVar = (state_transition * state_transition.t()) % (PredictionVar - StateVar) 
			+ StateVar;
        
		// Now predict the observation and its variance
		kalman_mean_(i) = std::real( arma::as_scalar(rotated_ma_terms * state_vector) );
        
		kalman_var_(i) = std::real( arma::as_scalar(rotated_ma_terms * PredictionVar * 
													rotated_ma_terms.t()) );
		kalman_var_(i) += measerr_scale * yerr_(i) * yerr_(i); // Add in measurement error contribution
        
		// Finally, update the innovation
		innovation = y_(i) - kalman_mean_(i);
	}
}

// Set the prior parameters.
void CARMA::SetPrior(double max_stdev, arma::vec phi_var) {
	// Prior on standard deviation of CAR(p) process is uniform and bounded
	// from above by max_stdev
	max_stdev_ = max_stdev;
	
	// Prior is phi|phi_var ~ N(0,phi_var)
	phi_var_ = phi_var;
}

// Calculate the logarithm of the posterior
double CARMA::LogDensity(arma::vec car_value) {

    arma::vec phi = car_value(arma::span(2,car_value.n_elem-1));
	
	// First get roots of equation z^p * phi(1/z), where
	//		z^p * phi(1/z) = z^p + phi_1 * z^{p-1} + ... + phi_{p-1} * z + phi_p
	arma::cx_vec phi_roots(p_);
	
	bool use_toms493 = false;
    bool success = false;
	if (use_toms493) {
		// Use Jenkins-Traub TOMS493 algorithm to find the roots. The C++ code for this
		// is a port from the original Fortran routine, so that's why the funny syntax...
		double op[101], zeroi[100], zeror[100];
		op[0] = 1.0; // Highest order term has coefficient equal to unity
		for (int i=1; i<p_+1; i++) {
			// Fill in the non-zero coefficients
			op[i] = phi(i-1);
		}
		
		int p_copy = p_;
		
		//bool success = rpoly_ak1(op, &p_copy, zeror, zeroi); // Find the roots
		if (!success) {
			// Roots finding algorithm failed to converge, so force rejection of this
			// proposal by setting log-posterior to negative infinity
			std::cout << "bad roots" << std::endl;
			phi.print("phi:");
			for (int i=0; i<p_+2; i++) {
				std::cout << op[i] << " ";
			}
			std::cout << std::endl;
			//exit(1);
			return -1.0 * arma::datum::inf;
		}
		
		for (int i=0; i<p_; i++) {
			// Save the roots to the phi_roots complex vector.
			phi_roots(i) = std::complex<double>(zeror[i],zeroi[i]);
		}
	} else {
		// Use Laguerre's method to find the roots. The C++ code for this is
		// adapted from the Numerical Recipes routine.
		
        arma::cx_vec phi_coefs(p_+1);
        
		phi_coefs(p_) = std::complex<double>(1.0,0.0);
		for (int i=0; i<p_; i++) {
			// polyroots() takes coefficients in ascending order, opposite compared to phi
			phi_coefs(i) = std::complex<double>(phi(p_-i-1),0.0);
		}
        
        //bool success = polyroots(phi_coefs, phi_roots, true);
        
        if (!success) {
            // Root finding algorithm failed, set log-posterior to negative infinity
            std::cout << "bad roots" << std::endl;
            phi.print("phi:");
            return -1.0 * arma::datum::inf;
        }
	}
	
	// Convert roots of phi(1/z) to roots of alpha(s), where
	//		alpha(s) = s^p + alpha_1 * s^{p-1} + ... + alpha_{p-1} * s + alpha_p
	arma::cx_vec alpha_roots = -kappa_ * (1.0 - phi_roots) / (1.0 + phi_roots);

	// Check for uniqueness and stationarity
	bool roots_are_ok = unique_roots(alpha_roots, tol_);
    
	if (roots_are_ok) {
		// Make sure all roots have positive real parts for stationarity
		for (int i=0; i<alpha_roots.n_elem; i++) {
			if (std::real(alpha_roots(i)) >= 0) {
				// Root has a non-negative real part, CAR(p) process is not stationary
				roots_are_ok = false;
			}
		}
	}
	
	if (!roots_are_ok) {
		// CAR(p) process is not stationary or we cannot use the Kalman filter,
		// so reject this value of phi by forcing the log-posterior to be equal
		// to negative infinity
		return -1.0 * arma::datum::inf;
	}
	
	// Prior bounds satisfied?
	if ( (car_value(0) < 0) || (car_value(0) > max_stdev_) ||
         (car_value(1) < 0.50) || (car_value(1) > 2.0)) {
		// Value of model standard deviation are above the prior bounds, so set 
		// logpost to be negative infinity.
		return -1.0 * arma::datum::inf;
	}
	
	// Calculate the Kalman Filter
	
	double ysigma = car_value(0); // The standard deviation of the CARMA(p) process
	double measerr_scale = car_value(1); // The scaling factor for the measurement errors
	
	KalmanFilter(ysigma, measerr_scale, alpha_roots);
	
	// Calculate the log-likelihood
	
	// TODO: SEE IF I CAN GET A SPEED INCREASE USING ITERATORS
	
	double logpost = 0.0;
	for (int i=0; i<time_.n_elem; i++) {
		logpost += -0.5 * log(measerr_scale * yerr_(i) * yerr_(i) + kalman_var_(i)) - 
		0.5 * (y_(i) - kalman_mean_(i)) * (y_(i) - kalman_mean_(i)) / 
		(measerr_scale * yerr_(i) * yerr_(i) + kalman_var_(i));
	}
    
    // Compute the prior on the measurement error scaling parameter. This is a scaled
	// inverse chi-square distribution with scaling parameter 1.0.
	double logprior = -0.5 * measerr_dof_ / measerr_scale - 
    (1.0 + measerr_dof_ / 2.0) * log(measerr_scale);
	
	// Add in the prior on phi to the log-likelihood
	logpost += -0.5 * arma::sum(phi % phi / phi_var_) + logprior;
    
	return logpost;
}

// Set the value of kappa, the prescribed moving average term.
void CARMA::SetKappa(double kappa) 
{
    kappa_ = kappa;
	
	// Set the moving average terms
	ma_terms_(0) = 1.0;
	for (int i=1; i<p_; i++) {
		ma_terms_(i) = boost::math::binomial_coefficient<double>(p_-1, i) / pow(kappa_,i);
	}
}

// Calculate the variance of a CAR(p) process, given the roots
// of its characteristic polynomial and driving noise amplitude
double CARMA::Variance(arma::cx_vec alpha_roots, double sigma)
{        	
	std::complex<double> CARMA_var(0.0,0.0);

	// Calculate the variance of a CAR(p) process
	for (int k=0; k<alpha_roots.n_elem; k++) {
		
		std::complex<double> ma_sum1(0.0,0.0);
		std::complex<double> ma_sum2(0.0,0.0);
		std::complex<double> denom_product(1.0,0.0);
		
		for (int l=0; l<alpha_roots.n_elem; l++) {
			ma_sum1 += ma_terms_(l) * std::pow(alpha_roots(k),l);
			ma_sum2 += ma_terms_(l) * std::pow(-alpha_roots(k),l);
			if (l != k) {
				denom_product *= (alpha_roots(l) - alpha_roots(k)) * 
					(std::conj(alpha_roots(l)) + alpha_roots(k));
			}
		}
		
		CARMA_var += ma_sum1 * ma_sum2 / (-2.0 * std::real(alpha_roots(k)) * denom_product);
	}
	
	// Variance is real-valued, so only return the real part of CARMA_var.
    return sigma * sigma * CARMA_var.real();
}

void CARMA::PrintInfo(bool print_data)
{
	if (print_data) {
		std::cout << "*****************   DATA   *****************" << std::endl;
		time_.print("Time:");
		dt_.print("dt:");
		y_.print("Y:");
		yerr_.print("Yerr:");
	}

	std::cout << "****************** PARAMETERS **************" << std::endl;
	std::cout << "kappa: " << kappa_ << std::endl;
	ma_terms_.print("Moving Average Terms:");
	std::cout << "mu: " << value_(0) << std::endl;
	std::cout << "ysigma: " << value_(1) << std::endl;
	arma::vec phi = value_(arma::span(2,value_.n_elem-1));
	phi.print("phi:");
	std::cout << "logpost: " << log_posterior_ << std::endl;
	std::cout << "Prior upper bound on CAR(p) standard deviation: " << max_stdev_ << std::endl;
	phi_var_.print("Prior variances for phi:");
}

/*********************************************************************
                                FUNCTIONS
 ********************************************************************/

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
