//
//  steps.hpp
//  yamcmc++
//
//  Created by Brandon Kelly on 3/2/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef __yamcmc____steps__
#define __yamcmc____steps__

// Standard includes
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
// Local includes
#include "random.hpp"
#include "parameters.hpp"
#include "proposals.hpp"

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Global object for generating random variables from various distributions,
// instantiated in steps.cpp
extern RandomGenerator RandGen;

////////// FUNCTION DEFINITIONS AND TEMPLATE FUNCTIONS /////////////

// Method to perform the rank-1 Cholesky update, needed for updating the
// proposal covariance matrix
void CholUpdateR1(arma::mat& S, arma::vec& v, bool downdate);

// Function to convert any streaming type to string
template <class T>
std::string to_string (const T& t) {
    std::stringstream ss;
	ss << t;
	std::string s = ss.str();
	return s;
}

/////////////////////////////////////////////////////////////////////

// Abstract base step class. Executes the main step function, such as taking a MH step or Gibbs step.
// The Step class will never be instantiated directly.
class Step {
public:
	virtual void DoStep() = 0;
	
	// Should return the label of the parameter associated with the step instance.
	virtual std::string ParameterLabel() {
		return " ";
	}
	
	// Return a string representation of the parameter value
	virtual std::string ParameterValue() {
		return " ";
	}
	// Should return the tracking status of the parameter associated with this step instance.
	virtual bool ParameterTrack() {
		return true;
	}
    
    // Return a pointer to the parameter
    virtual BaseParameter* GetParPointer() = 0;
};


// A Gibbs step draws from the Parameter::RandomPosterior function of a Parameter and then saves the result
// using Parameter:Save.
template <class ParValueType>
class GibbsStep: public Step {
public:
	/// Default constructor, for copy assignments, etc.
	GibbsStep() {}
    
	// Main constructor taking a reference to a Parameter object.
	GibbsStep(Parameter<ParValueType>& parameter) : parameter_(parameter) {}
	
	// Method to take a Gibbs step for the parameter. Draws from the parameter's Parameter::RandomPosterior function
    // and then saves the result using Parameter::Save.
	void DoStep() {
		parameter_.Save(parameter_.RandomPosterior());
	}

	// Return string of parameter label
	std::string ParameterLabel() {
		return parameter_.Label();
	}
	
	// Return string representation of parameter value
	std::string ParameterValue() {
		return parameter_.StringValue();
	}
    
    // Return a pointer to the parameter
    BaseParameter* GetParPointer() {
        return &parameter_;
    }
	
private:
	/// Reference to parameter associated with step instance.
	Parameter<ParValueType>& parameter_;
};

/****************************************************************************
 
 Metropolis-Hastings steps. These will require both a Parameter and Proposal
 object.
 
 ****************************************************************************/


// Metropolis Hastings step. Requires both a Parameter and Proposal instance.

template <class ParValueType>
class MetropStep: public Step {
public:
	// Main constructor taking a reference to a Parameter and Proposal object.
	MetropStep(Parameter<ParValueType>& parameter, Proposal<ParValueType>& proposal,
			   int report_iter=-1) :
	parameter_(parameter), proposal_(proposal), report_iter_(report_iter)
	{
		naccept_ = 0;
		niter_ = 0;
	}
	
	std::string ParameterLabel() {
		return parameter_.Label();
	}
	
	std::string ParameterValue() {
		return parameter_.StringValue();
	}
	
    // Method to perform the accept/reject part of the Metropolis-Hastings step. Returns a boolean indicating whether
    // the proposal was acceptec (True) or rejected (False).
	bool Accept(ParValueType new_value, ParValueType old_value) {
        double par_temp = parameter_.GetTemperature();
		// MH accept/reject criteria
		alpha_ = parameter_.LogDensity(new_value) / par_temp - parameter_.GetLogDensity() / par_temp
		+ proposal_.LogDensity(old_value, new_value) - proposal_.LogDensity(new_value, old_value);
		
		if (!arma::is_finite(alpha_)) {
			// New value of the log-posterior is not finite, so reject this
			// proposal
			return false;
		}
		
		double unif = uniform_(rng);
		alpha_ = std::min(exp(alpha_), 1.0);
		if (unif < alpha_) {
			return true;
		} else {
			return false;
		}
	}
	
	// Do Metropolis-Hastings Update
	void DoStep() {
		// Draw a new parameter
		ParValueType new_value = proposal_.Draw(parameter_.Value());
		ParValueType old_value = parameter_.Value();
		
		// MH accept/reject criteria
		if (Accept(new_value, old_value)) {
			naccept_++;
			parameter_.Save(new_value);
		}
		niter_++;
		
		if (niter_ == report_iter_) {
			// Give report on average acceptance rate
			Report();
		}
	}

	// Report on acceptance rates since last report
	void Report() {
		double arate = ((double)(naccept_)) / ((double)(niter_));
		std::cout << "Average Acceptance Rate Since Last Report: " << arate << std::endl;
		niter_ = 0;
		naccept_ = 0;
	}
    
    // Return the current value of the Metropolis-Hastings ratio
    double GetMetroRatio() {
        return alpha_;
    }
    
    // Return if parameter is tracked.
    bool ParameterTrack() {
        return parameter_.Track();
    }
    
    // Return a pointer to the parameter
    BaseParameter* GetParPointer() {
        return &parameter_;
    }
private:
	// References to parameter and proposal associated with step instance.
	Parameter<ParValueType>& parameter_;
	Proposal<ParValueType>& proposal_;
	boost::random::uniform_real_distribution<> uniform_;
	double alpha_; // Acceptance probability
	int naccept_; // The number of accepted steps
	int niter_; // The number of iterations performed
	int report_iter_; // The number of iterations until we print out the average acceptance rate
};

// Robust Adaptive Multivariate proposal for Metropolis-Hastings (RAM).
//
// Reference: Robust Adaptive Metropolis Algorithm with Coerced Acceptance Rate,
//			  M. Vihola, 2012, Statistics & Computing, 22, 997-1008

class AdaptiveMetro : public Step
{
public:
	// Constructor
	AdaptiveMetro(Parameter<arma::vec>& parameter, Proposal<double>& proposal,
				  arma::mat proposal_covar, double target_rate, int maxiter);
    
	std::string ParameterLabel() {
		return parameter_.Label();
	}
	
	std::string ParameterValue() {
		return parameter_.StringValue();
	}
	
	// Method to set the target acceptance rate
	void SetTargetRate(double target_rate) {
		target_rate_ = target_rate;
	}
	
	// Method to set the rate at which the step size sequence decays.
	void SetDecayRate(double gamma) {
		gamma_ = gamma;
	}
	
	// Method to determine whether a proposal is accepted
	bool Accept(arma::vec new_value, arma::vec old_value);
	
	// Method to perform the RAM step.
	void DoStep();

    // Return the current value of the Metropolis-Hastings ratio
    double GetMetroRatio() {
        return alpha_;
    }

    // Return the average acceptance rate thus far.
    double GetAcceptRate() {
        double arate = ((double)(naccept_)) / ((double)(niter_));
        return arate;
    }
    
    // Return the covariance matrix of the proposals
    arma::mat GetCovariance() {
        arma::mat covar = chol_factor_.t() * chol_factor_;
        return covar;
    }
    
    // Return if parameter is tracked.
    bool ParameterTrack() {
        return parameter_.Track();
    }
    
    // Return a pointer to the parameter
    BaseParameter* GetParPointer() {
        return &parameter_;
    }
	
private:
	/// References to parameter and proposal associated with step instance.
	Parameter<arma::vec>& parameter_;
	Proposal<double>& proposal_;
	boost::random::uniform_real_distribution<> uniform_;
	arma::mat chol_factor_; // Cholesky factor of proposal scale matrix
	double gamma_; // Rate of decay for step size update
	double target_rate_; // Target acceptance rate
	int niter_; // Number of iterations performed
	int naccept_; // Number of MHA proposals accepted
	int maxiter_; // Maximum number of iterations to update proposal scale matrix
	double alpha_; // Acceptance probability
};

// Class performing the exchange step used in Parallel Tempering
template <class ParValueType, class ParameterType>
class ExchangeStep : public Step
{
public:
    // Constructor
    ExchangeStep(Parameter<ParValueType>& parameter, int parameter_index, Ensemble<ParameterType>& ensemble,
                 int report_iter=-1) :
    parameter_(parameter), parameter_index_(parameter_index), ensemble_(ensemble), report_iter_(report_iter)
    {
        // Always propose swap between ensemble_(parameter_index) and ensemble(parameter_index-1), so
        // make sure parameter_index > 0.
        BOOST_ASSERT(parameter_index_ > 0);
        naccept_ = 0;
        niter_ = 0;
    }
    
	// Return string of parameter label
	std::string ParameterLabel() {
		return parameter_.Label();
	}
	
	// Return string representation of parameter value
	std::string ParameterValue() {
		return parameter_.StringValue();
	}
	
    // Do the exchange step.
    void DoStep() {
        
        // Get the log-posterior and the temperature for this parameter
        double this_logpost = parameter_.GetLogDensity();
        // Warmer temperature : this_temperature > other_temperature
        double this_temperature = parameter_.GetTemperature();
        
        // Do the same thing, but for the parameter we are trying to swap with
        double other_logpost = ensemble_[parameter_index_-1].GetLogDensity();
        double other_temperature = ensemble_[parameter_index_-1].GetTemperature();
        
        // Calculate the logarithm of the metropolis-hastings ratio
        double alpha = 1.0 / this_temperature * (other_logpost - this_logpost) +
        1.0 / other_temperature * (this_logpost - other_logpost);
        
        // Perform metropolis-hastings update
		double unif = uniform_(rng);
		alpha = std::min(exp(alpha), 1.0);
        if (!arma::is_finite(alpha)) {
            alpha = 0.0;
        }
        
		if (unif < alpha) {
            // Swap the parameter values
            ParValueType this_theta = parameter_.Value();
            parameter_.Save(ensemble_[parameter_index_-1].Value());
            ensemble_[parameter_index_-1].Save(this_theta);
            
            // Update the log-posterior values
            double this_logpost_new = other_logpost;
            parameter_.SetLogDensity(this_logpost_new);
            double other_logpost_new = this_logpost;
            ensemble_[parameter_index_-1].SetLogDensity(other_logpost_new);
            
            naccept_++;
		}
        
        niter_++;
        
		if (niter_ == report_iter_) {
			// Give report on average acceptance rate
			Report();
		}
        alpha_ = alpha;
    }
    
	// Report on acceptance rates since last report
	void Report() {
		double arate = ((double)(naccept_)) / ((double)(niter_));
		std::cout << "Average Exchange Acceptance Rate Since Last Report: " << arate << std::endl;
		niter_ = 0;
		naccept_ = 0;
	}

    // Return the current value of the Metropolis-Hastings ratio
    double GetMetroRatio() {
        return alpha_;
    }

    // Return if parameter is tracked.
    bool ParameterTrack() {
        return parameter_.Track();
    }
    
    // Return a pointer to the parameter
    BaseParameter* GetParPointer() {
        return &parameter_;
    }
    
private:
    Parameter<ParValueType>& parameter_; // The parameter associated with this step
    int parameter_index_; // The index of parameter_ in the ensemble
    Ensemble<ParameterType>& ensemble_; // The parameter ensemble
    int report_iter_; // Report on acceptance rates after this many iterations
    boost::random::uniform_real_distribution<> uniform_;
    int niter_; // The number of iterations performed since last report
    int naccept_; // The number of accepted exchanges since last report
    double alpha_; // Metropolis-hastings ratio
};

#endif /* defined(__yamcmc____steps__) */
