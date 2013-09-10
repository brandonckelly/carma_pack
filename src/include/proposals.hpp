//
//  proposals.h
//  yamcmc++
//
//  Created by Brandon Kelly on 3/2/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef __yamcmc____proposals__
#define __yamcmc____proposals__

#include <iostream>
// Local includes
#include "random.hpp"
#include "parameters.hpp"

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

// Abstract proposal class for Metropolis-Hastings sampler.
template <typename ProposalType>
class Proposal {
public:
    Proposal() {}
	virtual ProposalType Draw(ProposalType starting_value) = 0;
	virtual double LogDensity(ProposalType new_value, ProposalType starting_value) = 0;
};

//	Normal proposal for Metropolis-Hastings. Normal proposal draws from
//	N(starting_value, standard_deviation).
class NormalProposal : public Proposal<double> {
public:
	NormalProposal() {}
	
	// Constructor. standard_deviation is a tuning parameter for normal proposal.
	NormalProposal(double standard_deviation) : standard_deviation_(standard_deviation) {}
	
	//Draw from normal proposal
	double Draw(double starting_value);
	
	// Log probability of proposal new value, given starting value.
	double LogDensity(double new_value, double starting_value) {
		return 0.0; // Symmetric, doesn't matter.
	}
	
private:
	double standard_deviation_;
};


//  Student's t proposal for Metropolis-Hastings. Student's t proposal draws from
//	t(dof, starting_value, standard_deviation).
class StudentProposal : public Proposal<double> {
public:
	StudentProposal() {}
	
	// Constructor. standard_deviation is a tuning parameter for normal proposal.
	StudentProposal(double dof, double scale) :
    dof_(dof), scale_(scale) {}
	
	//Draw from normal proposal
	double Draw(double starting_value);
	
	// Log probability of proposal new value, given starting value.
	double LogDensity(double new_value, double starting_value) {
		return 0.0; // Symmetric, doesn't matter.
	}
	
private:
	double dof_;
	double scale_;
};


//	Multivariate Normal proposal for Metropolis-Hastings.
class MultiNormalProposal : public Proposal<arma::vec> {
public:
	MultiNormalProposal() {}
	
	// Constructor. covar is a tuning parameter for multivariate normal proposal.
	MultiNormalProposal(arma::mat covar) : covar_(covar) {}
	
	//Draw from multivariate normal proposal
	arma::vec Draw(arma::vec starting_value);
	
	// Log probability of proposal new value, given starting value.
	double LogDensity(arma::vec new_value, arma::vec starting_value) {
		return 0.0; // Symmetric, doesn't matter.
	}
	
private:
	arma::mat covar_;
};


//Log normal proposal for Metropolis-Hastings.
class LogNormalProposal : public Proposal<double> {
public:
    // Empy constructor
	LogNormalProposal() {}
    
	// Main constructor. Scale of steps are standard deviation of the logged parameter value:
    // logsd: Tuning parameter equal to the log standard deviation.
	LogNormalProposal(double logsd) : logsd_(logsd) {}
	
	// Draw from log normal proposal
    // starting_value: Starting value (unlogged mean of log normal).
	double Draw(double starting_value);
	
	// Log probability of proposal new value, given starting value.
	double LogDensity(double new_value, double starting_value) {
		double log_new_value = log(new_value);
		double logdens = -1.0 * log_new_value;
		return logdens;
	}
	
private:
	double logsd_;
};


/***************************************************************
 
 Proposals used for ensemble sampling, also called population MCMC.
 
 ****************************************************************/

// Base proposal class for ensemble (population) samplers.
template <class ParameterValueType, class ParameterType>
class EnsembleProposal: public Proposal<ParameterValueType> {
public:
    // Constructor
	EnsembleProposal(Ensemble<ParameterType>& ensemble, int parameter_index) :
    ensemble_(ensemble), parameter_index_(parameter_index) {}
	
	// Return a parameter object from the complementary ensemble
	virtual ParameterValueType GrabParameter() = 0;
	
	// Propose a new value of the parameter
	ParameterValueType Draw(ParameterValueType starting_value) {
		ParameterValueType p;
		return p;
	}
	
	// Return the logarithm of the proposal density, needed for the
	// Metropolis-Hastings update
	double LogDensity(ParameterValueType new_value, ParameterValueType starting_value) {
		return 0.0;
	}
	
protected:
	int parameter_index_; // Index of this parameter in the ensemble
    Ensemble<ParameterType>& ensemble_; // Reference to a ParameterEnsemble object
};


/*
 The stretch move of Goodman & Weare (2010, Comm. App. Math. & Comp. Sci., vol. 5, p65).
 For each Parameter object in the ensemble, this move randomly chooses another
 Parameter object. It then proposed a new value of the Parameter by randomly moving
 the parameter vector along the line connecting the two parameter vectors:
 
 X_k(iter+1) = X_j + Z * (X_k(iter) - X_j), Z ~ g(z)
 
 g(z) \propto 1 / sqrt(z), 1/a < z < a, a > 1.
 
 By default a = 2.
 */

// Currently Only valid for real vector types
template<class ParameterType>
class StretchProposal : public EnsembleProposal<arma::vec, ParameterType> {
public:
    // Constructor
    StretchProposal(Ensemble<ParameterType>& ensemble, int walker_index, double scaling_support=2.0) :
    EnsembleProposal<arma::vec, ParameterType>(ensemble, walker_index), scaling_support_(scaling_support)
    {
        // Set the bounds on the uniform distribution object
        int lowbound = 0;
        int upbound = this->ensemble_.size() - 1;
        boost::random::uniform_int_distribution<>::param_type unif_params(lowbound, upbound);
        uniform_.param(unif_params);
        other_parameter_index_ = -1;
    }
    
    // Set the support of the scaling parameter
    void SetScalingSupport(double scaling_support) {
        scaling_support_ = scaling_support; // scaling_support = a in notation above.
    }
    
    // Method to return the parameter value for a walker randomly chosen from the
    // complementary ensemble
    arma::vec GrabParameter()
    {
        do {
            // Randomly pick another walker from the complementary ensemble
            other_parameter_index_ = uniform_(rng);
        } while (other_parameter_index_ == this->parameter_index_);
        
        // Return the value of the parameter
        return this->ensemble_[other_parameter_index_].Value();
    }
    
    // Method to draw a proposal value of the parameter
    arma::vec Draw(arma::vec walker)
    {
        // First randomly choose another walker from the complementary ensemble
        arma::vec other_walker = GrabParameter();
        
        // Now randomly draw the scale parameter
        scale_ = RandGen.powerlaw(1.0 / scaling_support_, scaling_support_, -0.5);
        
        // Proposed value is along the line connecting the two parameters
        arma::vec new_value;
        new_value = other_walker + scale_ * (walker - other_walker);
        
        return new_value;
    }
    
    // Method to return the log-density of the transition kernel
    // starting_value -> new_value. This actually returns the logarithm
	// of the ratio of transition densities, since this is easier to compute
	// than the individual values of the densities of the transition kernels,
	// which would otherwise require the computation of an (nwalker - 1)-norm.
    double LogDensity(arma::vec new_value, arma::vec starting_value)
    {
		double logdens = 0.0; // Default value of the log-density
		
		// Get the current value of the parameter.
		arma::vec current_value = this->ensemble_[this->parameter_index_].Value();
		
		// Does the proposed parameter value = current parameter value?
        arma::vec etol(new_value.n_elem);
        etol.fill(1e-10);
		arma::uvec equal_vector = (arma::abs(current_value - new_value) < etol);
		
		if (arma::sum(equal_vector) == equal_vector.size()) {
			// We are calculating the density of the proposed -> current transition kernel,
			// so make this equal to scale^(nwalker-1).
			logdens = (new_value.size() - 1.0) * log(scale_);
		}
		// When proposed value != current value, then we are calculating the
		// density of the current -> proposed transition kernel. Force this to
		// be equal to zero so that the value of logdens computed above corresponds
		// to the logarithm of the ratio of the transition kernels.
        
        return logdens;
    }
    
private:
    double scaling_support_; // Support of the distribution of the scaling parameter (= a above)
	double scale_; // The most recent value of the scale parameter
	// Object to generate a random integer uniformly distributed over some range
	boost::random::uniform_int_distribution<> uniform_;
	// Current index for parameter in complementary ensemble used in the proposal
	int other_parameter_index_;
};


#endif /* defined(__yamcmc____proposals__) */
