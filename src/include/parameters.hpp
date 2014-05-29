//
//  parameters.hpp
//  yamcmc++
//
//  Created by Brandon Kelly on 4/15/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef yamcmc___parameters_hpp
#define yamcmc___parameters_hpp

// Standard includes
#include <iostream>
#include <string>
// Boost includes
#include <boost/ptr_container/ptr_vector.hpp>
// Local includes
#include "random.hpp"

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Global object for generating random variables from various distributions,
// instantiated in steps.cpp
extern RandomGenerator RandGen;

// This is the base Parameter class. It is abstract, so it should
// never be instantiated directly. Users should subclass the Parameter class,
// not BaseParameter class as the parameter value type is not specified in the
// BaseParameter class. Because of this, the MCMC step classes take a reference
// to the Parameter class.
class BaseParameter {
public:
    // Constructor
    BaseParameter() {}
    BaseParameter(bool track, std::string label, double temperature=1.0) :
    track_(track), label_(label), temperature_(temperature) {}
    
	// Return the current value of the log-posterior. Useful for
	// Metropolis steps so we don't have to compute the log-posterior
	// for the current value of the parameter more than once
	double GetLogDensity() {
		return log_posterior_;
	}
	
    // Method to directly set the log-posterior of the parameter. Useful for certain steps
    // when we do not need to recalculate the posterior. Used in the exchange step.
    void SetLogDensity(double logpost) {
        log_posterior_ = logpost;
    }
    
    // Return the value of the temperature used. Primarily used in tempered transitions.
    double GetTemperature() {
        return temperature_;
    }
    
	// Return a string representation of the parameter value.
	virtual std::string StringValue() {
		return " ";
	}
    
	// Parameter is tracked / saved. Return True if parameter is tracked.
	bool Track() {
		return track_;
	}
	
    // Set whether a parameter is tracked or not.
    void SetTracking(bool track) {
        track_ = track;
    }
    
	// Return the a string containing the parameter label. This is used to identify the parameter.
	std::string Label() {
		return label_;
	}
    
    // Set the size of the vector containing the MCMC samples. This will be overidden by the Parameter class.
    virtual void SetSampleSize(int sample_size) = 0;
    
    // Add a value to the set of MCMC samples. This will be overidden by the Parameter class.
    virtual void AddToSample(int current_iter) = 0;

    
protected:
	/// Should this variable be tracked?
	bool track_;
	/// Name of variable for tracking purposes.
	std::string label_;
    // Temperature value, used when doing tempered steps. By default this is one.
    double temperature_;
    double log_posterior_; // The log of the posterior distribution
};

// Templated abstract parameter class. Users should subclass the Parameter class because the
// Step classes need to know the type of the parameter values.
template<class ParValueType>
class Parameter : public BaseParameter {
public:
    // Constructor
    Parameter() {}
    Parameter(bool track, std::string label, double temperature=1.0) :
    BaseParameter(track, label, temperature) {}
    
	// Method to return the starting value for the parameter.
	virtual ParValueType StartingValue() = 0;

   // Method to set the starting value of the parameter
   virtual ParValueType SetStartingValue(ParValueType init) = 0;
	
	// Method to return the log of the probability density (plus constant).
	// value: Value of parameter to evaluate density at.
	virtual double LogDensity(ParValueType value) {
		return 0.0;
    }

	// Return a random draw from the posterior.
	// Random draw from posterior is called by GibbsStep.
	virtual ParValueType RandomPosterior() {
        ParValueType p;
		return p;
	}
	
	// Return the current value of the parameter.
	ParValueType Value() {
        return value_;
    }

	// Save a new value of the parameter.
	// new_value: New value to save.
	virtual void Save(ParValueType new_value) {
        value_ = new_value;
        log_posterior_ = LogDensity(new_value);
    }

    // Set the size of the vector containing the MCMC samples
    void SetSampleSize(int sample_size) {
        samples_.resize(sample_size);
        logposts_.resize(sample_size);
    }
    
    // Add a value to the set of MCMC samples
    void AddToSample(int current_iter) {
        // TODO: Should be able to replace this with an iterator for efficiency, since we probably
        // will always add values sequentially for MCMC samplers
        samples_[current_iter] = value_;
        logposts_[current_iter] = log_posterior_;
    }
    
    // Add a value and its log-posterior to the MCMC samples
    void AddToSample(int current_iter, ParValueType value, double logpost) {
        samples_[current_iter] = value;
        logposts_[current_iter] = logpost;
    }
    
    // Return a copy of the MCMC samples
    std::vector<ParValueType> GetSamples() {
        return samples_;
    }

    // Return a copy of the MCMC samples
    std::vector<double> GetLogLikes() {
        return logposts_;
    }
    
protected:
    ParValueType value_; // The current value of the parameter
    std::vector<ParValueType> samples_; // Vector containing the MCMC samples
    std::vector<double> logposts_; // Vector containing the posterior likelihoods
};

// This is the Ensemble class. It is basically a class
// containing a pointer container (boost::ptr_vector) that allows
// one to collect an ensemble of objects. This is the basis for
// the Ensemble MCMC samplers, which require both an ensemble of
// parameter and proposal objects.

template <class EnsembleType>
class Ensemble {
public:
    // Empty constructor
    Ensemble() {}
    
    // Add parameter object to the parameter ensemble
    void AddObject(EnsembleType* pObject) {
        the_objects_.push_back(pObject);
    }
    
    // Return the number of parameters in the ensemble
    int size() {
        return the_objects_.size();
    }
    
    // Access elements of the parameter ensemble. This done by overloading
    // the [] operator so that one can obtain a reference to a parameter
    // as parameter = ensemble[i].
    EnsembleType& operator [] (const int index)
    {
        return the_objects_[index];
    }
    
    EnsembleType const & operator [] (const int index) const
    {
        return the_objects_[index];
    }
    
    //private:
    boost::ptr_vector<EnsembleType> the_objects_;
};

#endif
