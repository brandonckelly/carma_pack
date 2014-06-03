/*
 *  samplers.hpp
 *  yamcmc++
 *
 *  Created on 11/18/12 by
 *
 *     Dr. Brandon C. Kelly
 *     Department of Physics
 *     University of California, Santa Barbara
 *     (bckelly80@gmail.com)
 *
 *  Routines to perform Markov Chain Monte Carlo (MCMC).
 */

#ifndef __yamcmc____samplers__
#define __yamcmc____samplers__

// Standard includes
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <set>
// Boost includes
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/progress.hpp>
#include <boost/timer.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
// Local includes
#include "random.hpp"
#include "steps.hpp"

// Basic MCMC options. These options are generally specified at the command line.
struct MCMCOptions {
	/// Sample size. (iterations - burnin)/(thin + 1).
	int sample_size;
	/// Thinning interval
	int thin;
	/// Burn in period.
	int burnin;
	/// Chains.
	int chains;
	/// Data file
	std::string data_file;
	/// Out file to save parameter chains.
	std::string out_file;

    int getSampleSize() { return sample_size; }
    void setSampleSize(int s) {sample_size = s;}
    
    int getThin() { return thin; }
    void setThin(int t) {thin = t;}
    
    int getBurnin() { return burnin; }
    void setBurnin(int b) {burnin = b;}

    int getChains() { return chains; }
    void setChains(int c) {chains = c;}

    std::string getDataFileName() { return data_file; }
    void setDataFileName(std::string s) {data_file = s;}

    std::string getOutFileName() { return out_file; }
    void setOutFileName(std::string s) {out_file = s;}
   
};

/********************
 FUNCTION PROTOTYPES
 ********************/

// Function to get working directory (the directory that the executable
// was called from)
std::string get_initial_directory();

// Function to prompt user for MCMC parameters and return a structure
// containing those parameters.
MCMCOptions mcmc_options_prompt(std::string idirectory);

// Function to test if we can write to a file
bool write_to_file(std::string filename);

// Function to test if we can read a file
bool read_from_file(std::string filename);

// Function to test if two doubles are within the machine precision of each other.
bool approx_equal(double a, double b);

// MCMC sampler. The sampler is the main MCMC object that holds all the MCMC steps for each parameter.  Running the sampler
// performs MCMC sampling for the model, saving results, and displaying progress. After instantiating the sampler, users
// should add all the required steps using the Sampler::AddStep method, which places each step onto a stack. The entire
// sampling process is run using Sampler::Run.
class Sampler {
public:
    // Constructor to initialize sampler. Takes a MCMCOptions struct as input.
    Sampler(int sample_size, int burnin, int thin=1) : sample_size_(sample_size), burnin_(burnin), thin_(thin) {};
	
    // Method to add Step to Sampler execution stack.
   void AddStep(Step* step);
	
    // Run sampler for a specific number of iterations.
    void Iterate(int number_of_iterations, bool progress = false);
	
	// Run MCMC sampler.
   void Run(arma::vec init);
	
    // Return number of steps in one sampler iteration.
    int NumberOfSteps() {
        return steps_.size();
    }
    
    // Return number of tracked steps in one sampler iteration.
    int NumberOfTrackedSteps() {
        return tracked_names_.size();
    }
    
    // Return names of parameters that we are tracking
    std::set<std::string> GetTrackedNames() {
        return tracked_names_;
    }
    
    // Return map of pointers to tracked parameter objects
    std::map<std::string, BaseParameter*> GetTrackedParams() {
        return p_tracked_parameters_;
    }
    
    // Save the parameter values after a iteration to a file
    virtual void SaveValues();
    
protected:
    int sample_size_;
    int current_iter_;
    int burnin_;
    int thin_;
   boost::ptr_vector<Step> steps_;
    std::map<std::string, BaseParameter*> p_tracked_parameters_;
    std::set<std::string> tracked_names_;
};

#endif /* defined(__yamcmc____samplers__) */
