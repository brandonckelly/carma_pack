/*
 *  samplers.cpp
 *  yamcmc++
 *
 *  Created on 11/18/12 by
 *
 *     Dr. Brandon C. Kelly
 *     Department of Physics
 *     University of California, Santa Barbara
 *     (bckelly80@gmail.com)
 *
 */

// local includes
#include "include/samplers.hpp"

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

// Add Step to Sampler stack.
void Sampler::AddStep(Step* step) {
	// Add step to step stack.
	steps_.push_back(step);

    if (steps_.back().ParameterTrack()) {
        // Add parameter name to tracking stack
        std::string par_label = steps_.back().ParameterLabel();
        tracked_names_.insert(par_label);
        p_tracked_parameters_[par_label] = steps_.back().GetParPointer();
    }
}

// Run sampler for a specific number of iterations.
void Sampler::Iterate(int number_of_iterations, bool progress) {
	if(progress) {
		boost::progress_display show_progress(number_of_iterations);
		for(int iter = 0; iter < number_of_iterations; ++iter) {
			for(int i = 0; i < steps_.size(); ++i) {
				steps_[i].DoStep();
			}
			++show_progress;
		}
	}
	else {
		for(int iter = 0; iter < number_of_iterations; ++iter) {
			for(int i = 0; i < steps_.size(); ++i) {
				steps_[i].DoStep();
			}
		}
	}
}

//Run the MCMC sampling procedure, including burning in, sampling, thinning, displaying progress, and saving results.
void Sampler::Run(arma::vec init) {
	// Timer
	boost::timer timer;
	current_iter_ = 0;
    
    // Allocate memory for MCMC samples
    for (std::set<std::string>::iterator it=tracked_names_.begin(); it!=tracked_names_.end(); ++it) {
        std::string parameter_label = *it;
        p_tracked_parameters_[parameter_label]->SetSampleSize(sample_size_);
    }
    
	// Status of sampler...
	std::cout << "Running sampler..." << std::endl;
	std::cout << "Number of steps added: " << NumberOfSteps() << std::endl;
	std::cout << "Number of tracked steps added: " << NumberOfTrackedSteps() << std::endl;
			
	// Setting starting value:
	std::cout << "Setting starting values..." << std::endl;
	int npar = static_cast<Parameter<arma::vec> *>(steps_[0].GetParPointer())->Value().n_elem;
	bool useInit = (init.n_elem == npar);
	if (useInit) 
	   std::cout << " Using user-provided values" << std::endl;
	else
	   std::cout << " Drawn from priors" << std::endl;
	   
	for (unsigned int i = 0; i < steps_.size(); ++i) {
	   Parameter<arma::vec> *par = static_cast<Parameter<arma::vec> *>(steps_[i].GetParPointer());
	   if (useInit) 
	      par->Save(par->SetStartingValue(init));
	   else 
	      par->Save(par->StartingValue());

	   // Just print out first set of parameter values
	   if (i == 0) {
	      std::cout << " ...Initializing " << par->Value() << std::endl;
	   }
	}
	
	// Burn in
	std::cout << "Burning in... (" << burnin_ << " iterations)" << std::endl;
	Iterate(burnin_, true);
	
	std::cout << std::endl << "Sampling..." << std::endl;
	boost::progress_display show_progress(sample_size_);
	for (int i = 0; i < sample_size_; ++i) {
		Iterate(thin_);
		// Save data
        SaveValues();
 		// Show progress
		++show_progress;
        ++current_iter_;
	}
	
	// Dump seed to file for future use
	// std::cout << "Saving seed to file seed.txt\n";
	// std::string idirectory = get_initial_directory();
	// RandGen.SaveSeed(idirectory + "/seed.txt");
	// std::cout << "Total elapsed time: " << timer.elapsed() << " seconds" << std::endl;
}

// Method to save the current values of the parameters to a file.
void Sampler::SaveValues()
{
    for (std::set<std::string>::iterator it=tracked_names_.begin(); it!=tracked_names_.end(); ++it) {
        std::string parameter_label = *it;
        p_tracked_parameters_[parameter_label]->AddToSample(current_iter_);
    }
}

// Function to return the working directory (the directory that the executable
// was called from). Returns a string object.
std::string get_initial_directory()
{
    std::string idirectory;
	boost::filesystem::path ipath;
	ipath = boost::filesystem::current_path();
	idirectory = ipath.string(); // The initial working directory
	
    return idirectory;
}

// Prompt user for MCMC parameters. These include the data file, sample size,
// thinning interval, length of burn-in period, number of chains to run,
// seed for the random number generator, and name of file to save the
// parameter chains to. Return a structure containing this information.
MCMCOptions mcmc_options_prompt(std::string idirectory)
{
    int sample_size, thin, burnin, chains;
    bool open_flag;
    unsigned long random_seed;
    std::string out_filename, data_filename;
    chains = 1;
    
    do
    {	// Make sure data file exists and can be opened.
        data_filename = "";
        std::cout << "Name of Data File: ";
        std::cin >> data_filename;
        data_filename = idirectory + "/" + data_filename;
        std::cout << "\n";
        open_flag = read_from_file(data_filename);
        if (!open_flag) {
            std::cout << "Attempt to open file " << data_filename
            << " failed. Specify a different file.\n";
        }
        // Repeat until user inputs a valid file.
    } while (!open_flag);
    do
    {	// Make sure we can write to output file.
        out_filename = " ";
        std::cout << "Name of Output File: ";
        std::cin >> out_filename;
        out_filename = idirectory + "/" + out_filename;
        std::cout << "\n";
        open_flag = write_to_file(out_filename);
        if (!open_flag) {
            std::cout << "Cannot write to file " << out_filename
            << ". Specify a different file.\n";
        }
        // Repeat until user inputs a valid file.
    } while (!open_flag);
    do
    {   // Make sure sample size is at least one.
        std::cout << "Sample Size: ";
        std::cin >> sample_size;
        std::cout << "\n";
        if (sample_size < 1) {
            std::cout << "Sample size must be at least one.\n";
        }
    } while (sample_size < 1);
    do
    {	// Make sure thinning interval is at least one.
        std::cout << "Thinning interval (1 = no thinning): ";
        std::cin >> thin;
        std::cout << "\n";
        if (thin < 1) {
            std::cout << "Thinning interval must be at least one.\n";
        }
    } while (thin < 1);
    do
    {	// Make sure we have non-negative number of burn-in iterations.
        std::cout << "Number of Burn-in Iterations (no-thinning): ";
        std::cin >> burnin;
        std::cout << "\n";
        if (burnin < 0) {
            std::cout << "Number of burn-in iterations cannot be negative.\n";
        }
    } while (burnin < 0);
    
    std::cout << "Seed for Random Number Generator " << "\n";
    std::cout << "(0 for time stamp, 1 attempts to load from seed.txt file): ";
    std::cin >> random_seed;
    std::cout << "\n";
    
    // Now store the input MCMC parameters into a structure.
    MCMCOptions mcmc_options;
    
    mcmc_options.data_file = data_filename;
    mcmc_options.out_file = out_filename;
    mcmc_options.chains = chains;
    mcmc_options.sample_size = sample_size;
    mcmc_options.burnin = burnin;
    mcmc_options.thin = thin;
    
    // Load the seed for the random number generator. This depends
    // on the input...
    if (random_seed == 0) {
        // User input zero, use current time stamp.
        random_seed = time(NULL);
		RandGen.SetSeed(random_seed);
    }
    else if (random_seed == 1) {
        // User input 1, so try to load seed from file
        std::string seed_filename = idirectory + "/seed.txt";
        if (read_from_file(seed_filename)) {
            // The seed file exists, so get seed from it
			RandGen.RecoverSeed(seed_filename);
        }
        else {
            // Seed file does not exist, so just use time stamp
            std::cout << "The seed file " << seed_filename
            << " could not be found, just using current time stamp...\n";
            random_seed = time(NULL);
			RandGen.SetSeed(random_seed);
        }
    }
    else {
        // Use user input for the seed
		RandGen.SetSeed(random_seed);
    }
    
    return mcmc_options;
}

// Function to test if we can write to a file. Return a boolean variable.
bool write_to_file(std::string filename){
    std::ofstream file(filename.c_str());
    // Is the file open?
    bool open_flag = file.is_open();
    if (open_flag == true) {
        // Great! We can write to this file, so close it now.
        file.close();
    }
    return open_flag;
}

// Function to test if we can read a file. Returns a boolean variable.
bool read_from_file(std::string filename)
{
    std::ifstream file(filename.c_str());
    // Is the file open?
    bool open_flag = file.is_open();
    if (open_flag == true) {
        // Great! We can read from this file, so close it now.
        file.close();
    }
    return open_flag;
}
