//
//  main.cpp
//  speed_comparison
//
//  Created by Brandon Kelly on 6/25/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#include <iostream>
#include "kfilter.hpp"
#include "carpack.hpp"
#include <time.h>

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

arma::mat carma_covariance(double sigsqr, arma:cx_vec ar_roots, arma::vec ma_coefs) {
    
}

int main(int argc, const char * argv[])
{
    int ny = {200, 400, 600, 1000};
    int ntests = 4;
    int ntries = 20;
    
    clock_t kalman_timer;
    clock_t inversion_timer;
    
    double mu = 10.0;
    double sigsqr = 1.4 * 1.4;
    arma::vec ma_coefs = {1.0, 2.3, 4.5, 0.2, 0.0, 0.3};
    arma::vec ar_roots_real = {-0.2, -0.2, -2.3, -2.3, -0.04, -0.04, -0.01};
    arma::vec ar_roots_imag = {1.2, -1.2, 0.01, -0.01, 0.5, -0.5, 0.0};
    arma::cx_vec ar_roots(ar_roots_real, ar_roots_imag);
    
    for (int k=0; k<ntests; k++) {
        std::cout << "Benchmark for " << ny[k] << " data points, average over " << ntries << " runs." << std::endl;
        
        arma::vec time = arma::linspace<arma::vec>(0.0, 10.0, ny[k]);
        arma::vec y = arma::randn<arma::vec>(ny[k]);
        arma::vec yerr = arma::ones<arma::vec>(ny[k]);
        
        CARMA carma(true, "carma", time, y, yerr, 7, 6);
        arma::vec theta = carma.StartingValue();
        
        KalmanFilterp kfilter(time, y, yerr);
        
        // first benchmark the kalman filter
        kalman_timer = clock();
        for (int t=0; t<ntries; t++) {
            kfilter.SetSigsqr(sigsqr);
            kfilter.SetOmega(ar_roots);
            kfilter.SetMA(ma_coefs);
            kfilter.Filter();
            double loglik = 0.0;
            for (int i=0; i<time.n_elem; i++) {
                double ycent = y(i) - kfilter.mean(i) - mu;
                logpost += -0.5 * log(kfilter.var(i)) - 0.5 * ycent * ycent / kfilter.var(i);
            }
        }
        kalman_timer = clock() - kalman_timer;
        double avg_seconds = double(kalman_timer) / CLOCKS_PER_SEC;
        std::cout << "Average for Kalman Filter is " << avg_seconds << " seconds."
        
        // now benchmark the direct inversion approach
        inversion_timer = clock();
        arma::mat carma_covar = carma_covariance(sigsqr, ar_roots, ma_coefs);
        
    }
    
    
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}

