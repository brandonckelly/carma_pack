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

arma::mat carma_covariance(double sigsqr, arma::cx_vec ar_roots, arma::vec ma_coefs, arma::vec time) {
    std::complex<double> car_var(0.0, 0.0);
    std::complex<double> denom(0.0, 0.0);
    std::complex<double> numer(0.0, 0.0);
    
    arma::mat varmat(time.n_elem, time.n_elem);
    for (int i=0; i<time.n_elem; i++) {
        for (int j=i; j<time.n_elem; j++) {
            // Calculate the variance of a CAR(p) process
            for (int k=0; k<ar_roots.n_elem; k++) {
                std::complex<double> denom_product(1.0,0.0);
                
                for (int l=0; l<ar_roots.n_elem; l++) {
                    if (l != k) {
                        denom_product *= (ar_roots(l) - ar_roots(k)) *
                        (std::conj(ar_roots(l)) + ar_roots(k));
                    }
                }
                denom = -2.0 * std::real(ar_roots(k)) * denom_product;
                
                int q = ma_coefs.n_elem;
                std::complex<double> ma_sum1(0.0,0.0);
                std::complex<double> ma_sum2(0.0,0.0);
                for (int l=0; l<q; l++) {
                    ma_sum1 += ma_coefs(l) * std::pow(ar_roots(k),l);
                    ma_sum2 += ma_coefs(l) * std::pow(-ar_roots(k),l);
                }
                numer = ma_sum1 * ma_sum2 * std::exp(ar_roots(k) * std::abs(time[j] - time[i]));
                
                car_var += numer / denom;
            }
            
            // Variance is real-valued, so only return the real part of CARMA_var.
            varmat(i,j) = car_var.real();
        }
    }
    
    return sigsqr * arma::symmatu(varmat);  // reflext upper triangle to lower triangle
}

int main(int argc, const char * argv[])
{
    int ny[4] = {200, 400, 600, 1000};
    int ntests = 4;
    int ntries = 20;
    
    clock_t kalman_timer;
    clock_t inversion_timer;
    
    double mu = 10.0;
    double sigsqr = 1.4 * 1.4;
    arma::vec ma_coefs = {1.0, 2.3, 4.5, 0.2, 0.0, 0.3, 0.0};
    arma::vec ar_roots_real = {-0.2, -0.2, -2.3, -2.3, -0.04, -0.04, -0.01};
    arma::vec ar_roots_imag = {1.2, -1.2, 0.01, -0.01, 0.5, -0.5, 0.0};
    arma::cx_vec ar_roots(ar_roots_real, ar_roots_imag);
    
    for (int k=0; k<ntests; k++) {
        std::cout << "Benchmark for " << ny[k] << " data points, average over " << ntries << " runs." << std::endl;
        
        arma::vec time = arma::linspace<arma::vec>(0.0, 10.0, ny[k]);
        arma::vec y = arma::randn<arma::vec>(ny[k]);
        arma::vec yerr = arma::ones<arma::vec>(ny[k]);
        
        CARMA carma(true, "carma", arma::conv_to<std::vector<double> >::from(time),
                    arma::conv_to<std::vector<double> >::from(y),
                    arma::conv_to<std::vector<double> >::from(yerr), 7, 6);
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
                loglik += -0.5 * ycent * ycent / kfilter.var(i);
            }
        }
        kalman_timer = clock() - kalman_timer;
        double avg_seconds = double(kalman_timer) / CLOCKS_PER_SEC;
        std::cout << "Average for Kalman Filter is " << avg_seconds << " seconds." << std::endl;
        
        // now benchmark the direct inversion approach
        inversion_timer = clock();
        for (int t=0; t<ntries; t++) {
            arma::mat carma_covar = carma_covariance(sigsqr, ar_roots, ma_coefs, time);
            arma::vec ycent = y - mu;
            double loglik2 = -0.5 * arma::as_scalar(ycent.t() * arma::inv_sympd(carma_covar) * ycent);
        }
        inversion_timer = clock() - inversion_timer;
        avg_seconds = double(inversion_timer) / CLOCKS_PER_SEC;
        std::cout << "Average for direct inversion is " << avg_seconds << " seconds." << std::endl << std::endl;
    }
}

