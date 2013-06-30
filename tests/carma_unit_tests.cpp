//
//  carma_unit_tests.cpp
//  carma_pack
//
//  Created by Brandon C. Kelly on 6/10/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include "carmcmc.hpp"
#include "carpack.hpp"
#include "kfilter.hpp"
#include <armadillo>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/chi_squared.hpp>

// Files containing simulated CAR(1) and CAR(5) time series, used for testing
std::string car1file("data/car1_test.dat");
std::string car5file("data/car5_test.dat");

// Compute the autocorrelation function of a series
arma::vec autocorr(arma::vec& y, int maxlag) {
    int ny = y.n_elem;
    arma::vec acorr = arma::zeros(maxlag);
    
    for (int lag=1; lag <= maxlag; lag++) {
        for (int i=0; i<ny-lag; i++) {
            acorr(lag-1) += y(i) * y(i+lag);
        }
    }
    
    double ssqr = 0.0;
    for (int i=0; i<ny; i++) {
        ssqr += y(i) * y(i);
    }
    
    return acorr / ssqr;
}

/*******************************************************************
                        TESTS FOR CAR1 CLASS
 *******************************************************************/

TEST_CASE("CAR1/constructor", "Make sure constructor sorts the time vector and removes duplicates.") {
    int ny = 100;
    arma::vec time0 = arma::linspace<arma::vec>(0.0, 100.0, ny);
    arma::vec y0 = arma::randn<arma::vec>(ny);
    arma::vec ysig = arma::zeros<arma::vec>(ny);

    // swap two elements so that time is out of order
    arma::vec time = time0;
    arma::vec y = y0;
    time(43) = time0(12);
    y(43) = y0(12);
    time(12) = time0(43);
    y(12) = y0(43);

    CAR1 car1_unordered(true, "CAR(1) - 1", time, y, ysig);
    
    // make sure CAR1 constructor sorted the time values
    time = car1_unordered.GetTime();
    REQUIRE(time(43) == time0(43));
    REQUIRE(time(12) == time0(12));
    arma::vec ycent = car1_unordered.GetTimeSeries();
    double ymean = arma::mean(y0);
    double frac_diff = std::abs(ycent(43) + ymean - y0(43)) / std::abs(y0(43));
    REQUIRE(frac_diff < 1e-8);
    frac_diff = std::abs(ycent(12) + ymean - y0(12)) / std::abs(y0(12));
    REQUIRE(frac_diff < 1e-8);
    
    // duplicate one of the elements of time
    time(43) = time(42);
    
    CAR1 car1_duplicate(true, "CAR(1) - 2", time, y, ysig);
    
    // make sure CAR1 constructor removed the duplicate value
    time = car1_duplicate.GetTime();
    REQUIRE(time.size() == (ny-1));
    REQUIRE(time(43) == time0(44)); // removed 43rd element from time vector
    ycent = car1_duplicate.GetTimeSeries();
    frac_diff = std::abs(ycent(43) + ymean - y0(44)) / std::abs(y0(44));
    REQUIRE(frac_diff < 1e-8);
}

TEST_CASE("CAR1/logpost_test", "Make sure the that CAR1.logpost_ == Car1.GetLogPost(theta) after running MCMC sampler") {
    int ny = 100;
    arma::vec time = arma::linspace<arma::vec>(0.0, 100.0, ny);
    arma::vec y = arma::randn<arma::vec>(ny);
    arma::vec ysig = 0.01 * arma::ones(ny);
    
    CAR1 car1_test(true, "CAR(1)", time, y, ysig);
    double max_stdev = 10.0 * arma::stddev(y); // For prior: maximum standard-deviation of CAR(1) process
    car1_test.SetPrior(max_stdev);
    
    // setup Robust Adaptive Metropolis step object
    StudentProposal tUnit(8.0, 1.0);
    arma::mat prop_covar(3,3);
    prop_covar.eye();
    int niter = 1000;
    double target_rate = 0.4;
    AdaptiveMetro RAM(car1_test, tUnit, prop_covar, target_rate, niter+1);
    RAM.Start();
    
    // perform a bunch of steps, which will update the car1_test.value_ and car1_test.log_posterior_ values.
    int logpost_neq_count = 0;
    for (int i=0; i<niter; i++) {
        RAM.DoStep();
        double logdens_stored = car1_test.GetLogDensity(); // stored value of log-posterior for current theta
        double logdens_computed = car1_test.LogDensity(car1_test.Value()); // explicitly calculate log-posterior for current theta
        if (std::abs(logdens_computed - logdens_stored) > 1e-10) {
            logpost_neq_count++; // count the number of time the two log-posterior values do not agree
        }
    }
    // make sure that saved logdensity is always equal to LogDensity(theta) for current thera value
    REQUIRE(logpost_neq_count == 0);
}

TEST_CASE("CAR5/logpost_test", "Make sure the that Car5.logpost_ == Car5.GetLogPost(theta) after running MCMC sampler") {
    int ny = 100;
    arma::vec time = arma::linspace<arma::vec>(0.0, 100.0, ny);
    arma::vec y = arma::randn<arma::vec>(ny);
    arma::vec ysig = 0.01 * arma::ones(ny);
    int p = 5;
    
    CARp car5_test(true, "CAR(5)", time, y, ysig, p);
    
    double max_stdev = 10.0 * arma::stddev(y); // For prior: maximum standard-deviation of CAR(1) process
    car5_test.SetPrior(max_stdev);
    
    // setup Robust Adaptive Metropolis step object
    StudentProposal tUnit(8.0, 1.0);
    arma::mat prop_covar(p+2,p+2);
    prop_covar.eye();
    int niter = 1000;
    double target_rate = 0.4;
    AdaptiveMetro RAM(car5_test, tUnit, prop_covar, target_rate, niter+1);
    RAM.Start();
    
    // perform a bunch of steps, which will update the car1_test.value_ and car1_test.log_posterior_ values.
    int logpost_neq_count = 0;
    for (int i=0; i<niter; i++) {
        RAM.DoStep();
        double logdens_stored = car5_test.GetLogDensity(); // stored value of log-posterior for current theta
        double logdens_computed = car5_test.LogDensity(car5_test.Value()); // explicitly calculate log-posterior for current theta
        if (std::abs(logdens_computed - logdens_stored) > 1e-10) {
            logpost_neq_count++; // count the number of time the two log-posterior values do not agree
        }
    }
    // make sure that saved logdensity is always equal to LogDensity(theta) for current thera value
    REQUIRE(logpost_neq_count == 0);
}

TEST_CASE("CAR1/prior_bounds", "Make sure CAR1::LogDensity returns -infinty when prior bounds are violated") {
    int ny = 100;
    arma::vec time = arma::linspace<arma::vec>(0.0, 100.0, ny);
    arma::vec y = arma::randn<arma::vec>(ny);
    arma::vec ysig = 0.01 * arma::ones(ny);
    
    CAR1 car1_test(true, "CAR(1)", time, y, ysig);
    double max_stdev = 10.0 * arma::stddev(y); // For prior: maximum standard-deviation of CAR(1) process
    car1_test.SetPrior(max_stdev);

    // prior bounds on omega
    double max_freq = 10.0;
	double min_freq = 1.0 / (10.0 * time.max());

    arma::vec bad_theta(3); // parameter value will violated the prior bounds
    double measerr_scale = 1.0;
    double omega = 2.0 * max_freq;
    double sigma = max_stdev / 10.0 * sqrt(2.0 * omega);
    bad_theta << sigma << measerr_scale << log(omega);
    
    double logpost = car1_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    
    omega = min_freq / 2.0;
    bad_theta(2) = log(omega);
    logpost = car1_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    
    omega = 1.0;
    sigma = -1.0;
    bad_theta(0) = sigma;
    bad_theta(2) = log(omega);
    logpost = car1_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);

    sigma = 100.0 * max_stdev;
    bad_theta(0) = sigma;
    logpost = car1_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);

    sigma = 1.0;
    bad_theta(0) = sigma;
    measerr_scale = 0.1;
    bad_theta(1) = measerr_scale;
    logpost = car1_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);

    measerr_scale = 4.0;
    bad_theta(1) = measerr_scale;
    logpost = car1_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
}

TEST_CASE("CAR5/prior_bounds", "Make sure CARp::LogDensity return -infinity when prior bounds are violated") {
    int ny = 100;
    arma::vec time = arma::linspace<arma::vec>(0.0, 100.0, ny);
    arma::vec y = arma::randn<arma::vec>(ny);
    arma::vec ysig = 0.01 * arma::ones(ny);
    int p = 9;
    
    CARp car9_test(true, "CAR(9)", time, y, ysig, p);
    double max_stdev = 10.0 * arma::stddev(y); // For prior: maximum standard-deviation of CAR(1) process
    car9_test.SetPrior(max_stdev);
    
    // prior bounds on lorentzian parameters
    double max_freq = 10.0;
	double min_freq = 1.0 / (10.0 * time.max());
    
    arma::vec bad_theta(p+2); // parameter value will violated the prior bounds
    bad_theta = car9_test.StartingValue();

    bad_theta = car9_test.StartingValue();
    bad_theta(0) = 2.0 * max_stdev;
    double logpost = car9_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    bad_theta(0) = -1.0;
    logpost = car9_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    
    bad_theta(0) = max_stdev / 10.0;
    bad_theta(1) = 0.1;
    logpost = car9_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    bad_theta(1) = 4.0;
    logpost = car9_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    
    bad_theta(1) = 1.0;
    int nbad_width = 0;
    for (int j=0; j<p/2; j++) {
        double qpo_width = bad_theta(3+2*j);
        bad_theta(3+2*j) = log(min_freq / 2.0);
        logpost = car9_test.LogDensity(bad_theta);
        if (logpost != -1.0 * arma::datum::inf) {
            nbad_width++;
        }
        bad_theta(3+2*j) = log(2.0 * max_freq);
        logpost = car9_test.LogDensity(bad_theta);
        if (logpost != -1.0 * arma::datum::inf) {
            nbad_width++;
        }
        bad_theta(3+2*j) = qpo_width;
    }
    REQUIRE(nbad_width == 0);
    
    double qpo_cent = bad_theta(2);
    bad_theta(2) = log(2.0 * max_freq);
    logpost = car9_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    bad_theta(2) = qpo_cent;
    qpo_cent = bad_theta(2+2*(p/2-1));
    bad_theta(2+2*(p/2-1)) = log(min_freq / 2.0);
    logpost = car9_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    bad_theta(2+2*(p/2-1)) = qpo_cent;
    int nbad_cent = 0;
    for (int j=1; j<p/2; j++) {
        // violate the ordering of the lorentzian centroids
        qpo_cent = bad_theta(2+2*j);
        bad_theta(2+2*j) = log(1.1) + bad_theta(2+2*(j-1));
        logpost = car9_test.LogDensity(bad_theta);
        if (logpost != -1.0 * arma::datum::inf) {
            nbad_cent++;
        }
        bad_theta(2+2*j) = qpo_cent;
    }
    REQUIRE(nbad_cent == 0);
}

TEST_CASE("CAR5/carp_variance", "Test the CARp::Variance method") {
    // generate some data
    int ny = 100;
    arma::vec time = arma::linspace<arma::vec>(0.0, 100.0, ny);
    arma::vec y = arma::randn<arma::vec>(ny);
    arma::vec ysig = 0.01 * arma::ones(ny);
    
    // CAR(5) process parameters
    double qpo_width[3] = {0.01, 0.01, 0.002};
    double qpo_cent[2] = {0.2, 0.02};
    double sigma = 2.3;
    int p = 5;
    
    // Construct the complex vector of roots of the characteristic polynomial:
    // alpha(s) = s^p + alpha_1 s^{p-1} + ... + alpha_{p-1} s + alpha_p
    arma::cx_vec alpha_roots(p);
    for (int i=0; i<p/2; i++) {
        alpha_roots(2*i) = std::complex<double> (-1.0 * qpo_width[i],qpo_cent[i]);
        alpha_roots(2*i+1) = std::conj(alpha_roots(2*i));
    }
	
    if ((p % 2) == 1) {
        // p is odd, so add in additional low-frequency component
        alpha_roots(p-1) = std::complex<double> (-1.0 * qpo_width[p/2], 0.0);
    }
    
    alpha_roots *= 2.0 * arma::datum::pi;
    
    CARp car5_process(true, "CAR(5)", time, y, ysig, p);
    
    double model_var = car5_process.Variance(alpha_roots, sigma);
    double model_var0 = 218432.09642016294; // known variance, computed from python module carma_pack
    double frac_diff = std::abs(model_var - model_var0) / std::abs(model_var0);
    REQUIRE(frac_diff < 1e-8);
}

TEST_CASE("CAR1/kalman_filter", "Test the Kalman Filter for a CAR(1) process") {
    // first grab the simulated Gaussian CAR(1) data set
    arma::mat car1_data;
    car1_data.load(car1file, arma::raw_ascii);
    
    arma::vec time = car1_data.col(0);
    arma::vec y = car1_data.col(1);
    arma::vec yerr = car1_data.col(2);
    int ny = y.n_elem;
    
    // CAR(1) process parameters
    double tau = 100.0;
    double omega = 1.0 / tau;
    double sigmay = 2.3;
    double sigma = sigmay * sqrt(2.0 / tau);
    double measerr_scale = 1.0;
    arma::vec theta(3);
    theta << sigma << measerr_scale << log(omega);
    
    CAR1 car1_process(true, "CAR(1)", time, y, yerr);
    
    // Compute and grab the kalman filter
    car1_process.KalmanFilter(theta);
    arma::vec kmean = car1_process.GetKalmanMean();
    arma::vec kvar = car1_process.GetKalmanVariance();
    
    // Compute the standardized residuals of the time series
    arma::vec sresid = (y - arma::mean(y) - kmean) / arma::sqrt(kvar);
    
    // First do simple test on mean and variance of standardized residuals
    //REQUIRE(std::abs(arma::mean(sresid)) < 3.0 / sqrt(ny));
    //REQUIRE(std::abs(arma::var(sresid) - 1.0) < 3.0 * sqrt(2.0 * ny) / ny);
    
    // Test that the standardized residuals are consistent with having a standard normal distribution using
    // the Anderson-Darling test statistic
    arma::vec sorted_sresid = arma::sort(sresid);
    boost::math::normal snorm;
    arma::vec snorm_cdf(ny);
    for (int i=0; i<ny; i++) {
        // compute the standard normal CDF of the standardized residuals
        snorm_cdf(i) = boost::math::cdf(snorm, sorted_sresid(i));
    }
    
    double AD_sum = 0.0;
    for (int i=0; i<ny; i++) {
        // compute the Anderson-Darling statistic
        AD_sum += (2.0 * (i+1) - 1) / ny * (log(snorm_cdf(i)) + log(1.0 - snorm_cdf(ny-1-i)));
    }
    double AD_stat = -ny - AD_sum;
    REQUIRE(AD_stat < 3.857); // critical value for 1% significance level
 
    // Now test that the autocorrelation function of the standardized residuals is consistent with a white noise process
    int maxlag = 100;
    arma::vec acorr_sresid = autocorr(sresid, maxlag);
    double acorr_95bound = 1.96 / sqrt(ny); // find number of autocorr values outside of 95% confidence interval
    int out_of_bounds = arma::accu(arma::abs(acorr_sresid) > acorr_95bound);
    REQUIRE(out_of_bounds < 11); // 99% significance level for binomial distribution with n = 100 and p = 0.05

    double max_asqr = arma::max(acorr_sresid % acorr_sresid);
    boost::math::chi_squared chisqr(1); // square of ACF has a chi-squared distribution with two DOF
    double chisqr_cdf = boost::math::cdf(chisqr, max_asqr * ny);
    double max_asqr_cdf = std::pow(chisqr_cdf, maxlag); // CDF of maximum of maxlag random variables having a chi-square distribution
    REQUIRE(max_asqr_cdf < 0.99); // test fails if probability of max(ACF) < 1%
    
    // Finally, test that the autocorrelation function of the square of the residuals is consistent with a white noise process
    arma::vec sres_sqr = sresid % sresid;
    sres_sqr -= arma::mean(sres_sqr);
    arma::vec acorr_ssqr = autocorr(sres_sqr, maxlag);
    out_of_bounds = arma::accu(arma::abs(acorr_ssqr) > acorr_95bound);
    REQUIRE(out_of_bounds < 11); // 99% significance level for binomial distribution with n = 100 and p = 0.05
    
    max_asqr = arma::max(acorr_ssqr % acorr_ssqr);
    chisqr_cdf = boost::math::cdf(chisqr, max_asqr * ny);
    max_asqr_cdf = std::pow(chisqr_cdf, maxlag); // CDF of maximum of maxlag random variables having a chi-square distribution
    REQUIRE(max_asqr_cdf < 0.99); // test fails if probability of max(ACF) < 1%
}

TEST_CASE("CAR5/kalman_filter", "Test the Kalman Filter for a CAR(5) process") {
    // first grab the simulated Gaussian CAR(1) data set
    arma::mat car5_data;
    car5_data.load(car5file, arma::raw_ascii);
    
    arma::vec time = car5_data.col(0);
    arma::vec y = car5_data.col(1);
    arma::vec yerr = car5_data.col(2);
    int ny = y.n_elem;
    
    // CAR(5) process parameters
    double qpo_width[3] = {0.01, 0.01, 0.002};
    double qpo_cent[2] = {0.2, 0.02};
    double sigmay = 2.3;
    double measerr_scale = 1.0;
    int p = 5;
    
    // Create the parameter vector, theta
	arma::vec theta(p+2);
    theta(0) = sigmay;
	theta(1) = measerr_scale;
    for (int i=0; i<p/2; i++) {
        theta(2+2*i) = log(qpo_cent[i]);
        theta(3+2*i) = log(qpo_width[i]);
    }
    // p is odd, so add in additional value of lorentz_width
    theta(p+1) = log(qpo_width[p/2]);
    
    CARp car5_process(true, "CAR(5)", time, y, yerr, p);
    
    // Compute and grab the kalman filter
    car5_process.KalmanFilter(theta);
    arma::vec kmean = car5_process.GetKalmanMean();
    arma::vec kvar = car5_process.GetKalmanVariance();
    
    // Compute the standardized residuals of the time series
    arma::vec sresid = (y - arma::mean(y) - kmean) / arma::sqrt(kvar);
    
    // First do simple test on mean and variance of standardized residuals
    REQUIRE(std::abs(arma::mean(sresid)) < 3.0 / sqrt(ny));
    REQUIRE(std::abs(arma::var(sresid) - 1.0) < 3.0 * sqrt(2.0 * ny) / ny);
    
    // Test that the standardized residuals are consistent with having a standard normal distribution using
    // the Anderson-Darling test statistic
    arma::vec sorted_sresid = arma::sort(sresid);
    boost::math::normal snorm;
    arma::vec snorm_cdf(ny);
    for (int i=0; i<ny; i++) {
        // compute the standard normal CDF of the standardized residuals
        snorm_cdf(i) = boost::math::cdf(snorm, sorted_sresid(i));
    }

    double AD_sum = 0.0;
    for (int i=0; i<ny; i++) {
        // compute the Anderson-Darling statistic
        AD_sum += (2.0 * (i+1) - 1) / ny * (log(snorm_cdf(i)) + log(1.0 - snorm_cdf(ny-1-i)));
    }
    double AD_stat = -ny - AD_sum;
    REQUIRE(AD_stat < 3.857); // critical value for 1% significance level
    
    // Now test that the autocorrelation function of the standardized residuals is consistent with a white noise process
    int maxlag = 100;
    arma::vec acorr_sresid = autocorr(sresid, maxlag);
    double acorr_95bound = 1.96 / sqrt(ny); // find number of autocorr values outside of 95% confidence interval
    int out_of_bounds = arma::accu(arma::abs(acorr_sresid) > acorr_95bound);
    REQUIRE(out_of_bounds < 11); // 99% significance level for binomial distribution with n = 100 and p = 0.05
    
    double max_asqr = arma::max(acorr_sresid % acorr_sresid);
    boost::math::chi_squared chisqr(1); // square of ACF has a chi-squared distribution with two DOF
    double chisqr_cdf = boost::math::cdf(chisqr, max_asqr * ny);
    double max_asqr_cdf = std::pow(chisqr_cdf, maxlag); // CDF of maximum of maxlag random variables having a chi-square distribution
    REQUIRE(max_asqr_cdf < 0.99); // test fails if probability of max(ACF) < 1%
    
    // Finally, test that the autocorrelation function of the square of the residuals is consistent with a white noise process
    arma::vec sres_sqr = sresid % sresid;
    sres_sqr -= arma::mean(sres_sqr);
    arma::vec acorr_ssqr = autocorr(sres_sqr, maxlag);
    out_of_bounds = arma::accu(arma::abs(acorr_ssqr) > acorr_95bound);
    REQUIRE(out_of_bounds < 11); // 99% significance level for binomial distribution with n = 100 and p = 0.05
    
    max_asqr = arma::max(acorr_ssqr % acorr_ssqr);
    chisqr_cdf = boost::math::cdf(chisqr, max_asqr * ny);
    max_asqr_cdf = std::pow(chisqr_cdf, maxlag); // CDF of maximum of maxlag random variables having a chi-square distribution
    REQUIRE(max_asqr_cdf < 0.99); // test fails if probability of max(ACF) < 1%
}

TEST_CASE("CAR1/mcmc_sampler", "Test RunEmsembleCarSampler on CAR(1) model") {
    std::cout << std::endl;
    std::cout << "Running test of MCMC sampler for CAR(1) model..." << std::endl << std::endl;
    
    // first grab the simulated Gaussian CAR(1) data set
    arma::mat car1_data;
    car1_data.load(car1file, arma::raw_ascii);
    
    arma::vec time = car1_data.col(0);
    arma::vec y = car1_data.col(1);
    arma::vec yerr = car1_data.col(2);
    
    // MCMC parameters
    int carp_order = 1;
    int nwalkers = 10;
    int sample_size = 100000;
    int burnin = 50000;
    
    // run the MCMC sampler
    std::pair<std::vector<arma::vec>, std::vector<double> > mcmc_out;
    mcmc_out = RunEnsembleCarSampler(sample_size, burnin, time, y, yerr, carp_order, nwalkers);
    std::vector<arma::vec> mcmc_sample;
    mcmc_sample = mcmc_out.first;
    
    // True CAR(1) process parameters
    double tau = 100.0;
    double omega = 1.0 / tau;
    double sigmay = 2.3;
    double sigma = sigmay * sqrt(2.0 / tau);
    double measerr_scale = 1.0;
    
    arma::vec omega_samples(mcmc_sample.size());
    arma::vec sigma_samples(mcmc_sample.size());
    arma::vec scale_samples(mcmc_sample.size());
    for (int i=0; i<mcmc_sample.size(); i++) {
        sigma_samples(i) = log(mcmc_sample[i](0));
        scale_samples(i) = mcmc_sample[i](1);
        omega_samples(i) = mcmc_sample[i](2);
    }
    
    // Make sure true parameters are within 3sigma of the marginal posterior means
    double sigma_zscore = (arma::mean(sigma_samples) - log(sigma)) / arma::stddev(sigma_samples);
    CHECK(std::abs(sigma_zscore) < 3.0);
    double scale_zscore = (arma::mean(scale_samples) - measerr_scale) / arma::stddev(scale_samples);
    CHECK(std::abs(scale_zscore) < 3.0);
    double omega_zscore = (arma::mean(omega_samples) - log(omega)) / arma::stddev(omega_samples);
    CHECK(std::abs(omega_zscore) < 3.0);
}

TEST_CASE("CAR5/mcmc_sampler", "Test RunEmsembleCarSampler on CAR(5) model") {
    std::cout << std::endl;
    std::cout << "Running test of MCMC sampler for CAR(5) model..." << std::endl << std::endl;
    
    // first grab the simulated Gaussian CAR(5) data set
    arma::mat car5_data;
    car5_data.load(car5file, arma::raw_ascii);
    
    arma::vec time = car5_data.col(0);
    arma::vec y = car5_data.col(1);
    arma::vec yerr = car5_data.col(2);
    
    // MCMC parameters
    int carp_order = 5;
    int nwalkers = 10;
    int sample_size = 100000;
    int burnin = 50000;
    
    // run the MCMC sampler
    std::pair<std::vector<arma::vec>, std::vector<double> > mcmc_out;
    mcmc_out = RunEnsembleCarSampler(sample_size, burnin, time, y, yerr, carp_order, nwalkers);
    std::vector<arma::vec> mcmc_sample;
    mcmc_sample = mcmc_out.first;
    
    // True CAR(5) process parameters
    double qpo_width[3] = {0.01, 0.01, 0.002};
    double qpo_cent[2] = {0.2, 0.02};
    double sigmay = 2.3;
    double measerr_scale = 1.0;
    int p = 5;
    
    // Create the parameter vector, theta
	arma::vec theta(p+2);
    theta(0) = log(sigmay);
	theta(1) = measerr_scale;
    for (int i=0; i<p/2; i++) {
        theta(2+2*i) = log(qpo_cent[i]);
        theta(3+2*i) = log(qpo_width[i]);
    }
    // p is odd, so add in additional value of lorentz_width
    theta(p+1) = log(qpo_width[p/2]);

    arma::vec sigma_samples(mcmc_sample.size());
    arma::vec scale_samples(mcmc_sample.size());
    arma::mat ar_samples(mcmc_sample.size(),p);
    for (int i=0; i<mcmc_sample.size(); i++) {
        sigma_samples(i) = log(mcmc_sample[i](0));
        scale_samples(i) = mcmc_sample[i](1);
        for (int j=0; j<p; j++) {
            ar_samples(i,j) = mcmc_sample[i](j+2);
        }
    }
    
    // Make sure true parameters are within 3sigma of the marginal posterior means
    double sigma_zscore = (arma::mean(sigma_samples) - log(sigmay)) / arma::stddev(sigma_samples);
    CHECK(std::abs(sigma_zscore) < 3.0);
    double scale_zscore = (arma::mean(scale_samples) - measerr_scale) / arma::stddev(scale_samples);
    CHECK(std::abs(scale_zscore) < 3.0);
    for (int j=0; j<p; j++) {
        double ar_zscore = (arma::mean(ar_samples.col(j)) - theta(2+j)) / arma::stddev(ar_samples.col(j));
        CHECK(std::abs(ar_zscore) < 3.0);
    }
}