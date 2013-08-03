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
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/distributions/chi_squared.hpp>

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Files containing simulated CAR(1) and CAR(5) time series, used for testing
std::string car1file("data/car1_test.dat");
std::string car5file("data/car5_test.dat");
std::string zcarmafile("data/zcarma5_test.dat");
std::string carmafile("data/carma_test.dat");

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

TEST_CASE("startup/rng_seed", "Set the seed for the random number generator for reproducibility.") {
    rng.seed(123456);
}

TEST_CASE("KalmanFilter/constructor", "Make sure constructor sorts the time vector and removes duplicates.") {
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

    double sigsqr = 1.0;
    double omega = 2.0;
    
    KalmanFilter1 Kfilter(time, y, ysig, sigsqr, omega);
    
    // make sure KalmanFilter1 constructor sorted the time values
    time = Kfilter.GetTime();
    REQUIRE(time(43) == time0(43));
    REQUIRE(time(12) == time0(12));
    arma::vec y2 = Kfilter.GetTimeSeries();
    double frac_diff = std::abs(y2(43) - y0(43)) / std::abs(y0(43));
    REQUIRE(frac_diff < 1e-8);
    frac_diff = std::abs(y2(12) - y0(12)) / std::abs(y0(12));
    REQUIRE(frac_diff < 1e-8);
    
    // duplicate one of the elements of time
    time(43) = time(42);
    
    KalmanFilter1 Kfilter_dup(time, y, ysig, sigsqr, omega);
    
    // make sure CAR1 constructor removed the duplicate value
    time = Kfilter_dup.GetTime();
    REQUIRE(time.size() == (ny-1));
    REQUIRE(time(43) == time0(44)); // removed 43rd element from time vector
    y2 = Kfilter_dup.GetTimeSeries();
    frac_diff = std::abs(y2(43)- y0(44)) / std::abs(y0(44));
    REQUIRE(frac_diff < 1e-8);
}

TEST_CASE("KalmanFilter1/Filter", "Test the Kalman Filter for a CAR(1) process") {
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
    double sigsqr = sigma * sigma;

    KalmanFilter1 Kfilter1(time, y, yerr, sigsqr, omega);
    
    // First test that the Kalman Filter is correctly initialized after reseting it
    Kfilter1.Reset();
    arma::vec kmean = Kfilter1.mean;
    REQUIRE(kmean(0) == 0.0);
    arma::vec kvar = Kfilter1.var;
    double kvar_expected = sigmay * sigmay + yerr(0) * yerr(0);
    REQUIRE(std::abs(kvar(0) - kvar_expected) < 1e-10);
    
    // Now test the one-step prediction
    Kfilter1.Update();
    kmean = Kfilter1.mean;
    kvar = Kfilter1.var;
    double sresid1 = (y(1) - kmean(1)) / sqrt(kvar(1));
    REQUIRE(std::abs(sresid1) < 3.0);
    
    // Compute and grab the kalman filter
    Kfilter1.Filter();
    kmean = Kfilter1.mean;
    kvar = Kfilter1.var;
    
    // Compute the standardized residuals of the time series
    arma::vec sresid = (y - kmean) / arma::sqrt(kvar);
    
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

TEST_CASE("KalmanFilter1/Predict", "Test interpolation/extrapolation for a CAR(1) process") {
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
    double sigsqr = sigma * sigma;
    
    KalmanFilter1 Kfilter1(time, y, yerr, sigsqr, omega);
    Kfilter1.Filter();
    
    // first test forecasting
    double tpredict = time(ny-1) + 0.14536 * tau;
    std::pair<double, double> kpredict = Kfilter1.Predict(tpredict);
    double pmean = kpredict.first;
    double pvar = kpredict.second;
    
    // construct covariance matrix of (tpredict,time)
    arma::mat covar(ny+1,ny+1);
    for (int i=0; i<ny+1; i++) {
        for (int j=0; j<ny+1; j++) {
            double timei, timej;
            if (i == 0) {
                timei = tpredict;
            } else {
                timei = time(i-1);
            }
            if (j == 0) {
                timej = tpredict;
            } else {
                timej = time(j-1);
            }
            double dt = std::abs(timei - timej);
            covar(i,j) = sigmay * sigmay * std::exp(-dt / tau);
            if (i == j && i > 0) {
                // add contribution from measurement errors
                covar(i,j) += yerr(i-1) * yerr(i-1);
            }
        }
    }
    
    covar = arma::symmatl(covar);
    
    // calculate prediction mean and variance the slow way
    double pmean_slow, pvar_slow;
    arma::mat subvar_inv = arma::inv(arma::sympd(covar.submat(1, 1, ny, ny)));
    arma::rowvec subcov = covar.submat(0,1,0,ny);
    pmean_slow = arma::as_scalar(subcov * subvar_inv * y);
    pvar_slow = covar(0,0) - arma::as_scalar(subcov * subvar_inv * subcov.t());
    
    // make sure predicted mean and variance computed from the kalman filter is equal to that
    // computed the slow way from the properties of the normal distribution
    double frac_diff = std::abs(pmean_slow - pmean) / std::abs(pmean_slow);
    REQUIRE(frac_diff < 1e-8);
    frac_diff = std::abs(pvar_slow - pvar) / std::abs(pvar_slow);
    REQUIRE(frac_diff < 1e-8);
    
    // now test backcasting
    tpredict = time(0) - 0.34561 * tau;
    kpredict = Kfilter1.Predict(tpredict);
    pmean = kpredict.first;
    pvar = kpredict.second;
    
    covar(0,arma::span(1,ny)) = sigmay * sigmay * exp(-arma::abs(tpredict - time.t()) / tau);
    covar(arma::span(1,ny), 0) = sigmay * sigmay * exp(-arma::abs(tpredict - time) / tau);
    covar = arma::symmatu(covar);
    subcov = covar.submat(0,1,0,ny);
    pmean_slow = arma::as_scalar(subcov * subvar_inv * y);
    pvar_slow = covar(0,0) - arma::as_scalar(subcov * subvar_inv * subcov.t());
    
    // make sure predicted mean and variance computed from the kalman filter is equal to that
    // computed the slow way from the properties of the normal distribution
    frac_diff = std::abs(pmean_slow - pmean) / std::abs(pmean_slow);
    REQUIRE(frac_diff < 1e-8);
    frac_diff = std::abs(pvar_slow - pvar) / std::abs(pvar_slow);
    REQUIRE(frac_diff < 1e-8);

    // finally, test interpolation    
    tpredict = arma::mean(time);
    Kfilter1.Reset();
    kpredict = Kfilter1.Predict(tpredict);
    pmean = kpredict.first;
    pvar = kpredict.second;
        
    covar(0,arma::span(1,ny)) = sigmay * sigmay * exp(-arma::abs(tpredict - time.t()) / tau);
    covar(arma::span(1,ny), 0) = sigmay * sigmay * exp(-arma::abs(tpredict - time) / tau);
    covar = arma::symmatu(covar);
    subcov = covar.submat(0,1,0,ny);
    pmean_slow = arma::as_scalar(subcov * subvar_inv * y);
    pvar_slow = covar(0,0) - arma::as_scalar(subcov * subvar_inv * subcov.t());
    
    // make sure predicted mean and variance computed from the kalman filter is equal to that
    // computed the slow way from the properties of the normal distribution
    frac_diff = std::abs(pmean_slow - pmean) / std::abs(pmean_slow);
    REQUIRE(frac_diff < 1e-8);
    frac_diff = std::abs(pvar_slow - pvar) / std::abs(pvar_slow);
    REQUIRE(frac_diff < 1e-8);
}

TEST_CASE("KalmanFilterp/Filter", "Test the Kalman Filter for a CAR(5) process") {
    // first grab the simulated Gaussian CAR(5) data set
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
    int p = 5;
    
    // Create the parameter vector, omega
	arma::vec omega(p);
    for (int i=0; i<p/2; i++) {
        omega(2*i) = qpo_cent[i];
        omega(1+2*i) = qpo_width[i];
    }
    // p is odd, so add in additional value of lorentz_width
    omega(p-1) = qpo_width[p/2];
    
    // construct the moving average coefficients, right now just assume q = 1
    arma::vec ma_coefs = arma::zeros<arma::vec>(p);
    ma_coefs(0) = 1.0;
    
    KalmanFilterp Kfilter(time, y, yerr, 1.0, omega, ma_coefs);
    
    // Get the roots of the AR(p) polynomial and compute the value of sigsqr given omega and sigmay
    arma::cx_vec ar_roots = Kfilter.ARRoots(omega);
    CARp car5_process(true, "CAR(5)", time, y, yerr, p);
    double sigsqr = sigmay * sigmay / car5_process.Variance(ar_roots, ma_coefs, 1.0);
    Kfilter.SetSigsqr(sigsqr);
    
    // First test that the Kalman Filter is correctly initialized after reseting it
    Kfilter.Reset();
    arma::vec kmean = Kfilter.mean;
    REQUIRE(kmean(0) == 0.0);
    arma::vec kvar = Kfilter.var;
    double kvar_expected = sigmay * sigmay + yerr(0) * yerr(0);
    REQUIRE(std::abs(kvar(0) - kvar_expected) < 1e-10);
    
    // Now test the one-step prediction
    Kfilter.Update();
    kmean = Kfilter.mean;
    kvar = Kfilter.var;
    double sresid1 = (y(1) - kmean(1)) / sqrt(kvar(1));
    REQUIRE(std::abs(sresid1) < 3.0);
    
    // Compute and grab the kalman filter
    Kfilter.Filter();
    kmean = Kfilter.mean;
    kvar = Kfilter.var;
    
    // Compute the standardized residuals of the time series
    arma::vec sresid = (y - kmean) / arma::sqrt(kvar);
    
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

TEST_CASE("./KalmanFilterp/Predict", "Test interpolation/extrapolation for a CAR(5) process") {
    // first grab the simulated Gaussian CAR(5) data set
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
    int p = 5;
    
    // Create the parameter vector, omega
	arma::vec omega(p);
    for (int i=0; i<p/2; i++) {
        omega(2*i) = qpo_cent[i];
        omega(1+2*i) = qpo_width[i];
    }
    // p is odd, so add in additional value of lorentz_width
    omega(p-1) = qpo_width[p/2];
    
    // construct the moving average coefficients, right now just assume q = 1
    arma::vec ma_coefs = arma::zeros<arma::vec>(p);
    ma_coefs(0) = 1.0;
    
    KalmanFilterp Kfilter(time, y, yerr, 1.0, omega, ma_coefs);
    Kfilter.Filter();
    
    // Get the roots of the AR(p) polynomial and compute the value of sigsqr given omega and sigmay
    arma::cx_vec ar_roots = Kfilter.ARRoots(omega);
    CARp car5_process(true, "CAR(5)", time, y, yerr, p);
    double sigsqr = sigmay * sigmay / car5_process.Variance(ar_roots, ma_coefs, 1.0);
    Kfilter.SetSigsqr(sigsqr);
    
    // first test forecasting
    double tpredict = time(ny-1) + 0.05 * (time(ny-1) - time(0));
    std::pair<double, double> kpredict = Kfilter.Predict(tpredict);
    double pmean = kpredict.first;
    double pvar = kpredict.second;
    
    // construct covariance matrix of (tpredict,time)
    arma::mat covar(ny+1,ny+1);
    for (int i=0; i<ny+1; i++) {
        for (int j=0; j<ny+1; j++) {
            double timei, timej;
            if (i == 0) {
                timei = tpredict;
            } else {
                timei = time(i-1);
            }
            if (j == 0) {
                timej = tpredict;
            } else {
                timej = time(j-1);
            }
            double dt = std::abs(timei - timej);
            covar(i,j) = car5_process.Variance(ar_roots, ma_coefs, sqrt(sigsqr), dt);
            if (i == j && i > 0) {
                // add contribution from measurement errors
                covar(i,j) += yerr(i-1) * yerr(i-1);
            }
        }
    }
    
    covar = arma::symmatl(covar);
    
    // calculate prediction mean and variance the slow way
    double pmean_slow, pvar_slow;
    arma::mat subvar_inv = arma::inv(arma::sympd(covar.submat(1, 1, ny, ny)));
    arma::rowvec subcov = covar.submat(0,1,0,ny);
    pmean_slow = arma::as_scalar(subcov * subvar_inv * y);
    pvar_slow = covar(0,0) - arma::as_scalar(subcov * subvar_inv * subcov.t());
    
    // make sure predicted mean and variance computed from the kalman filter is equal to that
    // computed the slow way from the properties of the normal distribution
    double frac_diff = std::abs(pmean_slow - pmean) / std::abs(pmean_slow);
    REQUIRE(frac_diff < 1e-6);
    frac_diff = std::abs(pvar_slow - pvar) / std::abs(pvar_slow);
    REQUIRE(frac_diff < 1e-6);
    
    // now test backcasting
    tpredict = time(0) - 0.01 * (time(ny-1) - time(0));
    kpredict = Kfilter.Predict(tpredict);
    pmean = kpredict.first;
    pvar = kpredict.second;
    
    for (int i=1; i<ny+1; i++) {
        double dt = std::abs(tpredict - time(i-1));
        covar(0,i) = car5_process.Variance(ar_roots, ma_coefs, sqrt(sigsqr), dt);
        covar(i,0) = covar(0,i);
    }
    covar = arma::symmatu(covar);
    subcov = covar.submat(0,1,0,ny);
    pmean_slow = arma::as_scalar(subcov * subvar_inv * y);
    pvar_slow = covar(0,0) - arma::as_scalar(subcov * subvar_inv * subcov.t());
    
    // make sure predicted mean and variance computed from the kalman filter is equal to that
    // computed the slow way from the properties of the normal distribution
    frac_diff = std::abs(pmean_slow - pmean) / std::abs(pmean_slow);
    REQUIRE(frac_diff < 1e-6);
    frac_diff = std::abs(pvar_slow - pvar) / std::abs(pvar_slow);
    REQUIRE(frac_diff < 1e-6);
    
    // finally, test interpolation
    tpredict = 166.0;
    Kfilter.Reset();
    kpredict = Kfilter.Predict(tpredict);
    pmean = kpredict.first;
    pvar = kpredict.second;
    
    for (int i=1; i<ny+1; i++) {
        double dt = std::abs(tpredict - time(i-1));
        covar(0,i) = car5_process.Variance(ar_roots, ma_coefs, sqrt(sigsqr), dt);
        covar(i,0) = covar(0,i);
    }
    covar = arma::symmatu(covar);
    subcov = covar.submat(0,1,0,ny);
    pmean_slow = arma::as_scalar(subcov * subvar_inv * y);
    pvar_slow = covar(0,0) - arma::as_scalar(subcov * subvar_inv * subcov.t());
    
    // make sure predicted mean and variance computed from the kalman filter is equal to that
    // computed the slow way from the properties of the normal distribution
    frac_diff = std::abs(pmean_slow - pmean) / std::abs(pmean_slow);
    REQUIRE(frac_diff < 1e-6);
    frac_diff = std::abs(pvar_slow - pvar) / std::abs(pvar_slow);
    REQUIRE(frac_diff < 1e-6);
}

TEST_CASE("./KalmanFilter/Simulate", "Test Simulated time series for a CAR(5) process.") {
    // first grab the simulated Gaussian CAR(5) data set
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
    int p = 5;
    
    // Create the parameter vector, omega
	arma::vec omega(p);
    for (int i=0; i<p/2; i++) {
        omega(2*i) = qpo_cent[i];
        omega(1+2*i) = qpo_width[i];
    }
    // p is odd, so add in additional value of lorentz_width
    omega(p-1) = qpo_width[p/2];
    
    // construct the moving average coefficients, right now just assume q = 1
    arma::vec ma_coefs = arma::zeros<arma::vec>(p);
    ma_coefs(0) = 1.0;
    
    KalmanFilterp Kfilter(time, y, yerr, 1.0, omega, ma_coefs);
    Kfilter.Filter();
    
    // Get the roots of the AR(p) polynomial and compute the value of sigsqr given omega and sigmay
    arma::cx_vec ar_roots = Kfilter.ARRoots(omega);
    CARp car5_process(true, "CAR(5)", time, y, yerr, p);
    double sigsqr = sigmay * sigmay / car5_process.Variance(ar_roots, ma_coefs, 1.0);
    Kfilter.SetSigsqr(sigsqr);
    
    // simulate the time series using the Kalman Filter
    double tsim_max = time(ny-1) + 0.05 * (time(ny-1) - time(0));
    double tsim_min = time(0) - 0.05 * (time(ny-1) - time(0));
    int nsim = 200;
    arma::vec tsim = arma::linspace<arma::vec>(tsim_min, tsim_max, nsim);
        
    arma::vec ysim = Kfilter.Simulate(tsim);
    
    // construct covariance matrix of (tpredict,time)
    arma::mat covar(ny+nsim,ny+nsim);
    arma::vec tcombined = time;
    tcombined.insert_rows(0, tsim);
    
    for (int i=0; i<ny+nsim; i++) {
        for (int j=0; j<ny+nsim; j++) {
            double dt = std::abs(tcombined(i) - tcombined(j));
            covar(i,j) = car5_process.Variance(ar_roots, ma_coefs, sqrt(sigsqr), dt);
            if ((i == j) && (i >= nsim)) {
                // add contribution from measurement errors to the diagonal
                covar(i,j) += yerr(i-nsim) * yerr(i-nsim);
            }
        }
    }
    covar = arma::symmatl(covar);
    
    // compute the conditional mean and variance of the simulated time series
    // directly from the properties of the multivariate gaussian distribution
    arma::vec cmean(nsim);
    arma::mat cvar(nsim,nsim);
    arma::mat subvar_inv = arma::inv(arma::sympd(covar.submat(nsim, nsim, ny+nsim-1, ny+nsim-1)));
    arma::mat subcov = covar.submat(0,nsim,nsim-1,ny+nsim-1);
    arma::mat simvar = covar.submat(0,0,nsim-1,nsim-1);
    cmean = subcov * subvar_inv * y;
    cvar = simvar - subcov * subvar_inv * subcov.t();
    
    // standardize the simulated time series
    arma::mat cvar_chol = arma::chol(cvar);
    arma::vec sresid = cvar_chol.t().i() * (ysim - cmean);
    
    // Test that the standardized residuals are consistent with having a standard normal distribution using
    // the Anderson-Darling test statistic
    arma::vec sorted_sresid = arma::sort(sresid);
    boost::math::normal snorm;
    arma::vec snorm_cdf(nsim);
    for (int i=0; i<nsim; i++) {
        // compute the standard normal CDF of the standardized residuals
        snorm_cdf(i) = boost::math::cdf(snorm, sorted_sresid(i));
    }
    
    double AD_sum = 0.0;
    for (int i=0; i<nsim; i++) {
        // compute the Anderson-Darling statistic
        AD_sum += (2.0 * (i+1) - 1) / nsim * (log(snorm_cdf(i)) + log(1.0 - snorm_cdf(nsim-1-i)));
    }
    double AD_stat = -nsim - AD_sum;
    REQUIRE(AD_stat < 3.857); // critical value for 1% significance level
    
    // Now test that the autocorrelation function of the standardized residuals is consistent with a white noise process
    int maxlag = 100;
    arma::vec acorr_sresid = autocorr(sresid, maxlag);
    double acorr_95bound = 1.96 / sqrt(nsim); // find number of autocorr values outside of 95% confidence interval
    int out_of_bounds = arma::accu(arma::abs(acorr_sresid) > acorr_95bound);
    REQUIRE(out_of_bounds < 11); // 99% significance level for binomial distribution with n = 100 and p = 0.05
    
    double max_asqr = arma::max(acorr_sresid % acorr_sresid);
    boost::math::chi_squared chisqr(1); // square of ACF has a chi-squared distribution with two DOF
    double chisqr_cdf = boost::math::cdf(chisqr, max_asqr * nsim);
    double max_asqr_cdf = std::pow(chisqr_cdf, maxlag); // CDF of maximum of maxlag random variables having a chi-square distribution
    REQUIRE(max_asqr_cdf < 0.99); // test fails if probability of max(ACF) < 1%    
}

TEST_CASE("CAR1/logpost_test", "Make sure the that CAR1.logpost_ == Car1.GetLogPost(theta) after running MCMC sampler") {
    int ny = 100;
    arma::vec time = arma::linspace<arma::vec>(0.0, 100.0, ny);
    arma::vec y = arma::randn<arma::vec>(ny);
    arma::vec ycent = y - arma::mean(y);
    arma::vec ysig = 0.01 * arma::ones(ny);
    
    CAR1 car1_test(true, "CAR(1)", time, y, ysig);
    KalmanFilter1 Kfilter(time, ycent, ysig);
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
        arma::vec theta = car1_test.Value();
        double logdens_computed = car1_test.LogDensity(theta); // explicitly calculate log-posterior for current theta
        // calculate log-posterior manually from the kalman filter object
        Kfilter.SetSigsqr(car1_test.ExtractSigsqr(theta));
        Kfilter.SetOmega(car1_test.ExtractAR(theta));
        Kfilter.SetMA(car1_test.ExtractMA(theta));
        arma::vec scaled_yerr = sqrt(theta(1)) * ysig;
        Kfilter.SetTimeSeriesErr(scaled_yerr);
        Kfilter.Filter();
        double logdens_kfilter = 0.0;
        for (int j=0; j<ny; j++) {
            logdens_kfilter += -0.5 * log(Kfilter.var(j)) -
            0.5 * (ycent(j) - Kfilter.mean(j)) * (ycent(j) - Kfilter.mean(j)) / Kfilter.var(j);
        }
        logdens_kfilter += car1_test.LogPrior(theta);
        double computed_diff = std::abs(logdens_computed - logdens_stored);
        bool no_match_computed = computed_diff > 1e-10;
        double kfilter_diff = std::abs(logdens_kfilter - logdens_stored);
        bool no_match_kfilter = kfilter_diff > 1e-10;
        if (no_match_computed || no_match_kfilter) {
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
    arma::vec ycent = y - arma::mean(y);
    arma::vec ysig = 0.01 * arma::ones(ny);
    int p = 5;
    
    ZCARMA car5_test(true, "ZCARMA(5)", time, y, ysig, p);
    KalmanFilterp Kfilter(time, ycent, ysig);
    Kfilter.SetMA(car5_test.ExtractMA(car5_test.Value()));
    
    double max_stdev = 10.0 * arma::stddev(y); // For prior: maximum standard-deviation of CAR(1) process
    car5_test.SetPrior(max_stdev);
    
    // setup Robust Adaptive Metropolis step object
    StudentProposal tUnit(8.0, 1.0);
    arma::mat prop_covar(p+3,p+3);
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
        arma::vec theta = car5_test.Value();
        double logdens_computed = car5_test.LogDensity(theta); // explicitly calculate log-posterior for current theta
        // calculate log-posterior manually from the kalman filter object
        arma::vec omega = car5_test.ExtractAR(theta);
        Kfilter.SetOmega(omega);
        arma::vec ma_coefs = car5_test.ExtractMA(theta);
        Kfilter.SetMA(ma_coefs);
        double sigsqr;
        arma::cx_vec ar_roots = car5_test.ARRoots(theta);
        sigsqr = theta(0) * theta(0) / car5_test.Variance(ar_roots, ma_coefs, 1.0);
        Kfilter.SetSigsqr(sigsqr);
        arma::vec scaled_yerr = sqrt(theta(1)) * ysig;
        Kfilter.SetTimeSeriesErr(scaled_yerr);
        Kfilter.Filter();
        double logdens_kfilter = 0.0;
        for (int j=0; j<ny; j++) {
            logdens_kfilter += -0.5 * log(Kfilter.var(j)) -
            0.5 * (ycent(j) - Kfilter.mean(j)) * (ycent(j) - Kfilter.mean(j)) / Kfilter.var(j);
        }
        logdens_kfilter += car5_test.LogPrior(theta);
        bool no_match_computed = std::abs(logdens_computed - logdens_stored) > 1e-10;
        bool no_match_kfilter = std::abs(logdens_kfilter - logdens_stored) > 1e-10;
        if (no_match_computed || no_match_kfilter) {
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

TEST_CASE("CARMA/prior_bounds", "Make sure CARp::LogDensity return -infinity when prior bounds are violated") {
    int ny = 100;
    arma::vec time = arma::linspace<arma::vec>(0.0, 100.0, ny);
    arma::vec y = arma::randn<arma::vec>(ny);
    arma::vec ysig = 0.01 * arma::ones(ny);
    int p = 9;
    int q = 4;
    
    CARMA carma_test(true, "CARMA(9,4)", time, y, ysig, p, q);
    double max_stdev = 10.0 * arma::stddev(y); // For prior: maximum standard-deviation of CAR(1) process
    carma_test.SetPrior(max_stdev);
    
    // prior bounds on lorentzian parameters
    double max_freq = 10.0;
	double min_freq = 1.0 / (10.0 * time.max());
    
    arma::vec bad_theta(p+2); // parameter value will violated the prior bounds
    bad_theta = carma_test.StartingValue();

    bad_theta = carma_test.StartingValue();
    bad_theta(0) = 2.0 * max_stdev;
    double logpost = carma_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    bad_theta(0) = -1.0;
    logpost = carma_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    
    bad_theta(0) = max_stdev / 10.0;
    bad_theta(1) = 0.1;
    logpost = carma_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    bad_theta(1) = 4.0;
    logpost = carma_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    
    bad_theta(1) = 1.0;
    int nbad_width = 0;
    for (int j=0; j<p/2; j++) {
        double qpo_width = bad_theta(3+2*j);
        bad_theta(3+2*j) = log(min_freq / 2.0);
        logpost = carma_test.LogDensity(bad_theta);
        if (logpost != -1.0 * arma::datum::inf) {
            nbad_width++;
        }
        bad_theta(3+2*j) = log(2.0 * max_freq);
        logpost = carma_test.LogDensity(bad_theta);
        if (logpost != -1.0 * arma::datum::inf) {
            nbad_width++;
        }
        bad_theta(3+2*j) = qpo_width;
    }
    REQUIRE(nbad_width == 0);
    
    double qpo_cent = bad_theta(2);
    bad_theta(2) = log(2.0 * max_freq);
    logpost = carma_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    bad_theta(2) = qpo_cent;
    qpo_cent = bad_theta(2+2*(p/2-1));
    bad_theta(2+2*(p/2-1)) = log(min_freq / 2.0);
    logpost = carma_test.LogDensity(bad_theta);
    REQUIRE(logpost == -1.0 * arma::datum::inf);
    bad_theta(2+2*(p/2-1)) = qpo_cent;
    int nbad_cent = 0;
    for (int j=1; j<p/2; j++) {
        // violate the ordering of the lorentzian centroids
        qpo_cent = bad_theta(2+2*j);
        bad_theta(2+2*j) = log(1.1) + bad_theta(2+2*(j-1));
        logpost = carma_test.LogDensity(bad_theta);
        if (logpost != -1.0 * arma::datum::inf) {
            nbad_cent++;
        }
        bad_theta(2+2*j) = qpo_cent;
    }
    REQUIRE(nbad_cent == 0);
    
    double ma_imag = bad_theta(p+2);
    bad_theta(p+2+2*(q/2-1)) = ma_imag;
    int nbad_imag = 0;
    for (int j=1; j<q/2; j++) {
        // violate the ordering of the lorentzian centroids
        ma_imag = bad_theta(2+p+2*j);
        bad_theta(p+2+2*j) = log(1.1) + bad_theta(p+2+2*(j-1));
        logpost = carma_test.LogDensity(bad_theta);
        if (logpost != -1.0 * arma::datum::inf) {
            nbad_imag++;
        }
        bad_theta(p+2+2*j) = ma_imag;
    }
    REQUIRE(nbad_imag == 0);
}

TEST_CASE("ZCARMA/variance", "Test the CARp::Variance method") {
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
    
    ZCARMA carma_process(true, "CAR(5)", time, y, ysig, p);
    double kappa = 0.7;
    // Set the moving average terms
    arma::vec ma_coefs(p);
	ma_coefs(0) = 1.0;
	for (int i=1; i<p; i++) {
		ma_coefs(i) = boost::math::binomial_coefficient<double>(p-1, i) / pow(kappa,i);
	}
    
    double model_var = carma_process.Variance(alpha_roots, ma_coefs, sigma);
    double model_var0 = 223003.230567; // known variance, computed from python module carma_pack
    double frac_diff = std::abs(model_var - model_var0) / std::abs(model_var0);
    REQUIRE(frac_diff < 1e-8);
}

TEST_CASE("./CAR1/mcmc_sampler", "Test RunEnsembleCarSampler on CAR(1) model") {
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
    mcmc_out = RunEnsembleCarmaSampler(sample_size, burnin, time, y, yerr, carp_order, 0, nwalkers);
    std::vector<arma::vec> mcmc_sample;
    mcmc_sample = mcmc_out.first;
    std::vector<double> logpost_samples = mcmc_out.second;
    
    // True CAR(1) process parameters
    double tau = 100.0;
    double omega = 1.0 / tau;
    double sigmay = 2.3;
    double sigma = sigmay * sqrt(2.0 / tau);
    double measerr_scale = 1.0;
    
    std::ofstream mcmc_outfile("data/car1_mcmc.dat");
    mcmc_outfile << "# sigma, measerr_scale, omega, logpost" << std::endl;
    
    arma::vec omega_samples(mcmc_sample.size());
    arma::vec sigma_samples(mcmc_sample.size());
    arma::vec scale_samples(mcmc_sample.size());
    for (int i=0; i<mcmc_sample.size(); i++) {
        sigma_samples(i) = log(mcmc_sample[i](0));
        scale_samples(i) = mcmc_sample[i](1);
        omega_samples(i) = mcmc_sample[i](2);
        mcmc_outfile << mcmc_sample[i](0) << " " << mcmc_sample[i](1) << " "
        << exp(mcmc_sample[i](2)) << " " << logpost_samples[i] << std::endl;
    }
    mcmc_outfile.close();
    
    // Make sure true parameters are within 3sigma of the marginal posterior means
    double sigma_zscore = (arma::mean(sigma_samples) - log(sigmay)) / arma::stddev(sigma_samples);
    CHECK(std::abs(sigma_zscore) < 3.0);
    double scale_zscore = (arma::mean(scale_samples) - measerr_scale) / arma::stddev(scale_samples);
    CHECK(std::abs(scale_zscore) < 3.0);
    double omega_zscore = (arma::mean(omega_samples) - log(omega)) / arma::stddev(omega_samples);
    CHECK(std::abs(omega_zscore) < 3.0);
}

TEST_CASE("./CAR5/mcmc_sampler", "Test RunEnsembleCarSampler on CAR(5) model") {
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
    mcmc_out = RunEnsembleCarmaSampler(sample_size, burnin, time, y, yerr, carp_order, 0, nwalkers);
    std::vector<arma::vec> mcmc_sample;
    mcmc_sample = mcmc_out.first;
    std::vector<double> logpost_samples = mcmc_out.second;
    
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

    std::ofstream mcmc_outfile("data/car5_mcmc.dat");
    mcmc_outfile << "# sigma, measerr_scale, (lorentz_cent,lorentz_width), logpost" << std::endl;
    
    arma::vec sigma_samples(mcmc_sample.size());
    arma::vec scale_samples(mcmc_sample.size());
    arma::mat ar_samples(mcmc_sample.size(),p);
    for (int i=0; i<mcmc_sample.size(); i++) {
        sigma_samples(i) = log(mcmc_sample[i](0));
        scale_samples(i) = mcmc_sample[i](1);
        for (int j=0; j<p; j++) {
            ar_samples(i,j) = mcmc_sample[i](j+2);
        }
        for (int j=0; j<theta.n_elem; j++) {
            mcmc_outfile << mcmc_sample[i](j) << " ";
        }
        mcmc_outfile << logpost_samples[i] << std::endl;
    }
    mcmc_outfile.close();
    
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

TEST_CASE("ZCARMA5/mcmc_sampler", "Test RunEnsembleCarSampler on ZCARMA(5) model") {
    std::cout << std::endl;
    std::cout << "Running test of MCMC sampler for ZCARMA(5) model..." << std::endl << std::endl;
    
    // first grab the simulated Gaussian CAR(5) data set
    arma::mat zcarma_data;
    zcarma_data.load(zcarmafile, arma::raw_ascii);
    
    arma::vec time = zcarma_data.col(0);
    arma::vec y = zcarma_data.col(1);
    arma::vec yerr = zcarma_data.col(2);
    
    // MCMC parameters
    int carp_order = 5;
    int nwalkers = 10;
    int sample_size = 100000;
    int burnin = 50000;
    
    // run the MCMC sampler
    std::pair<std::vector<arma::vec>, std::vector<double> > mcmc_out;
    mcmc_out = RunEnsembleCarmaSampler(sample_size, burnin, time, y, yerr, carp_order, carp_order-1,
                                       nwalkers, true, 1);
    std::vector<arma::vec> mcmc_sample;
    mcmc_sample = mcmc_out.first;
    std::vector<double> logpost_samples = mcmc_out.second;
    
    // True ZCARMA(5) process parameters
    double qpo_width[3] = {0.01, 0.01, 0.002};
    double qpo_cent[2] = {0.2, 0.02};
    double sigmay = 2.3;
    double measerr_scale = 1.0;
    double kappa = 0.7;
    int p = 5;

    // Create the parameter vector, theta
	arma::vec theta(p+3);
    theta(0) = log(sigmay);
	theta(1) = measerr_scale;
    for (int i=0; i<p/2; i++) {
        theta(2+2*i) = log(qpo_cent[i]);
        theta(3+2*i) = log(qpo_width[i]);
    }
    // p is odd, so add in additional value of lorentz_width
    theta(p+1) = log(qpo_width[p/2]);
    theta(p+2) = logit(kappa);
    
    std::ofstream mcmc_outfile("data/zcarma5_mcmc.dat");
    mcmc_outfile <<
    "# log sigma, measerr_scale, (lorentz_cent,lorentz_width), logit(kappa), logpost" << std::endl;
    
    arma::vec sigma_samples(mcmc_sample.size());
    arma::vec scale_samples(mcmc_sample.size());
    arma::mat ar_samples(mcmc_sample.size(),p);
    arma::vec kappa_samples(mcmc_sample.size()); // actually, logit(kappa)
    for (int i=0; i<mcmc_sample.size(); i++) {
        sigma_samples(i) = log(mcmc_sample[i](0));
        scale_samples(i) = mcmc_sample[i](1);
        for (int j=0; j<p; j++) {
            ar_samples(i,j) = mcmc_sample[i](j+2);
        }
        kappa_samples(i) = mcmc_sample[i](p+2);
        for (int j=0; j<theta.n_elem; j++) {
            mcmc_outfile << mcmc_sample[i](j) << " ";
        }
        mcmc_outfile << logpost_samples[i] << std::endl;
    }
    mcmc_outfile.close();
    
    // Make sure true parameters are within 3sigma of the marginal posterior means
    double sigma_zscore = (arma::mean(sigma_samples) - log(sigmay)) / arma::stddev(sigma_samples);
    CHECK(std::abs(sigma_zscore) < 3.0);
    double scale_zscore = (arma::mean(scale_samples) - measerr_scale) / arma::stddev(scale_samples);
    CHECK(std::abs(scale_zscore) < 3.0);
    for (int j=0; j<p; j++) {
        double ar_zscore = (arma::mean(ar_samples.col(j)) - theta(2+j)) / arma::stddev(ar_samples.col(j));
        CHECK(std::abs(ar_zscore) < 3.0);
    }
    double kappa_zscore = (arma::mean(kappa_samples) - logit(kappa)) / arma::stddev(kappa_samples);
    CHECK(std::abs(kappa_zscore) < 3.0);
}

TEST_CASE("CARMA/mcmc_sampler", "Test RunEnsembleCarSampler on CARMA(5,4) model") {
    std::cout << std::endl;
    std::cout << "Running test of MCMC sampler for CARMA(5,4) model..." << std::endl << std::endl;
    
    // first grab the simulated Gaussian ZCARMA(5) data set
    arma::mat carma_data;
    carma_data.load(zcarmafile, arma::raw_ascii);
    
    arma::vec time = carma_data.col(0);
    arma::vec y = carma_data.col(1);
    arma::vec yerr = carma_data.col(2);
    
    // MCMC parameters
    int carp_order = 5;
    int nwalkers = 10;
    int sample_size = 100000;
    int burnin = 50000;
    
    // run the MCMC sampler
    std::pair<std::vector<arma::vec>, std::vector<double> > mcmc_out;
    mcmc_out = RunEnsembleCarmaSampler(sample_size, burnin, time, y, yerr, carp_order, carp_order-1, nwalkers);
    std::vector<arma::vec> mcmc_sample;
    mcmc_sample = mcmc_out.first;
    std::vector<double> logpost_samples = mcmc_out.second;
    
    // True CAR(5) process parameters
    double qpo_width[3] = {0.01, 0.01, 0.002};
    double qpo_cent[2] = {0.2, 0.02};
    double sigmay = 2.3;
    double measerr_scale = 1.0;
    int p = 5;
    double kappa = 0.7;
    arma::vec ma_coefs(p);
	ma_coefs(0) = 1.0;
	for (int i=1; i<p; i++) {
		ma_coefs(i) = boost::math::binomial_coefficient<double>(p-1, i) / pow(kappa,i);
	}
    
    // Create the parameter vector, theta
    int q = p - 1;
	arma::vec theta(q+p+2);
    theta(0) = log(sigmay);
	theta(1) = measerr_scale;
    for (int i=0; i<p/2; i++) {
        theta(2+2*i) = log(qpo_cent[i]);
        theta(3+2*i) = log(qpo_width[i]);
    }
    // p is odd, so add in additional value of lorentz_width
    theta(p+1) = log(qpo_width[p/2]);
    for (int j=p+2; j<theta.n_elem; j++) {
        theta(j) = log(kappa); // MA polynomial only has a single real root at -kappa.
    }
    
    std::ofstream mcmc_outfile(carmafile);
    mcmc_outfile <<
    "# sigma, measerr_scale, (lorentz_cent,lorentz_width), (imag(MA root), real(MA_root)), logpost"
    << std::endl;
    
    arma::vec sigma_samples(mcmc_sample.size());
    arma::vec scale_samples(mcmc_sample.size());
    arma::mat ar_samples(mcmc_sample.size(),p);
    arma::mat ma_samples(mcmc_sample.size(),p-1);
    for (int i=0; i<mcmc_sample.size(); i++) {
        sigma_samples(i) = log(mcmc_sample[i](0));
        scale_samples(i) = mcmc_sample[i](1);
        for (int j=0; j<p; j++) {
            ar_samples(i,j) = mcmc_sample[i](j+2);
        }
        for (int j=0; j<p-1; j++) {
            ma_samples(i,j) = mcmc_sample[i](p+2+j);
        }
        for (int j=0; j<theta.n_elem; j++) {
            mcmc_outfile << mcmc_sample[i](j) << " ";
        }
        mcmc_outfile << logpost_samples[i] << std::endl;
    }
    mcmc_outfile.close();
    
    // Make sure true parameters are within 3sigma of the marginal posterior means
    double sigma_zscore = (arma::mean(sigma_samples) - log(sigmay)) / arma::stddev(sigma_samples);
    CHECK(std::abs(sigma_zscore) < 3.0);
    double scale_zscore = (arma::mean(scale_samples) - measerr_scale) / arma::stddev(scale_samples);
    CHECK(std::abs(scale_zscore) < 3.0);
    for (int j=0; j<p; j++) {
        double ar_zscore = (arma::mean(ar_samples.col(j)) - theta(2+j)) / arma::stddev(ar_samples.col(j));
        CHECK(std::abs(ar_zscore) < 3.0);
    }
    for (int j=0; j<p; j++) {
        double ma_zscore = (arma::mean(ma_samples.col(j)) - theta(p+2+j)) / arma::stddev(ma_samples.col(j));
        CHECK(std::abs(ma_zscore) < 3.0);
    }
}
