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
#include <armadillo>

// Files containing simulated CAR(1) and CAR(5) time series, used for testing
std::ifstream car1file("car1_test.dat");
std::ifstream car5file("car5_test.dat");

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