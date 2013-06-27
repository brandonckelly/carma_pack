//
//  kfilter.cpp
//  carma_pack
//
//  Created by Brandon Kelly on 6/27/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#include "include/kfilter.hpp"

// Reset the Kalman Filter for a CAR(1) process
void KalmanFilter1::Reset() {
    
    mean_(0) = 0.0;
    var_(0) = sigsqr_ / (2.0 * omega_) + yerr_(0) * yerr_(0);
    current_index_ = 1;
}

// Perform one iteration of the Kalman Filter for a CAR(1) process to update it
void KalmanFilter1::Update() {
    
    double rho, var_ratio, previous_var;
    rho = exp(-1.0 * omega_ * dt_(current_index_-1));
    previous_var = var_(current_index_-1) - yerr_(current_index_-1) * yerr_(current_index_-1);
    var_ratio = previous_var / var_(current_index_-1);
		
    // Update the Kalman filter mean
    mean_(current_index_) = rho * mean_(current_index_-1) +
        rho * var_ratio * (y_(current_index_-1) - mean_(current_index_-1));
		
    // Update the Kalman filter variance
    var_(current_index_) = sigsqr_ / (2.0 * omega_) * (1.0 - rho * rho) +
        rho * rho * previous_var * (1.0 - var_ratio);
    
    // add in contribution to variance from measurement errors
    var_(current_index_) += yerr_(current_index_) * yerr_(current_index_);
}

// Return the predicted value and its variance at time assuming a CAR(1) process
std::pair<double, double> KalmanFilter1::Predict(double time) {
    double rho, var_ratio, previous_var;
}