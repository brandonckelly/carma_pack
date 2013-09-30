#include <algorithm>
#include <numeric>
#include <vector>
#include "ceres/ceres.h"
#include <armadillo>
#include <carpack.hpp>
#include <ceres.hpp>
#include <ceres/dynamic_autodiff_cost_function.h>
#include "ceres/internal/scoped_ptr.h"

using ceres::CostFunction;
using ceres::NumericDiffCostFunction;
using ceres::NumericDiffMethod;
using ceres::DynamicAutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

std::vector<double>
RunCeres1(std::vector<double> time, std::vector<double> y, std::vector<double> yerr, 
          bool use_cauchy, const std::vector<double>& init) {

    double sum = std::accumulate(y.begin(), y.end(), 0.0);
    double mean = sum / y.size();
    double sq_sum = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
    double var = (sq_sum / y.size() - mean * mean);
	double max_stdev = 10.0 * std::sqrt(var); // For prior: maximum standard-deviation 
    
    std::shared_ptr<CAR1> car;
    car = std::shared_ptr<CAR1>(new CAR1(true, "CAR(1)", time, y, yerr));    
    car->SetPrior(max_stdev);
    arma::vec theta = car->StartingValue();
    if (init.size() == theta.n_elem) {
        // Check that input paramters are the right size; if so, override StartingValues
        theta = arma::conv_to<arma::vec>::from(init);
    }
    
    Problem problem;
    std::vector<double*> params;
    for (unsigned int i = 0; i < theta.n_elem; i++) {
        double* param = new double(theta[i]);
        params.push_back(param);
    }
    int npars = params.size();

    problem.AddResidualBlock(CreateCarpRuntimeNumericDiffCostFunction(new CarCostFunction<CAR1>(car, npars),
                                                                      ceres::CENTRAL, 1e-6),
                             NULL, params);
    
    
    ceres::Solver::Options options;
    //options.linear_solver_type = ceres::DENSE_SCHUR;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_type     = ceres::LINE_SEARCH;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::vector<double> solution(npars);
    for (int i = 0; i < npars; i++) {
        solution[i] = *params[i];
    }
          
    return solution;
}

std::vector<double>
RunCeres(std::vector<double> time, std::vector<double> y, std::vector<double> yerr, 
         int p, int q, bool do_zcarma, bool use_cauchy, const std::vector<double>& init) {

    double sum = std::accumulate(y.begin(), y.end(), 0.0);
    double mean = sum / y.size();
    double sq_sum = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
    double var = (sq_sum / y.size() - mean * mean);
	double max_stdev = 10.0 * std::sqrt(var); // For prior: maximum standard-deviation 
    
    std::shared_ptr<CARp> car;
    if (!do_zcarma) {
        if (q == 0) {
            car = std::shared_ptr<CARp>(new CARp(false, "CAR(p) Parameters", time, y, yerr, p));
        }
        else {
            car = std::shared_ptr<CARp>(new CARMA(false, "CARMA(p,q) Parameters", time, y, yerr, p, q));
        }
    } else {
        car = std::shared_ptr<CARp>(new ZCAR(false, "ZCAR(p) Parameters", time, y, yerr, p));
    }
    car->SetPrior(max_stdev);
    arma::vec theta = car->StartingValue();
    std::cout << "CAW " << init.size() << " " << theta.n_elem << std::endl;

    if (init.size() == theta.n_elem) {
        // Check that input paramters are the right size; if so, override StartingValues
        theta = arma::conv_to<arma::vec>::from(init);
    }
    
    Problem problem;
    std::vector<double*> params;
    for (unsigned int i = 0; i < theta.n_elem; i++) {
        double* param = new double(theta[i]);
        params.push_back(param);
    }
    int npars = params.size();

    problem.AddResidualBlock(CreateCarpRuntimeNumericDiffCostFunction(new CarCostFunction<CARp>(car, npars),
                                                                      ceres::CENTRAL, 1e-6),
                             NULL, params);
    
    
    ceres::Solver::Options options;
    //options.linear_solver_type = ceres::DENSE_SCHUR;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_type     = ceres::LINE_SEARCH;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::vector<double> cparams(npars);
    std::vector<double> solution(npars);
    for (int i = 0; i < npars; i++) {
        cparams[i]  = *params[i];
        solution[i] = *params[i];
    }

    /*
    ceres::internal::scoped_ptr<CostFunction> cfs[1];
    cfs[0].reset(CreateCarpRuntimeNumericDiffCostFunction(new CarpCostFunction(car, npars),
                                                          ceres::CENTRAL, 1e-6));
    CostFunction *cost_function = cfs[0].get();

    double *parameters[] = { params[0], };
    double residuals[1];
    std::vector<double> jacobian0(npars*npars);
    double *jacobians[1] = { &jacobian0[0], };
    cost_function->Evaluate(&parameters[0], &residuals[0], &jacobians[0]);
    for (int i = 0; i < npars; i++) {
        for (int j = 0; j < npars; j++) {
            std::cout << "Cov:" << i << " " << j << " " << jacobians[j + npars*i];
        }
        std::cout << std::endl;
    }
    */

    /*
    double *covariance_xx[] = { covariance_x[0] };
    ceres::Covariance::Options coptions;
    ceres::Covariance covariance(coptions);
    std::vector<std::pair<const double*, const double*> > covariance_blocks;
    covariance_blocks.push_back(std::make_pair(parameters, parameters));
    CHECK(covariance.Compute(covariance_blocks, &problem));
    std::vector<double*> covariance_x(npars*npars);
    double *covariance_xx[] = { covariance_x[0] };
    covariance.GetCovarianceBlock(parameters[0], parameters[0], covariance_xx[0]);
    for (int i = 0; i < npars; i++) {
        for (int j = 0; j < npars; j++) {
            std::cout << "Cov:" << i << " " << j << " " << covariance_xx[j + npars*i];
        }
        std::cout << std::endl;
    }
    */

          
    return solution;
}


bool RuntimeNumericDiffCostFunction::Evaluate(double const* const* parameters,
                                              double* residuals,
                                              double** jacobians) const 
{
    // Get the function value (residuals) at the the point to evaluate.
    bool success = function_->Evaluate(parameters, residuals, NULL);
    if (!success) {
        // Something went wrong; ignore the jacobian.
        return false;
    }
    if (!jacobians) {
        // Nothing to do; just forward.
        return true;
    }
    
    const std::vector<ceres::int16>& block_sizes = function_->parameter_block_sizes();
    CHECK(!block_sizes.empty());
    
    // Create local space for a copy of the parameters which will get mutated.
    int parameters_size = accumulate(block_sizes.begin(), block_sizes.end(), 0);
    std::vector<double> parameters_copy(parameters_size);
    std::vector<double*> parameters_references_copy(block_sizes.size());
    parameters_references_copy[0] = &parameters_copy[0];
    for (unsigned int block = 1; block < block_sizes.size(); ++block) {
        parameters_references_copy[block] = parameters_references_copy[block - 1]
            + block_sizes[block - 1];
    }
    // Copy the parameters into the local temp space.
    for (unsigned int block = 0; block < block_sizes.size(); ++block) {
        memcpy(parameters_references_copy[block],
               parameters[block],
               block_sizes[block] * sizeof(*parameters[block]));
    }
    
    for (unsigned int block = 0; block < block_sizes.size(); ++block) {
        std::cout << "JAC BLOCK " << block << std::endl;

        if (!jacobians[block]) {
            // No jacobian requested for this parameter / residual pair.
            continue;
        }
        if (!EvaluateJacobianForParameterBlock(function_,
                                               block_sizes[block],
                                               block,
                                               method_,
                                               relative_step_size_,
                                               residuals,
                                               &parameters_references_copy[0],
                                               jacobians)) {
            return false;
        }
    }
    return true;
}

ceres::CostFunction* CreateCarpRuntimeNumericDiffCostFunction(const ceres::CostFunction* cost_function,
                                                              ceres::NumericDiffMethod method,
                                                              double relative_step_size) {
    return new RuntimeNumericDiffCostFunction(cost_function,
                                              method,
                                              relative_step_size);
}
        

bool EvaluateJacobianForParameterBlock(const CostFunction* function,
                                       int parameter_block_size,
                                       int parameter_block,
                                       NumericDiffMethod method,
                                       double relative_step_size,
                                       double const* residuals_at_eval_point,
                                       double** parameters,
                                       double** jacobians) {
    using Eigen::Map;
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using Eigen::RowMajor;
    
    typedef Matrix<double, Dynamic, 1> ResidualVector;
    typedef Matrix<double, Dynamic, 1> ParameterVector;
    typedef Matrix<double, Dynamic, Dynamic, RowMajor> JacobianMatrix;
    
    int num_residuals = function->num_residuals();
    
    Map<JacobianMatrix> parameter_jacobian(jacobians[parameter_block],
                                           num_residuals,
                                           parameter_block_size);
    
    // Mutate one element at a time and then restore.
    Map<ParameterVector> x_plus_delta(parameters[parameter_block],
                                      parameter_block_size);
    ParameterVector x(x_plus_delta);
    ParameterVector step_size = x.array().abs() * relative_step_size;
    
    // To handle cases where a paremeter is exactly zero, instead use the mean
    // step_size for the other dimensions.
    double fallback_step_size = step_size.sum() / step_size.rows();
    if (fallback_step_size == 0.0) {
        // If all the parameters are zero, there's no good answer. Use the given
        // relative step_size as absolute step_size and hope for the best.
        fallback_step_size = relative_step_size;
    }
    
    // For each parameter in the parameter block, use finite differences to
    // compute the derivative for that parameter.
    for (int j = 0; j < parameter_block_size; ++j) {
        if (step_size(j) == 0.0) {
            // The parameter is exactly zero, so compromise and use the mean step_size
            // from the other parameters. This can break in many cases, but it's hard
            // to pick a good number without problem specific knowledge.
            step_size(j) = fallback_step_size;
        }
        x_plus_delta(j) = x(j) + step_size(j);
        
        ResidualVector residuals(num_residuals);
        if (!function->Evaluate(parameters, &residuals[0], NULL)) {
            // Something went wrong; bail.
            return false;
        }
        
        // Compute this column of the jacobian in 3 steps:
        // 1. Store residuals for the forward part.
        // 2. Subtract residuals for the backward (or 0) part.
        // 3. Divide out the run.
        parameter_jacobian.col(j) = residuals;
        
        double one_over_h = 1 / step_size(j);
        if (method == ceres::CENTRAL) {
            // Compute the function on the other side of x(j).
            x_plus_delta(j) = x(j) - step_size(j);
            
            if (!function->Evaluate(parameters, &residuals[0], NULL)) {
                // Something went wrong; bail.
                return false;
            }
            parameter_jacobian.col(j) -= residuals;
            one_over_h /= 2;
        } else {
            // Forward difference only; reuse existing residuals evaluation.
            parameter_jacobian.col(j) -=
                Map<const ResidualVector>(residuals_at_eval_point, num_residuals);
        }
        x_plus_delta(j) = x(j);  // Restore x_plus_delta.
        
        // Divide out the run to get slope.
        parameter_jacobian.col(j) *= one_over_h;
    }
    return true;
}

