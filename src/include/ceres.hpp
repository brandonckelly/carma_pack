#include "ceres/ceres.h"
#include <armadillo>
#include <carpack.hpp>

class CarpCostFunction : public ceres::CostFunction {
 public:
    
    explicit CarpCostFunction(std::shared_ptr<CARp> carp, int nterms)
        : carp_(carp), nterms_(nterms) {
        set_num_residuals(1);
        for (int i = 0; i < nterms; ++i) {
            mutable_parameter_block_sizes()->push_back(1);
        }
    }

    virtual ~CarpCostFunction() {
    }
    
    virtual bool Evaluate(double const* const* params,
                          double* residuals,
                          double** jacobians) const {
        arma::vec theta(nterms_);
        std::cout << "DEBUG ";
        for (int i = 0; i < nterms_; i++) {
            theta[i] = params[0][i];
            std::cout << params[0][i] << " ";
        }
        double chi2 = -2.0 * carp_->LogDensity(theta);
        residuals[0] = std::sqrt(chi2);
        std::cout << "logd=" << carp_->LogDensity(theta) << std::endl;
        return true;
    }


private:
    const std::shared_ptr<CARp> carp_;
    const int nterms_;
};


std::vector<double> RunCeres(std::vector<double> time, 
                             std::vector<double> y, 
                             std::vector<double> yerr, 
                             int p, int q, bool do_zcarma, bool use_cauchy);

//
// Taken from unsupported (yet) runtime_numeric_diff_cost_function.cc/h
//


class RuntimeNumericDiffCostFunction : public ceres::CostFunction {
public:
    RuntimeNumericDiffCostFunction(const ceres::CostFunction* function,
                                   ceres::NumericDiffMethod method,
                                   double relative_step_size)
        : function_(function),
          method_(method),
          relative_step_size_(relative_step_size) {
        *mutable_parameter_block_sizes() = function->parameter_block_sizes();
        set_num_residuals(function->num_residuals());
    }
    
    virtual ~RuntimeNumericDiffCostFunction() { }
    
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
private:
    const ceres::CostFunction* function_;
    ceres::NumericDiffMethod method_;
    double relative_step_size_;
};

// Don't put implementation in here, otherwise boost_python complains
//
// multiple definition of `CreateCarpRuntimeNumericDiffCostFunction(ceres::CostFunction const*, ceres::NumericDiffMethod, double)'
//
// as I don't have include guards installed
ceres::CostFunction* CreateCarpRuntimeNumericDiffCostFunction(const ceres::CostFunction* cost_function,
                                                              ceres::NumericDiffMethod method,
                                                              double relative_step_size);

bool EvaluateJacobianForParameterBlock(const ceres::CostFunction* function,
                                       int parameter_block_size,
                                       int parameter_block,
                                       ceres::NumericDiffMethod method,
                                       double relative_step_size,
                                       double const* residuals_at_eval_point,
                                       double** parameters,
                                       double** jacobians);
