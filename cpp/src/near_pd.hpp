#pragma once

#include <armadillo>

namespace fsb {

// Dense covariance defaults from Matrix 1.7-0; Dykstra correction and the
// final positive-eigenvalue step are always enabled.
struct NearPdOptions {
  double eig_tol = 1e-6;
  double conv_tol = 1e-7;
  double posd_tol = 1e-8;
  int max_iterations = 100;
  bool corr = false;
  bool keep_diag = false;
};

struct NearPdResult {
  arma::mat matrix;
  int iterations;
  bool converged;
  double rel_tol;
  double norm_f;
  double input_min_eigen;
};

NearPdResult near_pd(const arma::mat &input, const NearPdOptions &options = {});

} // namespace fsb
