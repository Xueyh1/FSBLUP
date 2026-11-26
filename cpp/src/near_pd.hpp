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

// Copy time is a diagnostic subset of the other three timers.
struct PdTimes {
  double eigen = 0, reconstruction = 0, matrix_work = 0, copy = 0;
  void add(const PdTimes &other) {
    eigen += other.eigen;
    reconstruction += other.reconstruction;
    matrix_work += other.matrix_work;
    copy += other.copy;
  }
};

struct NearPdResult {
  arma::mat matrix;
  int iterations;
  bool converged;
  double rel_tol;
  double norm_f;
  double input_min_eigen;
  PdTimes timing;
};

NearPdResult near_pd(const arma::mat &input, const NearPdOptions &options = {});

} // namespace fsb
