#pragma once
#include <armadillo>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fsb {
struct FitOptions {
  std::string solver = "pcg", varcomp = "ai-reml", trace = "exact",
              preconditioner = "jacobi";
  double lambda = 0, rtol = 1e-11, atol = 1e-13, score_tol = 1e-8,
         initial_g = 0, initial_e = 0;
  int rank = 16, trace_rank = 0, record_every = 10;
  int max_pcg = 2000, max_reml = 200, log_every = 10, max_backtracks = 54;
  size_t probes = 16, max_probes = 256, slq_depth = 12, batch = 8,
         slq_batch = 1;
  bool full_reorth = true, probe_lanczos = false, trace_rhs = false;
  double memory_mb = 8192;
  unsigned seed = 42;
  double mc_tol = .025, slq_tol = 1e-5;
  const std::vector<std::string> *ids = nullptr;
  std::function<void(const std::string &)> event;
};
struct Iteration {
  int iteration, backtracks;
  double g, e, lik, sg, se, step, condition;
  std::string status;
  size_t probes = 0;
  double paired_se = 0, depth_error = 0;
  int information_rank = -1;
  double face_gain = NAN;
  size_t face_matvec = 0;
};
struct FitAudit {
  std::string context, status;
  size_t n = 0, rhs = 0, matvec = 0, max_iterations = 0, zero_rhs = 0,
         residual_restarts = 0, centered_solves = 0, prec_rank = 0;
  double residual_ratio = 0, projected_residual = 0, orthogonality = 0;
  int scale_exponent = 0;
  size_t fixed_rank = 0, updates = 0, cross_rhs = 0;
  double operator_seconds = 0, workspace_mib = 0, estimation_seconds = 0,
         gls_seconds = 0, predict_seconds = 0;
  double initial_vu = NAN, initial_ve = NAN, final_lik = NAN, prec_seconds = 0;
  size_t probes = 0, slq_matvec = 0, trials = 0;
  double trace_identity_error = 0, logdet_se = 0, trace_se_g = 0,
         trace_se_e = 0, paired_se = 0, depth_error = 0, logratio_se = 0;
  std::vector<Iteration> steps;
};
// The training operator and cross operator close over one immutable set of
// training IDs.
struct TrainKernel {
  arma::vec diagonal;
  std::function<arma::mat(const arma::mat &)> apply, cross;
  std::function<arma::mat()> dense;
  std::vector<std::string> ids;
  std::shared_ptr<const arma::mat> low_rank;
  size_t dense_column_cost = 0;
};
struct Contrast {
  arma::vec eigen, z;
};
struct Estimates {
  double g, e, lik;
  std::string status;
};
struct Derivatives {
  double lik;
  arma::vec score;
  arma::mat information;
};
double solver_workspace(size_t n, size_t targets, size_t p, const FitOptions &);
Derivatives reml_derivatives(const Contrast &, double g, double e);
Estimates estimate_stochastic(const TrainKernel &, const arma::mat &,
                              const arma::vec &, double, double,
                              const FitOptions &, FitAudit &);
Estimates estimate_exact(const Contrast &, size_t n, double g, double e,
                         const FitOptions &, FitAudit &);
} // namespace fsb
