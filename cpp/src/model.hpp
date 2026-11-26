#pragma once
#include <armadillo>
#include <functional>
#include <string>
#include <vector>

namespace fsb {
struct FitOptions {
  std::string solver="pcg", varcomp="ai-reml", trace="exact", preconditioner="jacobi";
  double lambda=0,rtol=1e-11,atol=1e-13,score_tol=1e-8,initial_g=0,initial_e=0;
  int max_pcg=2000,max_reml=200,log_every=10;
  std::function<void(const std::string&)> event;
};
struct Iteration {
  int iteration,backtracks;
  double g,e,lik,sg,se,step,condition;
  std::string status;
};
struct FitAudit {
  std::string context,status;
  size_t n=0,rhs=0,matvec=0,max_iterations=0;
  double residual_ratio=0, projected_residual=0, orthogonality=0;
  std::vector<Iteration> steps;
};
// The training operator and cross operator close over one immutable set of training IDs.
struct TrainKernel {
  arma::vec diagonal;
  std::function<arma::mat(const arma::mat&)> apply,cross;
  std::function<arma::mat()> dense;
};
struct Contrast {arma::vec eigen,z;};
struct Estimates {double g,e,lik; std::string status;};
struct Derivatives {double lik; arma::vec score; arma::mat information;};
Derivatives reml_derivatives(const Contrast&,double g,double e);
Estimates estimate_exact(const Contrast&,size_t n,double g,double e,const FitOptions&,FitAudit&);
}
