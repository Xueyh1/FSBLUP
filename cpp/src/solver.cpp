#include "core.hpp"
namespace fsb {
Model solve(const mat &k, const mat &x, const vec &y, const uvec &train,
            double fixed) {
  require(k.n_rows == k.n_cols && k.n_rows == x.n_rows && y.n_elem == k.n_rows,
          "model dimensions");
  mat xt = x.rows(train);
  vec yt = y.elem(train);
  size_t n = xt.n_rows, p = x.n_cols;
  require(p > 0 && n > p && arma::rank(xt) == p,
          "training design rank deficient or no residual degrees of freedom");
  require(yt.is_finite() && x.is_finite(),
          "nonfinite training phenotype or design");
  mat q, r;
  require(arma::qr(q, r, xt), "design QR failed");
  mat contrast = q.tail_cols(n - p);
  mat kt = k.submat(train, train);
  vec eigen;
  mat basis;
  require(arma::eig_sym(eigen, basis, contrast.t() * kt * contrast),
          "REML decomposition failed");
  vec z = basis.t() * (contrast.t() * yt);
  require(arma::norm(z) > 64 * arma::datum::eps * arma::norm(yt),
          "constant response or exact fixed-effect fit");
  auto profile = [&](double log_lambda) {
    // Factor out the largest denominator. The profile is invariant to this
    // scale.
    vec d = eigen + std::exp(log_lambda);
    d /= d.max();
    return (n - p) * std::log(arma::mean(arma::square(z) / d)) +
           arma::accu(arma::log(d));
  };
  const double lo = std::log(1e-9), hi = std::log(1e9);
  double best = lo;
  string status;
  if (fixed > 0) {
    best = std::log(fixed);
    status = "fixed";
  } else if (eigen.max() - eigen.min() <=
             1e-12 * std::max(1., std::abs(eigen.max()))) {
    best = 0;
    status = "unidentifiable;lambda=1";
  } else {
    constexpr int bins = 64;
    double points[bins + 1], loss[bins + 1];
    for (int i = 0; i <= bins; ++i) {
      points[i] = lo + (hi - lo) * i / bins;
      loss[i] = profile(points[i]);
    }
    best = loss[0] <= loss[bins] ? lo : hi;
    double best_loss = profile(best);
    // Bracket each visible local minimum, including minima near the bounds.
    for (int i = 0; i <= bins; ++i) {
      if ((i && loss[i] > loss[i - 1]) || (i < bins && loss[i] > loss[i + 1]))
        continue;
      double a = points[std::max(0, i - 1)], b = points[std::min(bins, i + 1)];
      for (int j = 0; j < 65; ++j) {
        double m = a + (b - a) / 3, h = b - (b - a) / 3;
        if (profile(m) <= profile(h))
          b = h;
        else
          a = m;
      }
      double mid = (a + b) / 2, val = profile(mid);
      if (val < best_loss - 1e-13) {
        best = mid;
        best_loss = val;
      }
    }
    status = best == lo   ? "lower-bound"
             : best == hi ? "upper-bound"
                          : "interior";
  }
  double lambda = std::exp(best);
  double vu = arma::mean(arma::square(z) / (eigen + lambda));
  kt.diag() += lambda;
  mat lower;
  require(arma::chol(lower, kt, "lower"),
          "training covariance factorization failed");
  mat wx = arma::solve(arma::trimatl(lower), xt);
  vec wy = arma::solve(arma::trimatl(lower), yt), b;
  require(arma::solve(b, wx, wy, arma::solve_opts::no_approx), "GLS failed");
  vec residual = arma::solve(arma::trimatu(lower.t()), wy - wx * b);
  vec u = k.cols(train) * residual;
  return {b, u, x * b + u, lambda, vu, lambda * vu, profile(best), status};
}
} // namespace fsb
