#include "design.hpp"
#include <cmath>
namespace fsb {
// Apply the p QR reflectors to K and y.  Once the leading coordinate of a
// reflector has been eliminated, only the trailing covariance block is used.
void residual_spectrum(const mat &kernel, const mat &design,
                       const vec &phenotype, vec &eigen, vec &z) {
  const arma::uword n = design.n_rows, p = design.n_cols;
  // Wide designs leave a smaller contrast space. Blocked QR and matrix
  // multiplication avoid a long series of memory-bound rank-two updates.
  if (p >= 32 && p >= (n + 5) / 6) {
    mat q, r, basis;
    require(arma::qr(q, r, design), "design QR failed");
    const auto contrast = q.tail_cols(n - p);
    require(arma::eig_sym(eigen, basis, contrast.t() * kernel * contrast),
            "REML decomposition failed");
    z = basis.t() * (contrast.t() * phenotype);
    return;
  }
  mat k = kernel, x = design;
  vec y = phenotype;
  for (arma::uword j = 0; j < p; ++j) {
    vec v = x.col(j).tail(n - j);
    v /= arma::norm(v);
    v(0) += std::copysign(1., v(0));
    v /= arma::norm(v);
    if (j + 1 < p) {
      auto remaining = x.submat(j, j + 1, n - 1, p - 1);
      arma::rowvec projection = v.t() * remaining;
      remaining -= 2 * v * projection;
    }
    auto response = y.tail(n - j);
    response -= (2 * arma::dot(v, response)) * v;
    auto block = k.submat(j, j, n - 1, n - 1);
    vec w = block * v;
    w -= arma::dot(v, w) * v;
    // H K H = K - 2(v w' + w v') for H = I - 2v v'.
    for (arma::uword c = 0; c < v.n_elem; ++c)
      block.col(c) -= (2 * w(c)) * v + (2 * v(c)) * w;
  }
  mat basis;
  require(arma::eig_sym(eigen, basis, k.submat(p, p, n - 1, n - 1)),
          "REML decomposition failed");
  z = basis.t() * y.tail(n - p);
}
Model solve(const mat &k, const mat &x, const vec &y, const uvec &train,
            double fixed, bool variance_only) {
  if (fixed > 0) {
    FitOptions o;
    o.varcomp = "fixed";
    o.solver = "direct";
    o.lambda = fixed;
    FitAudit audit;
    return fit_model(k, x, y, train, o, audit);
  }
  require(k.n_rows == k.n_cols && k.n_rows == x.n_rows && y.n_elem == k.n_rows,
          "model dimensions");
  BalancedDesign design(x.rows(train), x);
  const mat &xt = design.train;
  vec yt = y.elem(train);
  size_t n = xt.n_rows, p = x.n_cols;
  require(p > 0 && n > p && arma::rank(xt) == p,
          "training design rank deficient or no residual degrees of freedom");
  require(yt.is_finite() && x.is_finite(),
          "nonfinite training phenotype or design");
  // Bound the magnitude before subtracting an intercept or squaring REML
  // contrasts. Power-of-two scaling preserves binary significands and lets
  // us restore variance units even when the square of the scale overflows.
  int exponent = 0;
  std::frexp(arma::abs(yt).max(), &exponent);
  if (exponent < -256 || exponent > 256)
    yt.transform([exponent](double v) { return std::scalbn(v, -exponent); });
  else
    exponent = 0;
  // Translate the response when the first design column is an intercept.
  // Contrasts and genomic predictions are invariant to this translation.
  const double offset = design.intercept < p ? yt(0) : 0.;
  yt -= offset;
  vec trend;
  require(arma::solve(trend, xt, yt, arma::solve_opts::no_approx),
          "fixed effect rank");
  double original_norm = arma::norm(yt);
  yt -= xt * trend;
  mat kt = k.submat(train, train);
  vec eigen, z;
  residual_spectrum(kt, xt, yt, eigen, z);
  require(arma::norm(z) > 64 * arma::datum::eps * original_norm,
          "constant response or exact fixed-effect fit");
  const vec z_squared = arma::square(z);
  auto profile = [&](double log_lambda) {
    // Factor out the largest denominator. The profile is invariant to this
    // scale.
    vec d = eigen + std::exp(log_lambda);
    d /= d.max();
    return (n - p) * std::log(arma::mean(z_squared / d)) +
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
  const double scaled_vu = arma::mean(z_squared / (eigen + lambda));
  const double vu = std::scalbn(scaled_vu, 2 * exponent);
  const double ve = std::scalbn(lambda * scaled_vu, 2 * exponent);
  require(
      vu > 0. && ve > 0. && std::isfinite(vu) && std::isfinite(ve),
      "variance components outside the representable range in phenotype units");
  const double objective =
      profile(best) + 2. * (n - p) * exponent * std::log(2.);
  require(std::isfinite(objective), "nonfinite REML profile objective");
  if (variance_only)
    return {{}, {}, {}, lambda, vu, ve, objective, status};
  kt.diag() += lambda;
  mat lower;
  require(arma::chol(lower, kt, "lower"),
          "training covariance factorization failed");
  mat wx = arma::solve(arma::trimatl(lower), xt);
  vec wy = arma::solve(arma::trimatl(lower), yt), b;
  require(arma::solve(b, wx, wy, arma::solve_opts::no_approx), "GLS failed");
  vec residual = arma::solve(arma::trimatu(lower.t()), wy - wx * b);
  vec u = k.cols(train) * residual;
  b += trend;
  vec yhat = design.target * b + u;
  if (design.intercept < p)
    yhat += offset * x.col(design.intercept);
  design.restore(b, offset);
  if (exponent != 0) {
    auto restore_units = [exponent](double v) {
      return std::scalbn(v, exponent);
    };
    b.transform(restore_units);
    u.transform(restore_units);
    yhat.transform(restore_units);
  }
  require(b.is_finite() && u.is_finite() && yhat.is_finite(),
          "nonfinite coefficients or predictions in phenotype units");
  return {b, u, yhat, lambda, vu, ve, objective, status};
}
} // namespace fsb
