#pragma once
#include "core.hpp"

namespace fsb {
// Estimate the change of basis on training rows only. Preserve column order
// so published coefficients correspond to the caller's original terms.
struct BalancedDesign {
  mat train, target, triangular;
  vec origin, scale;
  arma::uword intercept;
  BalancedDesign(const mat &xt, const mat &x)
      : train(xt), target(x), origin(x.n_cols, arma::fill::zeros),
        scale(x.n_cols, arma::fill::ones), intercept(x.n_cols) {
    require(train.n_rows > train.n_cols && train.n_cols &&
                train.n_cols == target.n_cols && train.is_finite() &&
                target.is_finite(),
            "training design rank deficient or no residual degrees of freedom");
    for (arma::uword j = 0; j < train.n_cols; ++j)
      if (arma::all(train.col(j) == 1)) {
        intercept = j;
        break;
      }
    for (arma::uword j = 0; j < train.n_cols; ++j) {
      if (j == intercept)
        continue;
      if (intercept < train.n_cols) {
        origin[j] = train(0, j);
        train.col(j) -= origin[j];
        target.col(j) -= origin[j];
      }
      scale[j] = arma::abs(train.col(j)).max();
      require(scale[j] > 0 && std::isfinite(scale[j]),
              "training design rank deficient");
      train.col(j) /= scale[j];
      target.col(j) /= scale[j];
    }
    if (train.n_cols == 1)
      return;
    mat q;
    require(arma::qr_econ(q, triangular, train), "training design QR failed");
    vec singular = arma::svd(triangular);
    require(singular.min() >
                double(train.n_rows) * arma::datum::eps * singular.max(),
            "training design rank deficient");
    double factor = std::sqrt(double(train.n_rows));
    triangular /= factor;
    target = arma::solve(arma::trimatl(triangular.t()), target.t(),
                         arma::solve_opts::no_approx)
                 .t();
    train = q * factor;
  }
  void restore(vec &b, double offset = 0) const {
    if (!triangular.empty())
      b = arma::solve(arma::trimatu(triangular), b,
                      arma::solve_opts::no_approx);
    b /= scale;
    if (intercept < b.n_elem)
      b[intercept] += offset - arma::dot(origin, b);
  }
};
} // namespace fsb
