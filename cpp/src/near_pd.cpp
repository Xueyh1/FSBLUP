// Dense C++ adaptation of Matrix 1.7-0, R/nearPD.R.
// Copyright (C) 2007-2019 Martin Maechler; nearcor.R (C) 2007 Jens
// Oehlschlägel. Original algorithm translation licensed under GPL-2.0-or-later.
// https://github.com/cran/Matrix/blob/1.7-0/R/nearPD.R
#include "near_pd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fsb {
namespace {

// V diag(d) V'. Reconstruct both triangles from the same values before the
// next symmetric LAPACK call to eliminate last-bit BLAS asymmetries.
arma::mat reconstruct(const arma::mat &vectors, const arma::vec &values) {
  arma::mat scaled = vectors;
  scaled.each_row() %= values.t();
  return arma::symmatl(scaled * vectors.t());
}

} // namespace

NearPdResult near_pd(const arma::mat &input, const NearPdOptions &options) {
  if (input.is_empty() || input.n_rows != input.n_cols || !input.is_finite())
    throw std::runtime_error("nearPD requires a nonempty finite square matrix");
  if (!std::isfinite(options.eig_tol) || options.eig_tol < 0 ||
      options.eig_tol >= 1 || !std::isfinite(options.conv_tol) ||
      options.conv_tol < 0 || !std::isfinite(options.posd_tol) ||
      options.posd_tol <= 0 || options.max_iterations <= 0)
    throw std::runtime_error("invalid nearPD tolerances or iteration limit");

  // The FSBLUP caller passes a symmetric covariance. As in nearPD's
  // ensureSymmetry path, start from the symmetric part for general inputs.
  const arma::mat original = 0.5 * input + 0.5 * input.t();
  const arma::vec original_diagonal = original.diag();
  arma::mat current = original;
  arma::mat correction(input.n_rows, input.n_cols, arma::fill::zeros);
  arma::vec eigenvalues;
  arma::mat eigenvectors;
  int iterations = 0;
  double input_min_eigen = 0.;
  double relative_change = std::numeric_limits<double>::infinity();

  auto enforce_diagonal = [&] {
    if (options.corr)
      current.diag().ones();
    else if (options.keep_diag)
      current.diag() = original_diagonal;
  };

  while (iterations < options.max_iterations &&
         relative_change > options.conv_tol) {
    arma::mat previous = std::move(current);
    arma::mat residual = previous - correction;
    if (!arma::eig_sym(eigenvalues, eigenvectors, residual))
      throw std::runtime_error("nearPD eigendecomposition failed");
    if (iterations == 0)
      input_min_eigen = eigenvalues.min();
    const arma::uvec positive =
        arma::find(eigenvalues > options.eig_tol * eigenvalues.max());
    if (positive.empty())
      throw std::runtime_error("nearPD: matrix seems negative semi-definite");
    current =
        reconstruct(eigenvectors.cols(positive), eigenvalues.elem(positive));
    correction = current - residual;
    enforce_diagonal();

    relative_change =
        arma::norm(previous - current, "inf") / arma::norm(previous, "inf");
    if (!std::isfinite(relative_change))
      throw std::runtime_error("nearPD relative change is not finite");
    ++iterations;
  }

  // Matrix's posdefify step uses the diagonal AFTER the PSD projection,
  // before flooring eigenvalues. Scaling restores that intermediate diagonal.
  if (!arma::eig_sym(eigenvalues, eigenvectors, current))
    throw std::runtime_error("nearPD final eigendecomposition failed");
  const double floor = options.posd_tol * std::abs(eigenvalues.max());
  if (eigenvalues.min() < floor) {
    const arma::vec target =
        arma::clamp(arma::vec(current.diag()), floor, arma::datum::inf);
    eigenvalues.transform([floor](double x) { return std::max(x, floor); });
    current = reconstruct(eigenvectors, eigenvalues);
    const arma::vec scale = arma::sqrt(target / current.diag());
    current.each_col() %= scale;
    current.each_row() %= scale.t();
    // dpoMatrix exposes one triangle symmetrically. Preserve that property
    // despite last-bit differences between (x*s_i)*s_j and (x*s_j)*s_i.
    current = arma::symmatl(current);
  }
  enforce_diagonal();
  if (!current.is_finite())
    throw std::runtime_error("nearPD produced a nonfinite matrix");

  const double norm_f = arma::norm(original - current, "fro");
  return {std::move(current), iterations, relative_change <= options.conv_tol,
          relative_change,    norm_f,     input_min_eigen};
}

} // namespace fsb
