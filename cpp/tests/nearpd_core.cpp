#include "near_pd.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
void check(bool pass, const std::string &message) {
  if (!pass)
    throw std::runtime_error(message);
}

void read_matrix(std::istream &in, arma::mat &m) {
  for (arma::uword i = 0; i < m.n_rows; ++i)
    for (arma::uword j = 0; j < m.n_cols; ++j)
      check(bool(in >> m(i, j)), "incomplete nearPD fixture");
}

void check_case(const std::string &line) {
  std::istringstream row(line);
  std::string name;
  int n, it;
  bool converged;
  double rel, norm;
  fsb::NearPdOptions opts;
  check(bool(row >> name >> n >> opts.corr >> opts.keep_diag >> opts.max_iterations >>
             opts.eig_tol >> opts.conv_tol >> opts.posd_tol >> it >> converged >> rel >> norm),
        "invalid nearPD case header");
  arma::mat x(n, n), expected(n, n);
  read_matrix(row, x);
  read_matrix(row, expected);
  auto result = fsb::near_pd(x, opts);
  check(result.iterations == it && result.converged == converged, name + ": convergence");
  double scale = std::max(1., arma::abs(expected).max());
  double error = arma::abs(result.matrix - expected).max();
  check(error <= 5e-12 * scale, name + ": matrix mismatch");
  check(std::abs(result.norm_f - norm) <= 1e-11 * std::max(scale, norm),
        name + ": Frobenius distance");
  check(std::abs(result.rel_tol - rel) < 2e-12 * std::max(1., rel), name + ": rel.tol");
  const double min_input = arma::eig_sym(0.5 * x + 0.5 * x.t()).min();
  check(std::abs(result.input_min_eigen - min_input) <= 2e-12 * scale,
        name + ": minimum eigenvalue of input");
  check(result.matrix.is_symmetric(2e-12 * scale), name + ": symmetry");
  arma::mat lower;
  check(arma::chol(lower, result.matrix, "lower"), name + ": Cholesky");
  if (opts.corr)
    check(arma::all(result.matrix.diag() == 1.), name + ": unit diagonal");
  else if (opts.keep_diag)
    check(arma::all(result.matrix.diag() == x.diag()), name + ": fixed diagonal");
  std::cout << name << " iterations=" << it << " max_abs_delta=" << error << '\n';
}

bool rejects(const arma::mat &x, const fsb::NearPdOptions &options = {}) {
  try {
    fsb::near_pd(x, options);
  } catch (const std::exception &) {
    return true;
  }
  return false;
}
} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 2, "usage: nearpd_core_test nearpd_cases.tsv");
    std::ifstream input(argv[1]);
    check(input.is_open(), "unreadable nearPD fixture");
    std::string line;
    size_t count = 0;
    while (std::getline(input, line))
      if (!line.empty() && line[0] != '#') {
        check_case(line);
        ++count;
      }
    check(count == 12, "incomplete fixture case count");
    check(rejects(arma::zeros<arma::mat>(2, 2)), "zero spectrum");
    check(rejects(-arma::eye<arma::mat>(3, 3)), "negative-definite input");
    check(rejects(arma::mat()), "empty input");
    check(rejects(arma::ones<arma::mat>(3, 2)), "rectangular input");
    arma::mat invalid = arma::eye<arma::mat>(2, 2);
    invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
    check(rejects(invalid), "nonfinite input");
    const arma::mat identity = arma::eye<arma::mat>(2, 2);
    fsb::NearPdOptions options;
    options.max_iterations = 0;
    check(rejects(identity, options), "zero iteration limit");
    options = {};
    options.posd_tol = 0.;
    check(rejects(identity, options), "nonpositive eigenvalue floor");
    options = {};
    options.conv_tol = std::numeric_limits<double>::infinity();
    check(rejects(identity, options), "nonfinite convergence tolerance");
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
