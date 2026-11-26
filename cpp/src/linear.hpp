#pragma once
#include "core.hpp"
namespace fsb {
struct System {
  const TrainKernel &k;
  double g, e;
  const FitOptions &opt;
  FitAudit &audit;
  vec diag;
  mat lower, scaled_features, woodbury;
  System(const TrainKernel &kernel, double genetic, double noise,
         const FitOptions &o, FitAudit &a)
      : k(kernel), g(genetic), e(noise), opt(o), audit(a),
        diag(g * k.diagonal + e) {
    require(diag.is_finite(), "nonfinite covariance diagonal");
    require(diag.min() > 0, "nonpositive covariance diagonal");
    if (opt.preconditioner == "nystrom" && g > 0 && !k.low_rank->empty()) {
      auto tick = std::chrono::steady_clock::now();
      mat f = std::sqrt(g) * (*k.low_rank);
      diag -= arma::sum(arma::square(f), 1);
      diag = arma::clamp(
          diag, std::max(e, arma::datum::eps * arma::mean(k.diagonal) * g),
          arma::datum::inf);
      scaled_features = f;
      scaled_features.each_col() /= diag;
      require(
          arma::chol(woodbury,
                     arma::eye(f.n_cols, f.n_cols) + f.t() * scaled_features,
                     "lower"),
          "Nystrom preconditioner factor failed");
      audit.prec_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - tick)
              .count();
    }
    if (opt.solver == "direct" && g > 0) {
      mat v = g * k.dense();
      v.diag() += e;
      require(arma::chol(lower, v, "lower"), "direct factorization failed");
    }
  }
  mat precondition(mat v) const {
    if (opt.preconditioner == "identity")
      return v;
    mat z = v;
    z.each_col() /= diag;
    if (!woodbury.empty())
      z -= scaled_features * arma::solve(arma::trimatu(woodbury.t()),
                                         arma::solve(arma::trimatl(woodbury),
                                                     scaled_features.t() * v));
    return z;
  }
  vec apply(const vec &v) {
    if (g == 0)
      return e * v;
    ++audit.matvec;
    return g * k.apply(v) + e * v;
  }
  vec solve_one(const vec &b, const std::string &rhs) {
    ++audit.rhs;
    if (opt.trace_rhs && opt.event)
      opt.event(audit.context + " rhs=" + rhs + " scale_g=" +
                std::to_string(g) + " scale_e=" + std::to_string(e));
    double norm = arma::norm(b);
    require(b.is_finite() && std::isfinite(norm), "nonfinite RHS " + rhs);
    vec x(b.n_elem, arma::fill::zeros);
    if (norm == 0) {
      ++audit.zero_rhs;
      if (opt.event)
        opt.event(audit.context + " rhs=" + rhs + " zero-rhs");
      return x;
    }
    const double tol = opt.atol + opt.rtol * norm;
    auto verify = [&](const vec &r) { return arma::norm(r) <= tol; };
    auto success = [&](const vec &r, int it) {
      audit.residual_ratio =
          std::max(audit.residual_ratio, arma::norm(r) / tol);
      audit.max_iterations = std::max(audit.max_iterations, size_t(it));
      if (opt.trace_rhs && opt.event)
        opt.event(audit.context + " rhs=" + rhs +
                  " iterations=" + std::to_string(it) +
                  " true_residual=" + std::to_string(arma::norm(r)));
    };
    if (g == 0) {
      x = b / e;
      vec r = b - e * x;
      require(verify(r), "boundary system residual failed");
      success(r, 0);
      return x;
    }
    if (opt.solver == "direct") {
      x = arma::solve(arma::trimatu(lower.t()),
                      arma::solve(arma::trimatl(lower), b));
      vec r = b - apply(x);
      require(verify(r), "direct true residual failed rhs=" + rhs);
      success(r, 1);
      return x;
    }
    vec r = b, z = r;
    z = precondition(r);
    vec p = z;
    double rho = arma::dot(r, z);
    for (int it = 1; it <= opt.max_pcg; ++it) {
      vec ap = apply(p);
      double curvature = arma::dot(p, ap);
      require(ap.is_finite() && std::isfinite(curvature) && std::isfinite(rho),
              "PCG nonfinite recurrence rhs=" + rhs);
      require(curvature > 0 && rho > 0, "PCG nonpositive curvature rhs=" + rhs);
      double step = rho / curvature;
      vec next = x + step * p;
      bool stalled = arma::all(next == x);
      x = std::move(next);
      r -= step * ap;
      require(x.is_finite() && r.is_finite(),
              "PCG nonfinite iterate rhs=" + rhs);
      if (opt.event && it % opt.log_every == 0)
        opt.event(audit.context + " rhs=" + rhs + " pcg=" + std::to_string(it) +
                  " residual=" + std::to_string(arma::norm(r)));
      if (verify(r) || stalled || it == opt.max_pcg) {
        r = b - apply(x);
        if (verify(r)) {
          success(r, it);
          return x;
        }
        ++audit.residual_restarts;
        require(!stalled, "PCG stagnation rhs=" + rhs);
        require(it < opt.max_pcg,
                "PCG iteration-limit rhs=" + rhs +
                    " true_residual=" + std::to_string(arma::norm(r)));
        z = r;
        z = precondition(r);
        p = z;
        rho = arma::dot(r, z);
        continue;
      }
      z = r;
      z = precondition(r);
      double nr = arma::dot(r, z);
      p = z + (nr / rho) * p;
      rho = nr;
    }
    throw std::runtime_error("invalid PCG iteration limit");
  }
};
struct Projection {
  System &sys;
  const mat &x;
  mat w, lower;
  Projection(System &s, const mat &design)
      : sys(s), x(design), w(x.n_rows, x.n_cols) {
    for (size_t j = 0; j < x.n_cols; ++j)
      w.col(j) = sys.solve_one(x.col(j), "X" + std::to_string(j));
    mat c = x.t() * w;
    c = .5 * c + .5 * c.t();
    require(arma::rcond(c) > 1e-14 && arma::chol(lower, c, "lower"),
            "ill-conditioned fixed-effect system");
  }
  vec small(const vec &rhs) const {
    return arma::solve(arma::trimatu(lower.t()),
                       arma::solve(arma::trimatl(lower), rhs));
  }
  vec project(const vec &h, vec &b) const {
    b = small(x.t() * h);
    return h - w * b;
  }
  struct Fit {
    vec b, r, residual;
  };
  Fit fit_response(const vec &y, const string &rhs) {
    vec b, r = project(sys.solve_one(y, rhs), b), res = y - x * b;
    const auto &o = sys.opt;
    auto &a = sys.audit;
    auto ratio = [](double norm, double tol) {
      return tol > 0 ? norm / tol : (norm == 0 ? 0. : arma::datum::inf);
    };
    auto check = [&] {
      a.projected_residual = ratio(arma::norm(sys.apply(r) - res),
                                   o.atol + o.rtol * arma::norm(res));
      a.orthogonality =
          ratio(arma::norm(x.t() * r),
                o.atol + o.rtol * arma::norm(x, "fro") * arma::norm(r));
      return a.projected_residual <= 1 && a.orthogonality <= 1;
    };
    if (!check()) {
      ++a.centered_solves;
      r = sys.solve_one(res, "centered-" + rhs);
      vec correction = small(x.t() * r);
      b += correction;
      r -= w * correction;
      res = y - x * b;
      require(
          check(),
          "projected residual or orthogonality failed after centering rhs=" +
              rhs);
    }
    return {std::move(b), std::move(r), std::move(res)};
  }
  vec apply_P(const vec &v) { return fit_response(v, "P").r; }
};
} // namespace fsb
