#include "linear.hpp"
namespace fsb {
Model fit_operator(const TrainKernel &input, const mat &xt, const mat &x,
                   const vec &y, const FitOptions &o, FitAudit &audit) {
  audit.n = y.n_elem;
  size_t n = y.n_elem, p = x.n_cols;
  require(xt.n_rows == n && xt.n_cols == p && n > p && p && arma::rank(xt) == p,
          "training design rank deficient or no residual degrees of freedom");
  require(y.is_finite() && xt.is_finite() && x.is_finite(),
          "nonfinite training phenotype or design");
  double work = 8. * y.n_elem * (14 + 4. * x.n_cols) + 16. * x.n_elem;
  if (o.preconditioner == "nystrom")
    work += 48. * y.n_elem * std::min(double(y.n_elem), double(o.rank));
  if (o.trace == "hutchinson")
    work += 8. * y.n_elem *
                (std::min(double(y.n_elem), 2. * o.slq_depth) + 12. * o.batch) +
            o.max_probes * (96. * o.slq_depth + 64.);
  if (o.solver == "direct" || (o.trace == "exact" && o.varcomp != "fixed"))
    work += 160. * y.n_elem * y.n_elem;
  require(work / 1048576. < o.memory_mb,
          "training solver workspace exceeds --memory-mb");
  audit.workspace_mib = work / 1048576.;
  TrainKernel kernel = input;
  kernel.apply = [fn = input.apply, &audit](const mat &v) {
    auto start = std::chrono::steady_clock::now();
    mat res = fn(v);
    audit.operator_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return res;
  };
  if (o.preconditioner == "nystrom") {
    auto tick = std::chrono::steady_clock::now();
    size_t n = y.n_elem, m = std::min(n, size_t(o.rank));
    mat basis(n, m, arma::fill::zeros);
    uvec ix(m);
    for (size_t i = 0; i < m; ++i) {
      ix[i] = (n * i) / m;
      basis(ix[i], i) = 1;
    }
    mat cols = kernel.apply(basis);
    audit.matvec += m;
    mat w = cols.rows(ix);
    vec d;
    mat q;
    require(arma::eig_sym(d, q, .5 * w + .5 * w.t()),
            "Nystrom eigensolve failed");
    uvec keep = arma::find(d > arma::datum::eps * m * d.max());
    mat f = cols * q.cols(keep);
    f.each_row() /= arma::sqrt(d.elem(keep)).t();
    kernel.low_rank = std::make_shared<mat>(std::move(f));
    audit.prec_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tick)
            .count();
  }
  int initial = 0, center = 0;
  vec yt = y;
  std::frexp(arma::abs(yt).max(), &initial);
  yt.transform([&](double v) { return std::scalbn(v, -initial); });
  double offset = arma::all(x.col(0) == 1) ? yt[0] : 0;
  yt -= offset;
  std::frexp(arma::abs(yt).max(), &center);
  yt.transform([&](double v) { return std::scalbn(v, -center); });
  int exponent = initial + center;
  audit.scale_exponent = exponent;
  audit.fixed_rank = p;
  vec ols;
  require(arma::solve(ols, xt, yt, arma::solve_opts::no_approx),
          "fixed effect rank");
  require(arma::norm(yt - xt * ols) > 64 * arma::datum::eps * arma::norm(yt),
          "constant response or exact fixed-effect fit");
  double g = 1, e = o.lambda, lik = NAN, objective = NAN;
  string status = "fixed";
  auto stage = std::chrono::steady_clock::now();
  if (o.varcomp != "fixed") {

    double v = arma::var(yt);
    if (v == 0)
      v = arma::dot(yt, yt) / n;
    double g0 = o.initial_g ? std::scalbn(o.initial_g, -2 * exponent) : v / 2;
    double e0 = o.initial_e ? std::scalbn(o.initial_e, -2 * exponent) : v / 2;
    require(g0 > 0 && e0 > 0 && std::isfinite(g0 + e0),
            "invalid initial variances");
    Estimates est;
    if (o.trace == "hutchinson")
      est = estimate_stochastic(kernel, xt, yt, g0, e0, o, audit);
    else {
      Contrast c;
      mat dense = kernel.dense();
      residual_spectrum(dense, xt, yt, c.eigen, c.z);
      est = estimate_exact(c, n, g0, e0, o, audit);
      if (est.e == 0) {
        mat L;
        require(arma::chol(L, dense),
                "residual-zero boundary requires SPD training covariance");
      }
    }
    g = est.g;
    e = est.e;
    status = est.status;
    lik = est.lik - double(n - p) * exponent * std::log(2.);
    // Historical profile objective = -2 contrast likelihood - df at a profiled
    // scale.
    if (o.trace == "exact")
      objective = -2 * lik - (n - p);
  }
  audit.estimation_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage)
          .count();
  stage = std::chrono::steady_clock::now();
  System sys(kernel, g, e, o, audit);
  Projection P(sys, xt);
  vec b, r = P.project(sys.solve_one(yt, "y"), b), res = yt - xt * b;
  auto check = [&] {
    audit.projected_residual =
        arma::norm(sys.apply(r) - res) / (o.atol + o.rtol * arma::norm(res));
    audit.orthogonality =
        arma::norm(xt.t() * r) /
        (o.atol + o.rtol * arma::norm(xt, "fro") * arma::norm(r));
    return audit.projected_residual <= 1 && audit.orthogonality <= 1;
  };
  if (!check()) {
    r = sys.solve_one(res, "centered");
    vec correction = P.small(xt.t() * r);
    b += correction;
    r -= P.w * correction;
    res = yt - xt * b;
    require(check(),
            "projected residual or orthogonality failed after centering");
  }
  audit.gls_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage)
          .count();
  stage = std::chrono::steady_clock::now();
  vec u(x.n_rows, arma::fill::zeros);
  if (g > 0)
    u = g * kernel.cross(r);
  if (o.varcomp == "fixed") {
    g = arma::dot(res, r) / (n - p);
    e = o.lambda * g;
    require(g > 0, "nonpositive fixed residual variance");
  }
  vec yh = x * b + u;
  for (auto &v : yh)
    v = std::scalbn(std::scalbn(v, center) + offset, initial);
  for (auto &v : u)
    v = std::scalbn(v, exponent);
  for (size_t j = 0; j < p; ++j)
    b[j] = std::scalbn(std::scalbn(b[j], center) + (j == 0 ? offset : 0.),
                       initial);
  double vu = std::scalbn(g, 2 * exponent), ve = std::scalbn(e, 2 * exponent);
  require(
      std::isfinite(vu) && std::isfinite(ve) && (!g || vu > 0) &&
          (!e || ve > 0),
      "variance components outside the representable range in phenotype units");
  require(b.is_finite() && u.is_finite() && yh.is_finite(),
          "nonfinite predictions in phenotype units");
  audit.predict_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage)
          .count();
  audit.final_lik = lik;
  /* diagnostics are kept in scaled units until TSV serialization, including
   * failed candidates. */
  audit.status = status;
  return {b, u, yh, g ? e / g : INFINITY, vu, ve, objective, status};
}
Model fit_model(const mat &k, const mat &x, const vec &y, const uvec &t,
                const FitOptions &o, FitAudit &a) {
  uvec ids = arma::regspace<uvec>(0, k.n_rows - 1);
  return fit_dense(k, ids, x, y, t, o, a);
}
Model fit_dense(const mat &k, const uvec &ids, const mat &x, const vec &y,
                const uvec &t, const FitOptions &o, FitAudit &a) {
  bool same = ids.n_elem == k.n_rows &&
              arma::all(ids == arma::regspace<uvec>(0, k.n_rows - 1));
  if (o.varcomp == "profile-reml") {
    auto m = same ? solve(k, x, y, t, 0, o.solver == "pcg")
                  : solve(k.submat(ids, ids), x, y, t, 0, o.solver == "pcg");
    if (o.solver == "direct") {
      a.n = t.n_elem;
      a.status = m.status;
      return m;
    }
    FitOptions fixed = o;
    fixed.varcomp = "fixed";
    fixed.lambda = m.lambda;
    auto result = fit_dense(k, ids, x, y, t, fixed, a);
    result.objective = m.objective;
    result.status = m.status;
    a.status = m.status;
    return result;
  }
  uvec gt = ids.elem(t);
  mat kt = k.submat(gt, gt);
  TrainKernel op{kt.diag(), [&](const mat &v) -> mat { return kt * v; },
                 [&](const mat &v) -> mat {
                   mat u(ids.n_elem, v.n_cols);
                   for (size_t i = 0; i < ids.n_elem; i += 1024) {
                     size_t j = std::min(i + 1024, size_t(ids.n_elem)) - 1;
                     u.rows(i, j) = k.submat(ids.subvec(i, j), gt) * v;
                   }
                   return u;
                 },
                 [&] { return kt; }};
  for (auto i : t)
    op.ids.push_back(o.ids ? (*o.ids)[i] : std::to_string(i));
  return fit_operator(op, x.rows(t), x, y.elem(t), o, a);
}
} // namespace fsb
