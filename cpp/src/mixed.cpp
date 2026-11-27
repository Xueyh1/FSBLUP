#include "design.hpp"
#include "linear.hpp"
namespace fsb {
double solver_workspace(size_t n, size_t targets, size_t p,
                        const FitOptions &o) {
  double work = 8. * n * (14 + 4. * p) + 16. * targets * p;
  if (o.preconditioner == "nystrom")
    work += 48. * n * std::min(double(n), double(o.rank));
  if (o.trace == "hutchinson") {
    if (o.compare_interiors && p > 1 && n <= 128 && n > p && n - p <= 32 &&
        n - p < p)
      work += 48. * n * n;
    double basis = o.full_reorth ? std::min(double(n), 2. * o.slq_depth) : 5.;
    work += 8. * n *
                (basis + 12. * o.batch +
                 5. * std::min(double(n), double(o.trace_rank))) +
            o.max_probes * (96. * o.slq_depth + 8. * o.trace_rank + 128.);
  }
  if (o.solver == "direct" || (o.trace == "exact" && o.varcomp != "fixed"))
    work += 160. * n * n;
  return work;
}
Model fit_operator(const TrainKernel &input, const mat &training,
                   const mat &targets, const vec &y, const FitOptions &o,
                   FitAudit &audit) {
  string context = std::move(audit.context);
  audit = {};
  audit.context = std::move(context);
  audit.n = y.n_elem;
  size_t n = y.n_elem, p = targets.n_cols;
  require(training.n_rows == n && y.is_finite(), "invalid training response");
  double work = solver_workspace(n, targets.n_rows, p, o);
  if (work / 1048576. >= o.memory_mb)
    throw std::runtime_error("training solver needs " +
                             std::to_string(work / 1048576.) +
                             " MiB, exceeding --memory-mb; use fixed PCG or an "
                             "operator with Hutchinson traces");
  audit.workspace_mib = work / 1048576.;
  BalancedDesign design(training, targets);
  const auto &xt = design.train, &x = design.target;
  TrainKernel kernel = input;
  kernel.dense = [fn = input.dense, cost = input.dense_column_cost, &audit] {
    audit.matvec += cost;
    return fn();
  };
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
    audit.prec_rank = keep.n_elem;
    kernel.low_rank = std::make_shared<mat>(std::move(f));
    audit.prec_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tick)
            .count();
  }
  int initial = 0, center = 0;
  vec yt = y;
  std::frexp(arma::abs(yt).max(), &initial);
  yt.transform([&](double v) { return std::scalbn(v, -initial); });
  const bool intercept = design.intercept < p;
  double offset = intercept ? yt[0] : 0;
  yt -= offset;
  vec trend;
  require(arma::solve(trend, xt, yt, arma::solve_opts::no_approx),
          "fixed effect rank");
  double response_norm = arma::norm(yt);
  yt -= xt * trend;
  require(arma::norm(yt) > 64 * arma::datum::eps * response_norm,
          "constant response or exact fixed-effect fit");
  std::frexp(arma::abs(yt).max(), &center);
  yt.transform([&](double v) { return std::scalbn(v, -center); });
  int exponent = initial + center;
  audit.scale_exponent = exponent;
  audit.fixed_rank = p;
  double g = 1, e = o.lambda, lik = NAN, objective = NAN;
  string status = "fixed";
  auto stage = std::chrono::steady_clock::now();
  if (o.varcomp != "fixed") {

    double v = arma::dot(yt, yt) / (n - p);
    double g0 = o.initial_g ? std::scalbn(o.initial_g, -2 * exponent) : v / 2;
    double e0 = o.initial_e ? std::scalbn(o.initial_e, -2 * exponent) : v / 2;
    if (o.initial_g > 0 && o.initial_e > 0 &&
        (!std::isfinite(g0 + e0) || g0 == 0 || e0 == 0)) {
      // The requested components can be representable in user units while
      // the fold's power-of-two units overflow. Preserve their proportions.
      long double total = static_cast<long double>(o.initial_g) + o.initial_e;
      g0 = v * (static_cast<long double>(o.initial_g) / total);
      e0 = v * (static_cast<long double>(o.initial_e) / total);
    }
    require(g0 > 0 && e0 > 0 && std::isfinite(g0 + e0),
            "invalid initial variances");
    audit.initial_vu = std::scalbn(g0, 2 * exponent);
    audit.initial_ve = std::scalbn(e0, 2 * exponent);
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
  auto [b, r, res] = P.fit_response(yt, "y");
  audit.gls_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage)
          .count();
  stage = std::chrono::steady_clock::now();
  vec u(x.n_rows, arma::fill::zeros);
  if (g > 0) {
    u = g * kernel.cross(r);
    ++audit.cross_rhs;
  }
  if (o.varcomp == "fixed") {
    g = arma::dot(res, r) / (n - p);
    e = o.lambda * g;
    require(g > 0, "nonpositive fixed residual variance");
  }
  vec yh = x * b + u;
  vec fixed_trend = x * trend;
  for (size_t i = 0; i < yh.n_elem; ++i)
    yh[i] =
        std::scalbn((std::scalbn(yh[i], center) + fixed_trend[i]) +
                        (intercept ? offset * targets(i, design.intercept) : 0),
                    initial);
  for (auto &v : u)
    v = std::scalbn(v, exponent);
  for (size_t j = 0; j < p; ++j)
    b[j] = std::scalbn(b[j], center) + trend[j];
  design.restore(b, offset);
  for (double &v : b)
    v = std::scalbn(v, initial);
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
  require(k.n_rows > 0, "empty fitting kernel");
  uvec ids = arma::regspace<uvec>(0, k.n_rows - 1);
  return fit_dense(k, ids, x, y, t, o, a);
}
Model fit_dense(const mat &k, const uvec &ids, const mat &x, const vec &y,
                const uvec &t, const FitOptions &o, FitAudit &a) {
  require(!ids.empty() && !t.empty() && k.n_rows == k.n_cols &&
              ids.n_elem == x.n_rows && y.n_elem == x.n_rows &&
              ids.max() < k.n_rows && t.max() < ids.n_elem,
          "invalid fitting dimensions or training indices");
  bool same = ids.n_elem == k.n_rows &&
              arma::all(ids == arma::regspace<uvec>(0, k.n_rows - 1));
  if (o.varcomp == "profile-reml") {
    auto m = same ? solve(k, x, y, t, 0, o.solver == "pcg")
                  : solve(k.submat(ids, ids), x, y, t, 0, o.solver == "pcg");
    if (o.solver == "direct") {
      a.n = t.n_elem;
      a.status = m.status;
      a.final_lik = -.5 * (m.objective + (t.n_elem - x.n_cols));
      return m;
    }
    FitOptions fixed = o;
    fixed.varcomp = "fixed";
    fixed.lambda = m.lambda;
    auto result = fit_dense(k, ids, x, y, t, fixed, a);
    result.objective = m.objective;
    result.status = m.status;
    a.status = m.status;
    a.final_lik = -.5 * (m.objective + (t.n_elem - x.n_cols));
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
