#include "linear.hpp"
#include <cstdint>
namespace fsb {
namespace {
uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
struct Rule {
  vec nodes, weights;
  double first() const { return arma::dot(weights, nodes); }
  double logdet(double g, double e) const {
    return arma::dot(weights, arma::log(g * nodes + e));
  }
};
struct ProbeSet {
  const TrainKernel &k;
  const FitOptions &o;
  FitAudit &audit;
  std::vector<uint64_t> keys;
  using Pair = std::pair<Rule, Rule>;
  std::vector<Pair> rules;
  std::optional<Pair> design_rule, response_rule;
  double traceK;
  ProbeSet(const TrainKernel &kernel, const FitOptions &options, FitAudit &a)
      : k(kernel), o(options), audit(a) {
    traceK = arma::accu(k.diagonal);
    for (auto &key : k.ids) {
      uint64_t h = 14695981039346656037ULL;
      for (unsigned char c : key)
        h = (h ^ c) * 1099511628211ULL;
      keys.push_back(h);
    }
    require(keys.size() == k.diagonal.n_elem, "missing stochastic probe IDs");
  }
  vec probe(size_t index) const {
    vec z(keys.size());
    auto tag = mix(uint64_t(o.seed) ^ mix(index));
    for (size_t i = 0; i < keys.size(); ++i)
      z[i] = (mix(keys[i] ^ tag) & 1) ? 1 : -1;
    return z;
  }
  Pair build(const vec &z, const vec *orthogonal_to = nullptr) {
    size_t depth =
        std::min<size_t>(z.n_elem - (orthogonal_to ? 1 : 0), 2 * o.slq_depth);
    bool full = o.full_reorth;
    mat basis;
    if (full)
      basis.set_size(z.n_elem, depth);
    double spectral_scale = 1;
    vec q = z / arma::norm(z), prev(z.n_elem, arma::fill::zeros), d(depth),
        b(depth);
    size_t used = 0;
    for (size_t j = 0; j < depth; ++j) {
      if (full)
        basis.col(j) = q;
      vec source = q;
      if (orthogonal_to)
        source -= *orthogonal_to * arma::dot(*orthogonal_to, source);
      vec v = k.apply(source);
      ++audit.slq_matvec;
      ++audit.matvec;
      if (orthogonal_to)
        v -= *orthogonal_to * arma::dot(*orthogonal_to, v);
      if (j)
        v -= b[j - 1] * prev;
      d[j] = arma::dot(q, v);
      v -= d[j] * q;
      if (full) {
        auto columns = basis.head_cols(j + 1);
        for (int pass = 0; pass < 2; ++pass)
          v -= columns * (columns.t() * v);
      } else {
        v -= q * arma::dot(q, v);
        if (j)
          v -= prev * arma::dot(prev, v);
      }
      if (orthogonal_to)
        v -= *orthogonal_to * arma::dot(*orthogonal_to, v);
      b[j] = arma::norm(v);
      used = j + 1;
      spectral_scale = std::max(spectral_scale, std::abs(d[j]) + b[j]);
      if (b[j] <= std::sqrt(arma::datum::eps) * spectral_scale || used == depth)
        break;
      prev = std::move(q);
      q = v / b[j];
    }
    auto rule = [&](size_t m) {
      mat t = arma::diagmat(d.head(m));
      if (m > 1) {
        t.diag(1) = b.head(m - 1);
        t.diag(-1) = b.head(m - 1);
      }
      Rule r;
      mat v;
      require(arma::eig_sym(r.nodes, v, t),
              "SLQ tridiagonal eigensolve failed");
      if (r.nodes.min() < -1e-10 * std::max(1., r.nodes.max()))
        throw std::runtime_error(
            "SLQ negative Ritz spectrum min=" + std::to_string(r.nodes.min()) +
            " max=" + std::to_string(r.nodes.max()) +
            " depth=" + std::to_string(m) +
            " contrast=" + std::to_string(orthogonal_to != nullptr));
      r.nodes = arma::clamp(r.nodes, 0., arma::datum::inf);
      r.weights = arma::dot(z, z) * arma::square(v.row(0).t());
      return r;
    };
    return {rule(std::min(used, o.slq_depth)), rule(used)};
  }
  void ensure(size_t count) {
    while (rules.size() < count)
      rules.push_back(build(probe(rules.size())));
  }
};
// Independent Jacobi CG per column; one operator application decodes a feature
// block for every currently active probe. Always verify the final true
// residual.
mat solve_probes(System &s, const mat &b) {
  size_t n = b.n_rows, m = b.n_cols;
  s.audit.rhs += m;
  mat x(n, m, arma::fill::zeros), r = b, z = r;
  z = s.precondition(r);
  mat p = z;
  arma::rowvec rho = arma::sum(r % z);
  uvec active = arma::regspace<uvec>(0, m - 1);
  double tol = s.opt.atol + s.opt.rtol * std::sqrt(double(n));
  for (int it = 1; it <= s.opt.max_pcg; ++it) {
    mat ap = s.g * s.k.apply(p.cols(active)) + s.e * p.cols(active);
    s.audit.matvec += active.n_elem;
    std::vector<arma::uword> next;
    for (size_t j = 0; j < active.n_elem; ++j) {
      auto c = active[j];
      double den = arma::dot(p.col(c), ap.col(j));
      require(den > 0 && std::isfinite(den) && rho[c] > 0,
              "probe PCG nonpositive curvature");
      double step = rho[c] / den;
      x.col(c) += step * p.col(c);
      r.col(c) -= step * ap.col(j);
      if (arma::norm(r.col(c)) <= tol) {
        vec real = b.col(c) - s.apply(x.col(c));
        double err = arma::norm(real);
        if (err <= tol) {
          s.audit.residual_ratio = std::max(s.audit.residual_ratio, err / tol);
          s.audit.max_iterations = std::max(s.audit.max_iterations, size_t(it));
          continue;
        }
        r.col(c) = real;
        z.col(c) = s.precondition(real);
        p.col(c) = z.col(c);
        rho[c] = arma::dot(real, z.col(c));
      } else {
        z.col(c) = s.precondition(r.col(c));
        double nr = arma::dot(r.col(c), z.col(c));
        p.col(c) = z.col(c) + (nr / rho[c]) * p.col(c);
        rho[c] = nr;
      }
      next.push_back(c);
    }
    if (next.empty())
      return x;
    active = uvec(next);
  }
  throw std::runtime_error("probe PCG iteration-limit");
}
struct State {
  vec score, lik;
  mat information, samples;
  double depth;
};
State evaluate(ProbeSet &probes, const mat &x, const vec &y, double g, double e,
               size_t count, bool score, bool verify = false) {
  probes.ensure(count);
  auto &audit = probes.audit;
  ++audit.trials;
  State s;
  s.lik.set_size(count);
  s.depth = 0;
  double offset = 0;
  vec quadratic(2), rank_correction(2);
  std::optional<System> sys;
  if (probes.o.probe_lanczos && !verify && x.n_cols == 1) {
    // The Krylov space of K (or Q2'KQ2) is shift-invariant. Store scalar
    // quadrature measures for this design and response only, never across
    // folds.
    if (!probes.design_rule) {
      vec unit = x.col(0) / arma::norm(x.col(0));
      probes.design_rule = probes.build(x.col(0));
      probes.response_rule = probes.build(y - unit * arma::dot(unit, y), &unit);
    }
    auto data = [&](const Rule &xr, const Rule &yr) {
      vec xi = 1 / (g * xr.nodes + e), yi = 1 / (g * yr.nodes + e);
      return std::log(arma::dot(xr.weights, xi)) + arma::dot(yr.weights, yi);
    };
    const auto &xr = probes.design_rule->second,
               &yr = probes.response_rule->second;
    offset = data(xr, yr);
    s.depth = std::abs(offset - data(probes.design_rule->first,
                                     probes.response_rule->first)) /
              x.n_rows;
    if (score) {
      vec xi = 1 / (g * xr.nodes + e), yi = 1 / (g * yr.nodes + e);
      vec ww = xr.weights % arma::square(xi) / arma::dot(xr.weights, xi),
          rr = yr.weights % arma::square(yi);
      rank_correction = {arma::dot(ww, xr.nodes), arma::accu(ww)};
      quadratic = {arma::dot(rr, yr.nodes), arma::accu(rr)};
      if (probes.o.varcomp == "ai-reml") {
        vec power = rr % yi;
        double off = arma::dot(power, yr.nodes);
        s.information =
            .5 * mat{{arma::dot(power, arma::square(yr.nodes)), off},
                     {off, arma::accu(power)}};
      }
    }
  } else {
    sys.emplace(probes.k, g, e, probes.o, audit);
    Projection p(*sys, x);
    vec b, r = p.project(sys->solve_one(y, "y"), b);
    offset = 2 * arma::accu(arma::log(p.lower.diag())) + arma::dot(y, r);
    if (score) {
      vec kr = probes.k.apply(r);
      ++audit.matvec;
      mat dr = arma::join_rows(kr, r);
      quadratic = dr.t() * r;
      if (probes.o.varcomp == "ai-reml") {
        vec pg = p.apply_P(kr), pe = p.apply_P(r);
        s.information = .5 * dr.t() * arma::join_rows(pg, pe);
        s.information = .5 * s.information + .5 * s.information.t();
      }
      mat kw = probes.k.apply(p.w);
      audit.matvec += p.w.n_cols;
      mat invwx = arma::solve(arma::trimatl(p.lower), p.w.t()),
          invkw = arma::solve(arma::trimatl(p.lower), kw.t());
      rank_correction = {arma::accu(invwx % invkw),
                         arma::accu(arma::square(invwx))};
    }
  }
  const bool control = x.n_rows <= 512;
  double delta = 0;
  double km = probes.traceK / x.n_rows, anchor = g * km + e;
  for (size_t j = 0; j < count; ++j) {
    auto &q = probes.rules[j];
    double fine = q.second.logdet(g, e);
    delta += fine - q.first.logdet(g, e);
    double centered = q.second.first() - probes.traceK;
    s.lik[j] = -.5 * (fine - (control ? g / anchor * centered : 0) + offset);
  }
  s.depth = std::max(s.depth, std::abs(delta) / (count * x.n_rows));
  audit.depth_error = std::max(audit.depth_error, s.depth);
  if (score)
    require(
        s.depth <= probes.o.slq_tol,
        "stochastic precision insufficient: SLQ depth; increase --slq-depth");
  if (!score)
    return s;
  s.samples.set_size(2, count);
  if (probes.o.probe_lanczos && !verify) {
    for (size_t j = 0; j < count; ++j) {
      const auto &q = probes.rules[j].second;
      vec inv = 1 / (g * q.nodes + e);
      double cv = control ? q.first() - probes.traceK : 0;
      s.samples(0, j) = arma::dot(q.weights, q.nodes % inv) -
                        rank_correction[0] - e * cv / (anchor * anchor);
      s.samples(1, j) = arma::dot(q.weights, inv) - rank_correction[1] +
                        g * cv / (anchor * anchor);
    }
  } else
    for (size_t i = 0; i < count; i += probes.o.batch) {
      size_t width = std::min(count - i, probes.o.batch);
      mat z(x.n_rows, width);
      for (size_t j = 0; j < width; ++j)
        z.col(j) = probes.probe(i + j);
      mat h = solve_probes(*sys, z), kz = probes.k.apply(z);
      audit.matvec += width;
      for (size_t j = 0; j < width; ++j) {
        // z' V^-1 D_i z minus tr(C^-1 W' D_i W) is Hutchinson on PD_i
        // with the stochastic fixed-effect term replaced by its exact mean.
        double cv =
            control ? probes.rules[i + j].second.first() - probes.traceK : 0;
        s.samples(0, i + j) = arma::dot(h.col(j), kz.col(j)) -
                              rank_correction[0] - e * cv / (anchor * anchor);
        s.samples(1, i + j) = arma::dot(h.col(j), z.col(j)) -
                              rank_correction[1] + g * cv / (anchor * anchor);
      }
    }
  vec traces = arma::mean(s.samples, 1);
  audit.trace_identity_error =
      std::abs(g * traces[0] + e * traces[1] - (x.n_rows - x.n_cols));
  audit.logdet_se = 2 * arma::stddev(s.lik) / std::sqrt(double(count));
  s.score = .5 * (quadratic - traces);
  vec se = arma::stddev(s.samples, 0, 1) / std::sqrt(double(count));
  audit.trace_se_g = se[0];
  audit.trace_se_e = se[1];
  return s;
}
} // namespace
Estimates estimate_stochastic(const TrainKernel &k, const mat &x, const vec &y,
                              double g, double e, const FitOptions &o,
                              FitAudit &a) {
  ProbeSet probes(k, o, a);
  size_t count = o.probes, df = x.n_rows - x.n_cols;
  vec ols = arma::solve(x, y);
  double scale = arma::dot(y - x * ols, y - x * ols) / df;
  double km = arma::mean(k.diagonal), floor = scale * 1e-9;
  vec units = {scale / km, scale};
  double design_logdet;
  double sign;
  arma::log_det(design_logdet, sign, x.t() * x);
  auto more = [&] {
    require(count < o.max_probes,
            "stochastic precision insufficient: probe budget exhausted");
    count = std::min(2 * count, o.max_probes);
    a.logratio_se = 0;
  };
  int it = 0;
  while (it <= o.max_reml) {
    a.probes = count;
    auto cur = evaluate(probes, x, y, g, e, count, true);
    double lik = arma::mean(cur.lik) + .5 * design_logdet;
    vec scaled = cur.score % units / df;
    if (o.event && it % o.log_every == 0)
      o.event(a.context + " reml=" + std::to_string(it) +
              " probes=" + std::to_string(count));
    bool boundary_g = g == 0, boundary_e = e <= floor;
    vec se = .5 * vec{a.trace_se_g, a.trace_se_e} % units / df / 1.;
    bool converged = true;
    for (size_t j = 0; j < 2; ++j) {
      if ((j == 0 && boundary_g) || (j == 1 && boundary_e)) {
        converged &= scaled[j] + 1.96 * se[j] <= o.score_tol;
      } else
        converged &= std::abs(scaled[j]) <= o.score_tol;
    }
    double condition = 0;
    vec delta;
    if (o.varcomp == "ai-reml") {
      condition = arma::rcond(cur.information);
      require(condition > 1e-14,
              "stochastic AI unidentifiable or ill-conditioned");
      delta = arma::solve(cur.information, cur.score);
      if (g > 0 && e > floor) {
        vec grad = arma::solve(cur.information, vec{-1 / g, 1 / e});
        a.logratio_se = .5 *
                        arma::stddev(arma::conv_to<vec>::from(
                            (grad.t() * cur.samples).t())) /
                        std::sqrt(double(count));
      }
      if (g + delta[0] < 0 || e + delta[1] < floor) {
        vec dg = {
            -g, std::max(floor - e, (cur.score[1] + cur.information(1, 0) * g) /
                                        cur.information(1, 1))};
        vec de = {
            std::max(-g, (cur.score[0] - cur.information(0, 1) * (floor - e)) /
                             cur.information(0, 0)),
            floor - e};
        auto gain = [&](const vec &v) {
          return arma::dot(cur.score, v) -
                 .5 * arma::dot(v, cur.information * v);
        };
        delta = gain(dg) > gain(de) ? dg : de;
      }
    } else {
      delta = 2 * arma::square(vec{g, e}) % cur.score / x.n_rows;
      if (g + delta[0] < 0 || e + delta[1] <= 0) {
        more();
        continue;
      }
    }
    if (converged && !boundary_g && !boundary_e) {
      // Estimate the sensitivity of log(e/g) to Monte Carlo score noise using
      // the observed curvature in log-variance coordinates. EM needs no P(D_i
      // r).
      const double h = .001;
      auto f = [&](double u, double v) {
        auto s = evaluate(probes, x, y, g * std::exp(u), e * std::exp(v), count,
                          false);
        return arma::mean(s.lik);
      };
      double f0 = arma::mean(cur.lik);
      mat H(2, 2);
      H(0, 0) = (2 * f0 - f(h, 0) - f(-h, 0)) / (h * h);
      H(1, 1) = (2 * f0 - f(0, h) - f(0, -h)) / (h * h);
      H(1, 0) = H(0, 1) =
          -(f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h);
      mat L;
      require(arma::chol(L, H),
              "stochastic precision insufficient: log-variance curvature");
      vec grad = arma::solve(H, vec{-1., 1.});
      grad %= vec{g, e};
      a.logratio_se =
          .5 *
          arma::stddev(arma::conv_to<vec>::from((grad.t() * cur.samples).t())) /
          std::sqrt(double(count));
    }
    if (converged) {
      if (1.96 * se.max() > o.mc_tol ||
          (!boundary_g && !boundary_e && 1.96 * a.logratio_se > o.mc_tol)) {
        more();
        continue;
      }
      if (o.probe_lanczos) {
        auto certified = evaluate(probes, x, y, g, e, count, true, true);
        require(arma::abs((cur.score - certified.score) % units).max() / df <=
                    o.score_tol,
                "stochastic precision insufficient: SLQ derivative vs PCG; "
                "increase --slq-depth");
      }
      string status = boundary_g   ? "approx-genetic-zero"
                      : boundary_e ? "approx-residual-floor"
                                   : "approx-interior";
      a.steps.push_back(
          {it, 0, g, e, lik, cur.score[0], cur.score[1], 0, condition, status});
      return {g, e, lik, status};
    }
    require(it < o.max_reml, "stochastic REML iteration-limit");
    double step = 1;
    bool resample = false;
    int bt = 0;
    for (; bt < 40; ++bt, step *= .5) {
      double ng = std::max(0., g + step * delta[0]),
             ne = std::max(floor, e + step * delta[1]);
      if (!std::isfinite(ng + ne))
        continue;
      auto trial = evaluate(probes, x, y, ng, ne, count, false);
      if (trial.depth > o.slq_tol)
        continue;
      vec diff = trial.lik - cur.lik;
      double gain = arma::mean(diff);
      a.paired_se = arma::stddev(diff) / std::sqrt(double(count));
      if (gain >= -1e-12 * (1 + std::abs(lik))) {
        if (gain < 1.96 * a.paired_se &&
            (1.96 * se.max() > o.mc_tol ||
             (g > 0 && e > floor && 1.96 * a.logratio_se > o.mc_tol))) {
          more();
          resample = true;
          break;
        }
        if (it % o.log_every == 0)
          a.steps.push_back({it, bt, g, e, lik, cur.score[0], cur.score[1],
                             step, condition, "approx-update"});
        g = ng;
        e = ne;
        ++it;
        break;
      }
    }
    if (resample)
      continue;
    require(bt < 40, "stochastic backtracking failed");
  }
  throw std::runtime_error("stochastic REML iteration-limit");
}
} // namespace fsb
