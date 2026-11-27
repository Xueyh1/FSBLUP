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
  double traceK, tail_mean = 0;
  std::unique_ptr<System> system;
  std::unique_ptr<Projection> projection;
  vec response, quad, rank_term;
  mat info, solved_samples;
  double offset_term = 0;
  mat directions;
  vec anchors;
  std::vector<vec> energy_deviation;
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
    const size_t rank = std::min(keys.size() - 1, size_t(o.trace_rank));
    if (rank) {
      mat sketch(keys.size(), rank);
      // Independent pilot domain. Conditional on this basis, each subsequent
      // Rademacher quadratic minus the known trace has expectation zero.
      for (size_t j = 0; j < rank; ++j)
        sketch.col(j) = probe((uint64_t(1) << 40) + j);
      mat q, r;
      require(arma::qr_econ(q, r, k.apply(sketch)), "trace control QR failed");
      mat small = q.t() * k.apply(q), rot;
      audit.matvec += 2 * rank;
      require(arma::eig_sym(anchors, rot, .5 * small + .5 * small.t()),
              "trace control eigensolve failed");
      anchors = arma::clamp(anchors, 0., arma::datum::inf);
      directions = q * rot;
      tail_mean =
          std::max(0., (traceK - arma::accu(anchors)) / (keys.size() - rank));
    }
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
    return quadrature(d, b, used, arma::dot(z, z), orthogonal_to != nullptr);
  }
  Pair quadrature(const vec &d, const vec &b, size_t used, double norm_squared,
                  bool contrast) {
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
            " max=" + std::to_string(r.nodes.max()) + " depth=" +
            std::to_string(m) + " contrast=" + std::to_string(contrast));
      r.nodes = arma::clamp(r.nodes, 0., arma::datum::inf);
      r.weights = norm_squared * arma::square(v.row(0).t());
      return r;
    };
    return {rule(std::min(used, o.slq_depth)), rule(used)};
  }
  void build_panel(size_t width) {
    // Separate scalar Lanczos recurrences share a streamed matmat, not a
    // block-Krylov space. Deflate each column independently on breakdown.
    const size_t n = keys.size(), depth = std::min(n, 2 * o.slq_depth);
    mat q(n, width), prev(n, width, arma::fill::zeros), d(depth, width),
        b(depth, width);
    uvec active = arma::regspace<uvec>(0, width - 1), length(width);
    vec scale(width, arma::fill::ones);
    for (size_t c = 0; c < width; ++c) {
      auto z = probe(rules.size() + c);
      if (!anchors.empty())
        energy_deviation.push_back(arma::square(directions.t() * z) - 1);
      q.col(c) = z / std::sqrt(double(n));
    }
    for (size_t j = 0; j < depth && !active.empty(); ++j) {
      mat v = k.apply(q.cols(active));
      audit.matvec += active.n_elem;
      audit.slq_matvec += active.n_elem;
      std::vector<arma::uword> next;
      for (size_t h = 0; h < active.n_elem; ++h) {
        auto c = active[h];
        auto col = v.col(h);
        if (j)
          col -= b(j - 1, c) * prev.col(c);
        d(j, c) = arma::dot(q.col(c), col);
        col -= d(j, c) * q.col(c);
        col -= arma::dot(q.col(c), col) * q.col(c);
        if (j)
          col -= arma::dot(prev.col(c), col) * prev.col(c);
        b(j, c) = arma::norm(col);
        length[c] = j + 1;
        scale[c] = std::max(scale[c], std::abs(d(j, c)) + b(j, c));
        if (j + 1 < depth && b(j, c) > std::sqrt(arma::datum::eps) * scale[c]) {
          prev.col(c) = q.col(c);
          q.col(c) = col / b(j, c);
          next.push_back(c);
        }
      }
      active = uvec(next);
    }
    for (size_t c = 0; c < width; ++c)
      rules.push_back(quadrature(d.col(c), b.col(c), length[c], n, false));
  }
  void ensure(size_t count) {
    while (rules.size() < count) {
      if (o.slq_batch > 1) {
        build_panel(std::min(count - rules.size(), o.slq_batch));
        continue;
      }
      vec z = probe(rules.size());
      rules.push_back(build(z));
      if (!anchors.empty())
        energy_deviation.push_back(arma::square(directions.t() * z) - 1);
    }
  }
  mat controls(double g, double e, size_t count) const {
    mat out(3, count, arma::fill::zeros);
    if (anchors.empty() && keys.size() > 512)
      return out;
    double mean = traceK / keys.size(), den = g * mean + e;
    mat f(3, anchors.n_elem);
    if (!anchors.empty()) {
      vec d = g * anchors + e;
      double tail = g * tail_mean + e;
      f.row(0) = (arma::log(d) - std::log(tail)).t();
      f.row(1) = (anchors / d - tail_mean / tail).t();
      f.row(2) = (1 / d - 1 / tail).t();
    }
    for (size_t j = 0; j < count; ++j) {
      double centered = rules[j].second.first() - traceK;
      if (!anchors.empty()) {
        out.col(j) = f * energy_deviation[j];
        centered -= arma::dot(anchors - tail_mean, energy_deviation[j]);
      }
      out(0, j) += g / den * centered;
      out(1, j) += e / (den * den) * centered;
      out(2, j) -= g / (den * den) * centered;
    }
    return out;
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
  double depth, ypy;
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
  System *sys = nullptr;
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
    s.ypy = arma::accu(yr.weights / (g * yr.nodes + e));
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
    if (!probes.system || probes.system->g != g || probes.system->e != e) {
      probes.projection.reset();
      probes.system = std::make_unique<System>(probes.k, g, e, probes.o, audit);
      probes.projection = std::make_unique<Projection>(*probes.system, x);
      probes.response = probes.projection->fit_response(y, "y").r;
      probes.offset_term =
          2 * arma::accu(arma::log(probes.projection->lower.diag())) +
          arma::dot(y, probes.response);
      probes.quad.reset();
      probes.solved_samples.reset();
    }
    sys = probes.system.get();
    auto &p = *probes.projection;
    const vec &r = probes.response;
    offset = probes.offset_term;
    s.ypy = arma::dot(y, probes.response);
    if (score && !probes.quad.empty()) {
      quadratic = probes.quad;
      rank_correction = probes.rank_term;
      s.information = probes.info;
    } else if (score) {
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
      mat invwx = arma::solve(arma::trimatl(p.lower), p.w.t(),
                              arma::solve_opts::no_approx),
          invkw = arma::solve(arma::trimatl(p.lower), kw.t(),
                              arma::solve_opts::no_approx);
      rank_correction = {arma::accu(invwx % invkw),
                         arma::accu(arma::square(invwx))};
      probes.quad = quadratic;
      probes.rank_term = rank_correction;
      probes.info = s.information;
    }
  }
  mat controls = probes.controls(g, e, count);
  double delta = 0;

  for (size_t j = 0; j < count; ++j) {
    auto &q = probes.rules[j];
    double fine = q.second.logdet(g, e);
    delta += fine - q.first.logdet(g, e);

    s.lik[j] = -.5 * (fine - controls(0, j) + offset);
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

      s.samples(0, j) = arma::dot(q.weights, q.nodes % inv) -
                        rank_correction[0] - controls(1, j);
      s.samples(1, j) =
          arma::dot(q.weights, inv) - rank_correction[1] - controls(2, j);
    }
  } else {
    size_t saved = std::min(count, size_t(probes.solved_samples.n_cols));
    if (saved)
      s.samples.head_cols(saved) = probes.solved_samples.head_cols(saved);
    for (size_t i = saved; i < count; i += probes.o.batch) {
      size_t width = std::min(count - i, probes.o.batch);
      mat z(x.n_rows, width);
      for (size_t j = 0; j < width; ++j)
        z.col(j) = probes.probe(i + j);
      mat h = solve_probes(*sys, z), kz = probes.k.apply(z);
      audit.matvec += width;
      for (size_t j = 0; j < width; ++j) {
        // z' V^-1 D_i z minus tr(C^-1 W' D_i W) is Hutchinson on PD_i
        // with the stochastic fixed-effect term replaced by its exact mean.

        s.samples(0, i + j) = arma::dot(h.col(j), kz.col(j)) -
                              rank_correction[0] - controls(1, i + j);
        s.samples(1, i + j) = arma::dot(h.col(j), z.col(j)) -
                              rank_correction[1] - controls(2, i + j);
      }
    }
    probes.solved_samples = s.samples;
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
// Profile the common covariance scale analytically at each variance ratio.
// The same ordered probes supply both likelihoods, including control variates.
struct FaceCheck {
  double g = 0, e = 0, gain = -INFINITY;
  bool unresolved = false;
};
FaceCheck compare_face(ProbeSet &p, const mat &x, const vec &y,
                       const vec &face_lik, double scale, double km,
                       size_t count, int it, double design_logdet) {
  FaceCheck best;
  double df = x.n_rows - x.n_cols;
  for (int j = -8; j <= 8; ++j) {
    double ratio = std::exp2(j * 1.5) * km;
    double g = scale / (km + ratio), e = ratio * g;
    auto before = p.audit.matvec;
    auto state = evaluate(p, x, y, g, e, count, false);
    double factor = state.ypy / df;
    require(factor > 0 && std::isfinite(factor),
            "invalid profiled stochastic scale");
    g *= factor;
    e *= factor;
    vec lik = state.lik +
              .5 * (design_logdet + state.ypy - df - df * std::log(factor));
    vec diff = lik - face_lik;
    double gain = arma::mean(diff),
           se = arma::stddev(diff) / std::sqrt(double(count));
    double error = 3.5 * se + state.depth * x.n_rows +
                   64 * arma::datum::eps * (1 + std::abs(arma::mean(lik)));
    auto &step = p.audit.steps.emplace_back(
        Iteration{it, 0, g, e, arma::mean(lik), NAN, NAN, 0, 0,
                  "face-challenger", count, se, state.depth});
    step.face_gain = gain;
    step.face_matvec = p.audit.matvec - before;
    if (gain > error && state.depth <= p.o.slq_tol && gain > best.gain) {
      best.g = g;
      best.e = e;
      best.gain = gain;
    }
    if (gain + error >= 0)
      best.unresolved = true;
  }
  return best;
}
} // namespace
Estimates estimate_stochastic(const TrainKernel &k, const mat &x, const vec &y,
                              double g, double e, const FitOptions &o,
                              FitAudit &a) {
  ProbeSet probes(k, o, a);
  size_t count = o.probes, df = x.n_rows - x.n_cols;
  vec ols = arma::solve(x, y, arma::solve_opts::no_approx);
  double scale = arma::dot(y - x * ols, y - x * ols) / df;
  double km = arma::mean(k.diagonal), floor = scale * 1e-9;
  vec units = {scale / km, scale};
  double design_logdet;
  double sign;
  arma::log_det(design_logdet, sign, x.t() * x);
  // At g=0 the inward score has an exact linear-time trace even in
  // stochastic mode. Avoid a multiplicative EM approach to zero forever.
  mat qx, rx;
  require(arma::qr_econ(qx, rx, x), "fixed design QR failed");
  vec residual = y - x * ols;
  mat kq = k.apply(qx);
  vec ky = k.apply(residual);
  a.matvec += qx.n_cols + 1;
  double zero_score = .5 * (arma::dot(residual, ky) / (scale * scale) -
                            (probes.traceK - arma::accu(qx % kq)) / scale);
  double zero_lik = -.5 * df * (std::log(scale) + 1);
  bool restarted = false;
  int it = 0;
  while (it <= o.max_reml) {
    a.updates = it;
    a.probes = count;
    auto cur = evaluate(probes, x, y, g, e, count, true);
    double lik = arma::mean(cur.lik) + .5 * design_logdet;
    vec scaled = cur.score % units / df;
    if (o.event && it % o.log_every == 0)
      o.event(a.context + " reml=" + std::to_string(it) +
              " probes=" + std::to_string(count));
    int info_rank =
        o.varcomp == "ai-reml" ? int(arma::rank(cur.information)) : -1;
    bool boundary_g = g == 0, boundary_e = e <= floor;
    vec se = .5 * vec{a.trace_se_g, a.trace_se_e} % units / df / 1.;
    const bool zero_kkt = zero_score * units[0] / df < -o.score_tol;
    if (zero_kkt && cur.score[0] < 0 && zero_lik >= lik && !restarted) {
      vec face_lik(count, arma::fill::value(zero_lik));
      auto best = compare_face(probes, x, y, face_lik, scale, km, count, it,
                               design_logdet);
      if (std::isfinite(best.gain)) {
        a.steps.push_back({it, 0, best.g, best.e, zero_lik + best.gain, NAN,
                           NAN, 0, 0, "interior-restart", count});
        g = best.g;
        e = best.e;
        restarted = true;
        continue;
      }
      if (best.unresolved) {
        require(count < o.max_probes,
                "stochastic precision insufficient: probe budget exhausted "
                "comparing boundary");
        count = std::min(2 * count, o.max_probes);
        continue;
      }
      a.steps.push_back({it, 0, 0, scale, zero_lik, zero_score, 0, 0, 0,
                         "genetic-zero", count});
      return {0, scale, zero_lik, "genetic-zero"};
    }
    bool converged = true;
    for (size_t j = 0; j < 2; ++j) {
      if ((j == 0 && boundary_g) || (j == 1 && boundary_e)) {
        converged &= scaled[j] + 1.96 * se[j] <= o.score_tol;
      } else
        converged &= std::abs(scaled[j]) <= o.score_tol;
    }
    auto more = [&] {
      a.steps.push_back({it, 0, g, e, lik, cur.score[0], cur.score[1], 0, 0,
                         "increase-probes", count, a.paired_se, cur.depth,
                         info_rank});
      require(count < o.max_probes,
              "stochastic precision insufficient: probe budget exhausted");
      count = std::min(2 * count, o.max_probes);
      a.logratio_se = 0;
    };
    double condition = 0;
    vec delta;
    if (o.varcomp == "ai-reml") {
      condition = arma::rcond(cur.information);
      if (condition <= 1e-14)
        throw std::runtime_error(
            "stochastic AI unidentifiable or ill-conditioned, rank=" +
            std::to_string(info_rank));
      delta =
          arma::solve(cur.information, cur.score, arma::solve_opts::no_approx);
      if (g > 0 && e > floor) {
        vec grad = arma::solve(cur.information, vec{-1 / g, 1 / e},
                               arma::solve_opts::no_approx);
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
      vec grad = arma::solve(H, vec{-1., 1.}, arma::solve_opts::no_approx);
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
      if (boundary_g || boundary_e) {
        auto best = compare_face(probes, x, y, cur.lik + .5 * design_logdet,
                                 scale, km, count, it, design_logdet);
        if (std::isfinite(best.gain) && !restarted) {
          g = best.g;
          e = best.e;
          restarted = true;
          continue;
        }
        if (best.unresolved) {
          more();
          continue;
        }
      }
      string status = boundary_g   ? "approx-genetic-zero"
                      : boundary_e ? "approx-residual-floor"
                                   : "approx-interior";
      a.steps.push_back({it, 0, g, e, lik, cur.score[0], cur.score[1], 0,
                         condition, status, count, a.paired_se, cur.depth,
                         info_rank});
      return {g, e, lik, status};
    }
    require(it < o.max_reml, "stochastic REML iteration-limit");
    double step = 1;
    bool resample = false;
    int bt = 0;
    for (; bt <= o.max_backtracks; ++bt, step *= .5) {
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
        if (it % o.record_every == 0)
          a.steps.push_back({it, bt, g, e, lik, cur.score[0], cur.score[1],
                             step, condition,
                             bt ? "approx-damped-update" : "approx-update",
                             count, a.paired_se, cur.depth, info_rank});
        g = ng;
        e = ne;
        ++it;
        break;
      }
    }
    if (resample)
      continue;
    require(bt <= o.max_backtracks, "stochastic backtracking failed");
  }
  throw std::runtime_error("stochastic REML iteration-limit");
}
} // namespace fsb
