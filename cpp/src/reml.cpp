#include "core.hpp"
namespace fsb {
// In orthonormal error contrasts P and both covariance derivatives are
// diagonal. This also evaluates (D_i r)' P (D_j r) without an extra
// training-system solve.
Derivatives reml_derivatives(const Contrast &c, double g, double e) {
  vec d = g * c.eigen + e;
  require(d.min() > 0 && d.is_finite(), "nonpositive contrast variance");
  vec inv = 1 / d, t = c.z % inv, common = arma::square(t) - inv;
  vec energy = arma::square(t) % inv;
  double off = arma::dot(energy, c.eigen);
  return {-.5 * (arma::accu(arma::log(d)) + arma::dot(c.z, t)),
          .5 * vec{arma::dot(c.eigen, common), arma::accu(common)},
          .5 * mat{{arma::dot(energy, arma::square(c.eigen)), off},
                   {off, arma::accu(energy)}}};
}
Estimates estimate_exact(const Contrast &c, size_t n, double g, double e,
                         const FitOptions &o, FitAudit &audit) {
  auto evaluate = [&](double a, double b) {
    ++audit.trials;
    return reml_derivatives(c, a, b);
  };
  size_t df = c.z.n_elem;
  double scale = arma::mean(arma::square(c.z)), km = arma::mean(c.eigen);
  require(scale > 0 && std::isfinite(scale),
          "constant response or exact fixed-effect fit");
  if (c.eigen.max() - c.eigen.min() <= 1e-12 * std::max(1., c.eigen.max())) {
    g = e = scale / (1 + km);
    auto d = evaluate(g, e);
    audit.steps.push_back(
        {0, 0, g, e, d.lik, d.score[0], d.score[1], 0, 0, "unidentifiable"});
    return {g, e, d.lik, "unidentifiable;lambda=1"};
  }
  const vec units = {scale / std::max(km, 1e-300), scale};
  // Optimize the scale analytically on each feasible face and check its inward
  // score.
  std::vector<Estimates> boundary;
  auto face = [&](double a, double b, int inward, std::string name) {
    auto d = evaluate(a, b);
    if (d.score[inward] * units[inward] / df <= o.score_tol)
      boundary.push_back({a, b, d.lik, name});
  };
  face(0, scale, 0, "genetic-zero");
  if (c.eigen.min() > 0)
    face(arma::mean(arma::square(c.z) / c.eigen), 0, 1, "residual-zero");
  auto best_face = [&]() {
    return *std::max_element(boundary.begin(), boundary.end(),
                             [](auto &a, auto &b) { return a.lik < b.lik; });
  };
  // Check credible interior alternatives before an active-set boundary
  // shortcut.
  const vec energy = arma::square(c.z);
  auto profile = [&](double log_ratio) {
    double ratio = std::exp(log_ratio);
    vec denom = c.eigen + ratio;
    double a = arma::mean(energy / denom);
    ++audit.trials;
    double lik = -.5 * (df * (1 + std::log(a)) + arma::accu(arma::log(denom)));
    return Estimates{a, a * ratio, lik, "profile"};
  };
  // Inspect the profiled likelihood on the existing contrast spectrum.
  // Adjacent grid slopes bracket competing maxima; refine those intervals
  // without constructing another kernel or altering the EM update rule.
  std::vector<Estimates> peaks;
  double interior_bound = -INFINITY;
  double pivot = std::log(km);
  std::vector<double> grid;
  for (int j = -72; j <= 72; ++j) {
    double log_ratio = pivot + .5 * j;
    if (c.eigen.min() + std::exp(log_ratio) <= 0)
      continue;
    grid.push_back(log_ratio);
  }
  std::vector<double> values;
  for (double z : grid) {
    values.push_back(profile(z).lik);
    interior_bound = std::max(interior_bound, values.back());
  }
  for (size_t j = 1; j + 1 < grid.size(); ++j) {
    if (!(values[j] > values[j - 1] + 1e-12 &&
          values[j] > values[j + 1] + 1e-12))
      continue;
    double lo = grid[j - 1], hi = grid[j + 1];
    for (int h = 0; h < 26; ++h) {
      double left = (2 * lo + hi) / 3, right = (lo + 2 * hi) / 3;
      if (profile(left).lik < profile(right).lik)
        lo = left;
      else
        hi = right;
    }
    peaks.push_back(profile((lo + hi) / 2));
    interior_bound = std::max(interior_bound, peaks.back().lik);
  }
  double q = arma::mean(energy / (g * c.eigen + e));
  if (q < .01 || q > 100) {
    g *= q;
    e *= q;
  }
  if (peaks.size() > 1) {
    auto best = std::max_element(
        peaks.begin(), peaks.end(),
        [](const auto &a, const auto &b) { return a.lik < b.lik; });
    for (const auto &v : peaks)
      audit.steps.push_back(
          {0, 0, v.g, v.e, v.lik, 0, 0, 0, 0, "profile-peak"});
    g = best->g;
    e = best->e;
  }
  audit.initial_vu = std::scalbn(g, 2 * audit.scale_exponent);
  audit.initial_ve = std::scalbn(e, 2 * audit.scale_exponent);
  double previous = -INFINITY;
  for (int it = 0; it <= o.max_reml; ++it) {
    audit.updates = it;
    auto d = evaluate(g, e);
    vec s = d.score % units;
    double scaled = arma::abs(s).max() / df;
    auto record = [&](std::string status, double step, int bt,
                      double condition) {
      if (it == 0 || it % o.record_every == 0 || status != "update") {
        audit.steps.push_back({it, bt, g, e, d.lik, d.score[0], d.score[1],
                               step, condition, status});
        if (o.varcomp == "ai-reml")
          audit.steps.back().information_rank = arma::rank(d.information);
      }
      if (o.event && (it % o.log_every == 0 || status != "update"))
        o.event(audit.context + " reml=" + std::to_string(it) +
                " status=" + status + " score=" + std::to_string(scaled));
    };
    auto accept_face = [&]() {
      audit.updates = it + 1;
      auto b = best_face();
      auto f = evaluate(b.g, b.e);
      audit.steps.push_back(
          {it + 1, 0, b.g, b.e, b.lik, f.score[0], f.score[1], 0, 0, b.status});
      return b;
    };
    if (scaled <= o.score_tol &&
        std::abs(arma::mean(energy / (g * c.eigen + e)) - 1) <= o.score_tol &&
        (it == 0 ||
         std::abs(d.lik - previous) <= o.score_tol * (1 + std::abs(d.lik)))) {
      record("interior", 0, 0, 0);
      if (!boundary.empty() && best_face().lik > d.lik + 1e-10)
        return accept_face();
      return {g, e, d.lik, "interior"};
    }
    // KKT-certified face with higher likelihood than the current iterate.
    // Accept only when the score points out of the positive quadrant.
    if (!boundary.empty()) {
      auto b = best_face();
      if (b.lik >= interior_bound - 1e-12 &&
          b.lik >= d.lik - 64 * arma::datum::eps * (1 + std::abs(d.lik)) &&
          ((b.g == 0 && d.score[0] < 0) || (b.e == 0 && d.score[1] < 0))) {
        audit.updates = it + 1;
        record("active-set", 0, 0, 0);
        auto f = evaluate(b.g, b.e);
        audit.steps.push_back({it + 1, 0, b.g, b.e, f.lik, f.score[0],
                               f.score[1], 0, 0, b.status});
        return b;
      }
    }
    if (it == o.max_reml) {
      record("iteration-limit", 0, 0, 0);
      throw std::runtime_error(o.varcomp + " iteration-limit");
    }
    vec delta;
    double condition = 0;
    if (o.varcomp == "em-reml") {
      delta = 2 * arma::square(vec{g, e}) % d.score / n;
    } else {
      mat ai = d.information % (units * units.t());
      condition = arma::rcond(ai);
      if (condition < 1e-14 ||
          !arma::solve(delta, ai, s, arma::solve_opts::no_approx)) {
        if (!boundary.empty() && best_face().lik >= d.lik)
          return accept_face();
        record("singular-information", 0, 0, condition);
        throw std::runtime_error("ai-reml singular average information, rank=" +
                                 std::to_string(arma::rank(ai)));
      }
      delta %= units;
      // Maximize the local AI quadratic on the feasible quadrant. Reoptimizing
      // the free coordinate on a face avoids getting stuck at a zero-length
      // step.
      if (g + delta[0] < 0 || e + delta[1] < 0) {
        vec dg = {-g, std::max(-e, (d.score[1] + d.information(1, 0) * g) /
                                       d.information(1, 1))};
        vec de = {std::max(-g, (d.score[0] + d.information(0, 1) * e) /
                                   d.information(0, 0)),
                  -e};
        auto gain = [&](const vec &h) {
          return arma::dot(d.score, h) - .5 * arma::dot(h, d.information * h);
        };
        delta = gain(dg) > gain(de) ? dg : de;
      }
    }
    double step = 1;
    int bt = 0;
    for (; bt <= o.max_backtracks; ++bt, step *= .5) {
      double ng = g + step * delta[0], ne = e + step * delta[1];
      if (ng >= 0 && ne >= 0 && ng + ne > 0 && (ne > 0 || c.eigen.min() > 0) &&
          std::isfinite(ng + ne)) {
        auto trial = evaluate(ng, ne);
        if (trial.lik >= d.lik - 32 * arma::datum::eps * (1 + std::abs(d.lik)))
          break;
      }
      if (o.varcomp == "em-reml")
        throw std::runtime_error("em-reml invalid or nonmonotone update");
    }
    require(bt <= o.max_backtracks, "ai-reml backtracking failed");
    record("update", step, bt, condition);
    previous = d.lik;
    g += step * delta[0];
    e += step * delta[1];
  }
  throw std::logic_error("unreachable REML termination");
}
} // namespace fsb
