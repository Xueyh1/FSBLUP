#include "core.hpp"
#include <random>
namespace fsb {
Labels random_labels(const Ids &ids, int folds, int repeats, unsigned seed) {
  require(folds >= 2 && size_t(folds) <= ids.size() && repeats > 0,
          "invalid fold/repeat count");
  Labels labels;
  std::mt19937 rng(seed);
  Ids sorted = ids;
  std::sort(sorted.begin(), sorted.end());
  for (int r = 1; r <= repeats; ++r) {
    auto shuffled = sorted;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    auto idx = positions(ids, shuffled);
    auto &row = labels[r];
    row.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
      row[idx[i]] = 1 + i % folds;
  }
  return labels;
}
static string invalid_truth(const vec &y, const string &metric) {
  if (y.empty())
    return "empty validation fold";
  if (metric == "cor" && (y.n_elem < 2 || y.max() == y.min()))
    return "constant validation response";
  if (metric == "auc") {
    require(arma::all((y == 0) || (y == 1)), "AUC requires labels 0/1");
    if (y.max() == y.min())
      return "single validation class";
  }
  return "";
}
std::vector<Fold> split(const Labels &labels, bool temporal, const vec &truth,
                        const string &metric) {
  require(metric == "cor" || metric == "rmse" || metric == "auc",
          "unknown metric: " + metric);
  std::vector<Fold> out;
  for (auto &[rep, v] : labels) {
    require(v.size() == truth.n_elem, "fold table incomplete");
    std::set<int> keys(v.begin(), v.end());
    if (temporal) {
      require(keys == std::set<int>{0, 1}, "time split requires both groups");
      keys.erase(0);
    } else {
      require(keys.size() >= 2 && *keys.begin() == 1 &&
                  *keys.rbegin() == int(keys.size()),
              "empty or missing fold");
    }
    for (auto f : keys) {
      std::vector<size_t> t, s;
      for (size_t i = 0; i < v.size(); ++i)
        (v[i] == f ? s : t).push_back(i);
      uvec train = arma::conv_to<uvec>::from(t),
           valid = arma::conv_to<uvec>::from(s);
      out.push_back(
          {rep, f, train, valid, invalid_truth(truth.elem(valid), metric)});
    }
  }
  return out;
}
double score(const vec &truth, const vec &pred, const string &metric) {
  require(pred.n_elem == truth.n_elem && truth.is_finite() && pred.is_finite(),
          "invalid score vectors");
  auto reason = invalid_truth(truth, metric);
  require(reason.empty(), reason);
  if (metric == "rmse")
    return std::sqrt(arma::mean(arma::square(truth - pred)));
  if (metric == "cor") {
    vec a = truth - arma::mean(truth), b = pred - arma::mean(pred);
    double d = arma::norm(a) * arma::norm(b);
    require(d > 0, "constant predictions");
    return arma::dot(a, b) / d;
  }
  require(metric == "auc", "unknown metric");
  uvec order = arma::sort_index(pred);
  double pairs = 0, neg = 0, pos = 0;
  for (size_t i = 0; i < order.n_elem;) {
    size_t j = i;
    double p = 0;
    while (j < order.n_elem && pred[order[j]] == pred[order[i]])
      p += truth[order[j++]];
    double n = (j - i) - p;
    pairs += p * (neg + n / 2);
    neg += n;
    pos += p;
    i = j;
  }
  return pairs / (neg * pos);
}
Evaluation evaluate_weight(FusionPlan &plan, double a, double b, const mat &x,
                           const vec &y, const vec &truth,
                           const std::vector<Fold> &folds, const string &metric,
                           PdPolicy policy) {
  Evaluation e;
  e.scores.resize(folds.size(), NAN);
  string location;
  try {
    require(!folds.empty(), "no validation folds");
    auto k = plan.combine(a, b, nullptr, policy, PdDiagnostics::tuning);
    e.correction = k.correction;
    for (const auto &block : k.pd_blocks) {
      e.nearpd_blocks += block.projected;
      e.max_relative_norm_f =
          std::max(e.max_relative_norm_f, block.relative_norm_f);
    }
    double sum = 0;
    size_t count = 0;
    for (size_t j = 0; j < folds.size(); ++j) {
      auto &f = folds[j];
      if (!f.reason.empty())
        continue;
      location = "repeat " + std::to_string(f.repeat) + " fold " +
                 std::to_string(f.fold) + ": ";
      auto fit = solve(k.k, x, y, f.train);
      e.scores[j] = score(truth.elem(f.valid), fit.u.elem(f.valid), metric);
      require(std::isfinite(e.scores[j]), "nonfinite validation score");
      sum += e.scores[j];
      ++count;
    }
    require(count > 0, "no scorable fold: " + folds.front().reason);
    e.value = sum / count;
    require(std::isfinite(e.value), "nonfinite mean validation score");
  } catch (const std::runtime_error &ex) {
    e.reason = location + ex.what();
  }
  return e;
}
Search search(bool strict, bool minimize,
              const std::function<Evaluation(double, double)> &evaluate) {
  Search out;
  std::set<std::pair<double, double>> seen;
  auto visit = [&](string stage, double a, double b) {
    if (!seen.emplace(a, b).second)
      return;
    Evaluation e = evaluate(a, b);
    double v = e.value;
    if (e.reason.empty() && std::isfinite(v) &&
        (!std::isfinite(out.best) ||
         (minimize ? v < out.best - 1e-12 : v > out.best + 1e-12))) {
      out.alpha = a;
      out.beta = b;
      out.best = v;
    }
    out.trials.push_back({stage, a, b, std::move(e)});
  };
  for (int i = 0; i <= 4; ++i)
    for (int j = 0; j <= 4 - i; ++j)
      visit("grid", i / 4., j / 4.);
  if (!std::isfinite(out.best)) {
    out.stop = "no eligible grid point";
    return out;
  }
  double step = .125;
  out.stop = "round limit";
  for (int r = 1; r <= 10; ++r) {
    auto a = out.alpha, b = out.beta, old = out.best;
    for (double da : {-step, step})
      for (double db : {-step, step}) {
        double x = a + da, y = b + db;
        if (strict && (x < 0 || y < 0 || x + y > 1))
          continue;
        visit("refine" + std::to_string(r), x, y);
      }
    if (std::abs(old - out.best) < 1e-4) {
      out.stop = "improvement < 1e-4";
      break;
    }
    step /= 2;
  }
  return out;
}
} // namespace fsb
