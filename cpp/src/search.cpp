#include "core.hpp"
#include <chrono>
#include <future>
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
// Keep a power-of-two exponent for the running sum of squared differences.
// This also handles an individual difference larger than DBL_MAX when its
// root mean square over all rows is representable.
static double root_mean_square(const vec &a, const vec *b = nullptr) {
  int largest = 0;
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; ++i) {
    double other = b ? (*b)[i] : 0;
    if (a[i] == other)
      continue;
    int exponent, adjustment;
    std::frexp(std::max(std::abs(a[i]), std::abs(other)), &exponent);
    double mantissa =
        std::frexp(std::scalbn(a[i], -exponent) - std::scalbn(other, -exponent),
                   &adjustment);
    exponent += adjustment;
    if (sum == 0)
      largest = exponent;
    if (exponent > largest) {
      sum = std::scalbn(sum, 2 * (largest - exponent));
      largest = exponent;
    }
    sum += std::scalbn(mantissa * mantissa, 2 * (exponent - largest));
  }
  return std::scalbn(std::sqrt(sum / a.n_elem), largest);
}
double score(const vec &truth, const vec &pred, const string &metric) {
  require(pred.n_elem == truth.n_elem && truth.is_finite() && pred.is_finite(),
          "invalid score vectors");
  auto reason = invalid_truth(truth, metric);
  require(reason.empty(), reason);
  if (metric == "rmse")
    return root_mean_square(truth, &pred);
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
void record_pd(Evaluation &e, const std::vector<PdBlock> &blocks) {
  e.correction = pd_summary(blocks);
  for (const auto &block : blocks) {
    e.timing.add(block.timing);
    e.nearpd_blocks += block.projected;
    e.max_relative_norm_f =
        std::max(e.max_relative_norm_f, block.relative_norm_f);
  }
}
void assess_folds(Evaluation &e, const std::vector<Fold> &folds,
                  const vec &truth, const string &metric, double a, double b,
                  const std::function<Model(const Fold &, FitAudit &)> &fit) {
  e.scores.resize(folds.size(), NAN);
  require(!folds.empty(), "no validation folds");
  long double sum = 0;
  size_t count = 0;
  for (size_t j = 0; j < folds.size(); ++j) {
    const auto &f = folds[j];
    if (!f.reason.empty())
      continue;
    auto &audit = e.fits.emplace_back();
    audit.context = "alpha=" + std::to_string(a) +
                    " beta=" + std::to_string(b) +
                    " repeat=" + std::to_string(f.repeat) +
                    " fold=" + std::to_string(f.fold);
    auto tick = std::chrono::steady_clock::now();
    try {
      auto model = fit(f, audit);
      e.scores[j] = score(truth.elem(f.valid), model.u.elem(f.valid), metric);
      require(std::isfinite(e.scores[j]), "nonfinite validation score");
    } catch (const CertificationUnavailable &ex) {
      e.outcome = "uncertified";
      e.reason = audit.context + ": " + ex.what();
      audit.status = e.reason;
    } catch (const std::runtime_error &ex) {
      e.reason = audit.context + ": " + ex.what();
      e.outcome = "fit_failed";
      audit.status = e.reason;
    }
    e.fold_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tick)
            .count();
    if (!e.reason.empty())
      return;
    sum += e.scores[j];
    ++count;
  }
  require(count > 0, "no scorable fold: " + folds.front().reason);
  e.value = sum / count;
  require(std::isfinite(e.value), "nonfinite mean validation score");
}
Evaluation evaluate_weight(const FusionPlan &plan, double a, double b,
                           const mat &x, const vec &y, const vec &truth,
                           const std::vector<Fold> &folds, const string &metric,
                           PdPolicy policy, const FitOptions *options) {
  Evaluation e;
  e.scores.resize(folds.size(), NAN);
  try {
    auto k = plan.combine(a, b, nullptr, policy, PdDiagnostics::tuning);
    record_pd(e, k.pd_blocks);
    assess_folds(
        e, folds, truth, metric, a, b, [&](const Fold &f, FitAudit &audit) {
          return options ? fit_model(k.k, x, y, f.train, *options, audit)
                         : solve(k.k, x, y, f.train);
        });
  } catch (const std::runtime_error &ex) {
    e.outcome = "invalid";
    e.reason = ex.what();
  }
  return e;
}
bool ScoreRule::better(double candidate, double best) const {
  const long double delta = static_cast<long double>(candidate) - best;
  return (minimize ? -delta : delta) > 1e-12L * scale;
}
bool ScoreRule::stopped(double before, double after) const {
  return std::abs(static_cast<long double>(before) - after) < 1e-4L * scale;
}
ScoreRule scoring_rule(const string &metric, const vec &truth,
                       const vec &response, const string &policy) {
  require(policy == "relative" || policy == "legacy", "unknown score policy");
  if (metric != "rmse" || policy == "legacy")
    return {metric == "rmse", 1};
  double scale = root_mean_square(truth);
  if (!scale)
    scale = root_mean_square(response);
  return {true, scale > 0 ? scale : 1};
}
Search search(bool strict, ScoreRule rule,
              const std::function<Evaluation(double, double)> &evaluate,
              size_t workers,
              const std::function<void(double, double, Evaluation &)> &retry) {
  require(workers > 0, "invalid worker count");
  Search out;
  using Point = std::pair<double, double>;
  std::set<Point> seen;
  std::vector<Point> batch;
  auto enqueue = [&](double a, double b) {
    if (seen.emplace(a, b).second)
      batch.emplace_back(a, b);
  };
  auto run_batch = [&](const string &stage) {
    // Join every task before reading any result or moving the refinement
    // centre.
    const size_t width = std::min(workers, batch.size());
    std::vector<Evaluation> results(batch.size());
    auto evaluate_stripe = [&](size_t start) {
      for (size_t i = start; i < batch.size(); i += width)
        results[i] = evaluate(batch[i].first, batch[i].second);
    };
    if (width <= 1)
      evaluate_stripe(0);
    else {
      std::vector<std::future<void>> tasks;
      for (size_t i = 1; i < width; ++i)
        tasks.push_back(std::async(std::launch::async, evaluate_stripe, i));
      evaluate_stripe(0);
      for (auto &task : tasks)
        task.get();
    }
    for (size_t i = 0; i < batch.size(); ++i) {
      auto [a, b] = batch[i];
      auto &e = results[i];
      if (retry)
        retry(a, b, e);
      double v = e.value;
      if (e.reason.empty() && std::isfinite(v) &&
          (!std::isfinite(out.best) || rule.better(v, out.best))) {
        out.alpha = a;
        out.beta = b;
        out.best = v;
      }
      out.trials.push_back({stage, a, b, std::move(e)});
    }
    batch.clear();
  };
  for (int i = 0; i <= 4; ++i)
    for (int j = 0; j <= 4 - i; ++j)
      enqueue(i / 4., j / 4.);
  run_batch("grid");
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
        enqueue(x, y);
      }
    run_batch("refine" + std::to_string(r));
    if (rule.stopped(old, out.best)) {
      out.stop = "improvement < 1e-4";
      break;
    }
    step /= 2;
  }
  return out;
}
} // namespace fsb
