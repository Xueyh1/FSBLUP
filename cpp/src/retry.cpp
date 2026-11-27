#include "core.hpp"
#include <numeric>
namespace fsb {
namespace {
size_t record_attempt(Evaluation &e, size_t limit) {
  size_t probes = 0, rhs = 0, mv = 0;
  for (const auto &f : e.fits) {
    probes += f.probes;
    rhs += f.rhs;
    mv += f.matvec;
  }
  e.attempts =
      std::to_string(limit) + ":" + (e.reason.empty() ? "ok" : e.reason) +
      " [rhs=" + std::to_string(rhs) + ",matvec=" + std::to_string(mv) +
      ",probes=" + std::to_string(probes) + "]";
  e.probe_limit = limit;
  return probes;
}
} // namespace
void retry_batch(const std::vector<Weight> &points,
                 std::vector<Evaluation> &results,
                 const std::vector<Fold> &folds, const FitOptions &options,
                 size_t &remaining, bool legacy,
                 const EvaluateFolds &evaluate) {
  if (options.trace != "hutchinson")
    return;
  const size_t required =
      std::count_if(folds.begin(), folds.end(),
                    [](const Fold &f) { return f.reason.empty(); });
  for (auto &e : results) {
    e.kept_probes = record_attempt(e, options.max_probes);
    if (!e.fits.empty() && !e.reason.empty())
      e.discarded_probes = e.fits.back().probes;
    e.kept_probes -= e.discarded_probes;
  }
  auto visit = [&](size_t i) {
    auto &e = results[i];
    bool pending = e.outcome == "pending";
    if (!required || (!e.probe_exhausted && !pending))
      return false;
    size_t limit = pending ? e.probe_limit : 2 * e.probe_limit;
    size_t reserve = (legacy ? required : 1) * limit;
    if (reserve > remaining)
      return false;
    remaining -= reserve;
    FitOptions next = options;
    next.max_probes = limit;
    size_t j = 0;
    if (!legacy)
      while (j < folds.size() &&
             (!folds[j].reason.empty() || std::isfinite(e.scores[j])))
        ++j;
    require(legacy || j < folds.size(), "missing fold in retry ledger");
    auto [a, b] = points[i];
    auto fresh =
        legacy ? evaluate(a, b, next, folds) : evaluate(a, b, next, {folds[j]});
    size_t usage = record_attempt(fresh, limit);
    if (legacy)
      e.discarded_probes += e.kept_probes;
    size_t failed = (!fresh.reason.empty() && !fresh.fits.empty())
                        ? fresh.fits.back().probes
                        : 0;
    e.discarded_probes += failed;
    fresh.kept_probes = (legacy ? 0 : e.kept_probes) + usage - failed;
    if (!legacy) {
      if (fresh.reason.empty())
        e.scores[j] = fresh.scores.at(0);
      fresh.scores = e.scores;
      long double sum = 0;
      size_t count = 0;
      for (double v : fresh.scores)
        if (std::isfinite(v)) {
          sum += v;
          ++count;
        }
      if (fresh.reason.empty() && count < required) {
        fresh.outcome = "pending";
        fresh.reason = "retry budget insufficient for remaining folds";
        fresh.value = NAN;
      } else if (count == required)
        fresh.value = sum / count;
    }
    fresh.attempts = e.attempts + " | " + fresh.attempts;
    fresh.retry_spent = e.retry_spent + reserve;
    fresh.retry_used = e.retry_used + usage;
    fresh.discarded_probes = e.discarded_probes;
    fresh.fold_attempts += e.fold_attempts;
    fresh.successful_fold_fits += e.successful_fold_fits;
    fresh.fold_seconds += e.fold_seconds;
    fresh.timing.add(e.timing);
    e.fits.insert(e.fits.end(), std::make_move_iterator(fresh.fits.begin()),
                  std::make_move_iterator(fresh.fits.end()));
    fresh.fits = std::move(e.fits);
    e = std::move(fresh);
    return true;
  };
  if (legacy)
    for (size_t i = 0; i < results.size(); ++i)
      while (visit(i)) {
      }
  else {
    std::vector<size_t> order(results.size());
    std::iota(order.begin(), order.end(), 0);
    bool progress;
    do {
      std::stable_sort(order.begin(), order.end(), [&](size_t l, size_t r) {
        auto done = [&](size_t i) {
          return std::count_if(results[i].scores.begin(),
                               results[i].scores.end(),
                               [](double x) { return std::isfinite(x); });
        };
        return done(l) > done(r);
      });
      progress = false;
      for (auto i : order)
        progress |= visit(i);
    } while (progress && remaining);
    for (auto &e : results)
      if (e.outcome == "pending") {
        e.outcome = "fit_failed";
        e.probe_exhausted = true;
      }
  }
  for (auto &e : results)
    if (!e.reason.empty())
      e.discarded_probes += e.kept_probes;
}
} // namespace fsb
