#include "cache.hpp"
#include "cli.hpp"
#include "near_pd.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <thread>
#include <unistd.h>
using namespace fsb;
namespace {
size_t entries(const fs::path &p) {
  return std::distance(fs::directory_iterator(p), fs::directory_iterator());
}
int kib(const char *field) {
  std::ifstream f("/proc/self/smaps_rollup");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind(field, 0) == 0)
      return std::stoi(line.substr(std::strlen(field)));
  return -1;
}
void exercise_retry(const fs::path &dir) {
  const auto kernel =
      (fs::path(FSBLUP_TEST_DATA).parent_path() / "pcg_data" / "kernel.tsv")
          .string();
  Options options(0, nullptr, "fit");
  options.v = {
      {"A", kernel},           {"G", kernel},
      {"O", kernel},           {"pheno", (dir / "retry-trait.tsv").string()},
      {"trait", "y"},          {"year", "0"},
      {"metric", "rmse"},      {"strict-simplex", "true"},
      {"trace", "hutchinson"}, {"probe-solver", "lanczos"},
      {"slq-depth", "24"},     {"probes", "2"},
      {"max-probes", "2"},     {"mc-tol", "1e-12"},
      {"retry-probes", "12"},  {"parallel", "weights"},
      {"threads", "2"}};
  Timing timing;
  auto out = dir / "retry-fit";
  bool failed = false;
  try {
    run("fit", options, out, 128, timing);
  } catch (const TuningFailure &e) {
    failed = string(e.what()).find("probe budget exhausted") != string::npos;
  }
  require(failed, "expected exhausted stochastic search");
  auto summary = read_table(out / "tuning.tsv");
  require(summary.column("retry_reserved").at(0) == "12",
          "retry budget not consumed");
}
void exercise(const Layers &layers, const fs::path &path,
              const fs::path &cache_dir) {
  mat feature(9, 3);
  for (size_t j = 0; j < feature.n_cols; ++j)
    for (size_t i = 0; i < feature.n_rows; ++i)
      feature(i, j) = (i + 2 * j) % 3;
  feature(4, 1) = arma::datum::nan;
  require(genomic(feature).k.is_finite() && omics(feature, false).k.is_finite(),
          "repeated raw kernels");
  mat testk = layers.a.values;
  vec testy = {-3., -.2, 1.7, 4.5, -1.6, 2.9, 3.5, 1.};
  mat design(8, 1, arma::fill::ones);
  FitOptions pcg;
  pcg.varcomp = "fixed";
  pcg.lambda = .5;
  FitAudit audit;
  uvec train = {0, 1, 2, 3, 4, 5, 6};
  auto m = fit_model(testk, design, testy, train, pcg, audit);
  require(m.u.is_finite(), "repeated PCG");
  FitOptions exact;
  FitAudit derived;
  auto fitted = fit_model(testk, design, testy, train, exact, derived);
  require(fitted.u.is_finite() && !derived.steps.empty(), "repeated AI fit");
  exact.varcomp = "em-reml";
  exact.max_reml = 8000;
  auto em = fit_model(testk, design, testy, train, exact, derived);
  require(em.u.is_finite(), "repeated EM fit");
  vec diagonal = {.002, .01, .1, 1, 5, 10, 20, 100};
  vec response = {.0778695253, 1.118391287, 2.157258318, .692639708,
                  .998079965,  -6.03329651, .521829145,  .46722856};
  FitOptions stochastic;
  stochastic.trace = "hutchinson";
  stochastic.probe_lanczos = true;
  auto all = arma::regspace<uvec>(0, 7);
  auto internal = fit_model(arma::diagmat(diagonal), design, response, all,
                            stochastic, derived);
  require(internal.status == "approx-interior" && internal.vu > 1,
          "repeated stochastic face selection");
  exercise_retry(cache_dir);
  Cache cache(cache_dir, 256);
  Ids cache_ids(cache.ped.ids.begin(), cache.ped.ids.begin() + 8);
  auto cache_folds =
      split({{1, {1, 2, 1, 2, 1, 2, 1, 2}}}, false, testy, "rmse");
  auto cache_search = search(
      true, true,
      [&](double a, double b) {
        auto prepared = cache.prepare(cache_ids, a, b, PdPolicy::strict, 256);
        Evaluation ev;
        assess_folds(ev, cache_folds, testy, "rmse", a, b,
                     [&](const Fold &fold, FitAudit &trace) {
                       auto op = prepared.restrict(cache_ids, fold.train, 256);
                       return fit_operator(op, design.rows(fold.train), design,
                                           testy.elem(fold.train), pcg, trace);
                     });
        return ev;
      },
      2);
  require(std::isfinite(cache_search.best), "repeated cache CV");
  pcg.max_pcg = 1;
  bool failed = false;
  try {
    fit_model(testk, design, testy, train, pcg, audit);
  } catch (const std::runtime_error &) {
    failed = true;
  }
  require(failed, "missing PCG failure");
  const FusionPlan plan(layers);
  vec y = {-3., -.2, 1.7, 4.5, -1.6, 2.9, 3.5, 1.};
  mat x(8, 1, arma::fill::ones);
  auto splits = split({{1, {1, 2, 1, 2, 1, 2, 1, 2}}}, false, y, "rmse");
  auto result = search(
      true, true,
      [&](double a, double b) {
        // Interleave ordinary per-candidate failures and genuine projected
        // fits.
        return evaluate_weight(plan, a, b, x, y, y, splits, "rmse",
                               b > .5 ? PdPolicy::strict : PdPolicy::near_pd);
      },
      4);
  require(result.trials.size() >= 15 && std::isfinite(result.best),
          "repeat search");
  require(std::any_of(result.trials.begin(), result.trials.end(),
                      [](const Trial &t) { return !t.result.reason.empty(); }),
          "lost failure");
  write_file(path, [](std::ostream &f) { f << "ok\n"; });
  try {
    write_file(path,
               [](std::ostream &) { throw std::runtime_error("injected"); });
  } catch (const std::runtime_error &) {
  }
  auto failure = search(
      false, true,
      [](double, double) {
        Evaluation e;
        e.reason = "unusable";
        return e;
      },
      4);
  require(failure.trials.size() == 15 && !std::isfinite(failure.best),
          "failed batch");
}
} // namespace
int main(int argc, char **argv) {
  try {
    int repeats = argc == 2 ? integer(argv[1], 1) : 8;
    Ids ids = {"01", "02", "03", "04", "05", "06", "07", "08"};
    mat a = arma::eye(8, 8), g = arma::eye(5, 5);
    a(0, 3) = a(3, 0) = .9;
    g(0, 0) = 4;
    Layers layers{{ids, a},
                  {{ids.begin() + 3, ids.end()}, g},
                  {{ids.begin() + 5, ids.end()}, arma::eye(3, 3)}};
    auto path = fs::temp_directory_path() /
                ("fsblup-repeat-" + std::to_string(getpid()));
    auto cache_dir = path.string() + "-cache";
    fs::create_directory(cache_dir);
    fs::path data = FSBLUP_TEST_DATA;
    prepare_cache(data / "ped.tsv", data / "snp.tsv", data / "omics.tsv", false,
                  cache_dir, 256, true);
    fs::create_directory(fs::path(cache_dir) / "retry-fit");
    auto traits = read_table(data.parent_path() / "pcg_data" / "trait.tsv");
    write_file(fs::path(cache_dir) / "retry-trait.tsv", [&](auto &out) {
      out << "ID\ty\tyear\n";
      for (size_t j = 0; j < traits.rows.size(); ++j) {
        out << traits.rows[j][0] << '\t'
            << (j < 32 ? traits.rows[j][1] : std::to_string(1 + j / 30.))
            << '\t' << (j >= 32) << '\n';
      }
    });
    // Start sanitizer and pthread runtime services before recording the
    // process baseline. Join a single short-lived worker and let procfs
    // retire its task entry before exercising any model code.
    std::thread([] {}).join();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    size_t threads = entries("/proc/self/task");
    for (int i = 0; i < 6; ++i)
      exercise(layers, path, cache_dir);
    size_t fd = entries("/proc/self/fd");
    std::cout << "iteration\tallocated_bytes\trss_KiB\tpss_KiB\tfds\tthreads\n";
    for (int i = 0; i <= repeats; ++i) {
      if (i)
        exercise(layers, path, cache_dir);
      if (i == 0 || i == repeats || i % 10 == 0) {
        // The kernel can briefly list a joined thread after pthread_join
        // returns.
        for (int retry = 0; retry < 50 && entries("/proc/self/task") != threads;
             ++retry)
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        std::cout << i << '\t' << (mallinfo2().uordblks + mallinfo2().hblkhd)
                  << '\t' << kib("Rss:") << '\t' << kib("Pss:") << '\t'
                  << entries("/proc/self/fd") << '\t'
                  << entries("/proc/self/task") << '\n';
        require(entries("/proc/self/fd") == fd &&
                    entries("/proc/self/task") == threads,
                "resource count changed");
      }
    }
    fs::remove(path);
    fs::remove_all(cache_dir);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
