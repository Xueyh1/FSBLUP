#include "core.hpp"
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
void exercise(const Layers &layers, const fs::path &path) {
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
    size_t threads = entries("/proc/self/task");
    for (int i = 0; i < 6; ++i)
      exercise(layers, path);
    size_t fd = entries("/proc/self/fd");
    std::cout << "iteration\tallocated_bytes\trss_KiB\tpss_KiB\tfds\tthreads\n";
    for (int i = 0; i <= repeats; ++i) {
      if (i)
        exercise(layers, path);
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
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
