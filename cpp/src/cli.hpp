#pragma once
#include "core.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <unistd.h>
extern "C" void openblas_set_num_threads(int);
extern "C" int openblas_get_num_threads();
extern "C" char *openblas_get_config();
namespace fsb {
struct Options {
  std::map<string, string> v;
  bool has(string k) const { return v.count(k); }
  string get(string k) const {
    require(has(k), "missing --" + k);
    return v.at(k);
  }
  string get(string k, string def) const { return has(k) ? v.at(k) : def; }
  Options(int argc, char **argv, string cmd) {
    std::set<string> allowed = {"out", "threads", "memory-mb"};
    if (cmd == "build")
      allowed.insert({"ped", "snp", "omics", "omics-type"});
    else {
      allowed.insert({"A", "G", "O", "alpha", "beta", "pd-policy"});
      if (cmd == "fit")
        allowed.insert({"pheno", "id", "trait", "folds", "repeats", "seed",
                        "folds-file", "year", "ngen", "metric", "score-col",
                        "covar", "factor", "strict-simplex", "write-k"});
      else
        require(cmd == "fuse", "unknown subcommand: " + cmd);
    }
    for (int i = 2; i < argc; ++i) {
      string arg = argv[i];
      require(arg.rfind("--", 0) == 0, "expected --option: " + arg);
      string key = arg.substr(2);
      require(allowed.count(key) && !has(key),
              "unknown or duplicate option: " + arg);
      if (key == "strict-simplex" || key == "write-k") {
        v[key] = "true";
        continue;
      }
      require(++i < argc && string(argv[i]).rfind("--", 0) != 0,
              "missing value for " + arg);
      v[key] = argv[i];
    }
  }
};
using Clock = std::chrono::steady_clock;
struct Timing {
  Clock::time_point start = Clock::now(), last = start;
  std::vector<std::pair<string, double>> times;
  void mark(string stage) {
    auto now = Clock::now();
    times.emplace_back(stage,
                       std::chrono::duration<double>(now - last).count());
    last = now;
  }
  void save(const fs::path &dir) {
    write_file(dir / "timing.tsv", [&](std::ostream &f) {
      f << "stage\tseconds\n";
      for (auto &[s, t] : times)
        f << s << '\t' << t << '\n';
    });
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    write_file(dir / "runtime.tsv", [&](std::ostream &g) {
      g << "blas\tthreads\tpeak_rss_kib\twall_seconds\n"
        << openblas_get_config() << '\t' << openblas_get_num_threads() << '\t'
        << usage.ru_maxrss << '\t'
        << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    });
  }
};
struct TuningFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};
void run(const string &, const Options &, const fs::path &, double, Timing &);
} // namespace fsb
