#include "cli.hpp"
#include <csignal>
using namespace fsb;
namespace {
const char *help = R"(fsblup 0.4.0
  prepare [--snp-format f64|2bit] --ped PED --snp SNP --omics OMICS [--omics-type generic|fpkm] --out CACHE
  fit --cache CACHE --pheno TABLE --trait TRAIT [fit options] --out DIR
  build --ped PED --snp SNP --omics OMICS [--omics-type generic|fpkm] --out DIR
  fuse --A A --G G --O O --alpha AWT --beta BWT --out DIR
  fit --pheno TABLE --id ID --trait TRAIT --A A --G G --O O --out DIR
      [--alpha AWT --beta BWT | --folds N --repeats N --seed N | --folds-file FILE
       | --year CUTOFF | --ngen CUTOFF] [--metric cor|rmse|auc] [--score-col COL]
      [--covar COL,...] [--factor COL,...] [--parallel blas|weights] [--strict-simplex] [--write-k]
  fit: --solver pcg|direct --varcomp ai-reml|em-reml|profile-reml|fixed
       --trace exact|hutchinson --probes N --max-probes N --probe-seed N --slq-depth N --lambda L --preconditioner jacobi|identity|nystrom --pcg-rtol R --pcg-atol A
       --pcg-maxiter N --pcg-rank N --reml-maxiter N --reml-tol T --init-vu G --init-ve E
  Cache: --cache-backend auto|operator|dense --cache-storage auto|stream|resident
  Random trace: --probe-solver pcg|lanczos --probe-batch N --mc-tol T --slq-tol T --slq-reorth full|local --trace-rank R
  Logging: --log-level off|error|warn|info|debug|trace --log-file FILE --log-every N
  All commands accept --threads N (default 1) and --memory-mb N (dense estimate).
  fuse and fit accept --pd-policy strict|nearPD (default strict).
  Input tables have headers. Pedigree columns are ID,Sire,Dam (case-insensitive).
  Default tuning uses balanced 5-fold CV twice with seed 42. --out must be new or empty.
)";
struct Destination {
  fs::path dest, work;
  Destination(fs::path path) : dest(path.lexically_normal()) {
    if (dest.has_relative_path() && dest.filename().empty())
      dest = dest.parent_path();
    require(!dest.empty() && (!fs::exists(dest) ||
                              (fs::is_directory(dest) && fs::is_empty(dest))),
            "output directory must be new or empty: " + dest.string());
    if (dest.has_parent_path())
      fs::create_directories(dest.parent_path());
    work = dest.parent_path() /
           (dest.filename().string() + ".tmp." + std::to_string(getpid()));
    require(fs::create_directory(work), "cannot reserve temporary output");
  }
  ~Destination() {
    std::error_code ec;
    fs::remove_all(work, ec);
  }
  void publish() { fs::rename(work, dest); }
};
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc < 2 || string(argv[1]) == "--help" ||
        (argc == 3 && string(argv[2]) == "--help")) {
      std::cout << help;
      return 0;
    }
    if (string(argv[1]) == "--version") {
      std::cout << "fsblup 0.4.0\n";
      return 0;
    }
    string cmd = argv[1];
    Options opt(argc, argv, cmd);
    int threads = integer(opt.get("threads", "1"), 1);
    auto parallel = opt.get("parallel", "blas");
    require(parallel == "blas" || parallel == "weights",
            "--parallel requires blas or weights");
    require(parallel != "weights" || !opt.has("alpha"),
            "--parallel weights requires automatic tuning");
    openblas_set_num_threads(parallel == "weights" ? 1 : threads);
    double cap = number(opt.get("memory-mb", "8192"));
    // Linux first signals SIGXFSZ at a file-size limit. Let write/close
    // return EFBIG so the checked output path can unwind its staging directory.
    if (std::signal(SIGXFSZ, SIG_IGN) == SIG_ERR)
      throw std::runtime_error("cannot configure file-size limit handling");
    Destination dest(opt.get("out"));
    string log_path = opt.get("log-file", "");
    fs::path publish_log;
    if (opt.has("log-file")) {
      require(!log_path.empty(), "empty --log-file");
      for (const auto &key :
           {"A", "G", "O", "pheno", "ped", "snp", "omics", "folds-file"}) {
        if (opt.has(key) && fs::exists(log_path) && fs::exists(opt.get(key)))
          require(!fs::equivalent(log_path, opt.get(key)),
                  "log file conflicts with input " + string(key));
      }
      auto absolute = fs::absolute(log_path).lexically_normal();
      if (opt.has("cache") && fs::exists(log_path)) {
        auto rel = fs::weakly_canonical(log_path).lexically_relative(
            fs::weakly_canonical(opt.get("cache")));
        require(rel.empty() || *rel.begin() == "..",
                "log file conflicts with input cache");
      }
      auto relative = absolute.lexically_relative(
          fs::absolute(dest.dest).lexically_normal());
      if (!relative.empty() && *relative.begin() != "..") {
        publish_log = dest.work / relative;
        log_path = (dest.work / ".fsblup-internal-log").string();
      }
    }
    integer(opt.get("log-every", "10"), 1);
    RunLog logger(opt, log_path, cmd);
    auto publish = [&] {
      logger.verify();
      if (!publish_log.empty()) {
        require(!fs::exists(publish_log),
                "log file conflicts with result file");
        fs::create_directories(publish_log.parent_path());
        fs::rename(log_path, publish_log);
      }
      dest.publish();
    };
    Timing tm;
    tm.log = &logger;
    try {
      run(cmd, opt, dest.work, cap, tm);
    } catch (const TuningFailure &e) {
      write_file(dest.work / "failure.txt",
                 [&](std::ostream &f) { f << e.what() << '\n'; });
      tm.mark("failed_search");
      tm.save(dest.work);
      logger.info(string("stage=failed reason=") + e.what());
      publish();
      throw;
    } catch (const std::exception &e) {
      logger.error(string("stage=failed reason=") + e.what());
      logger.verify();
      throw;
    }
    tm.save(dest.work);
    logger.info("stage=complete");
    publish();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "fsblup: " << e.what() << '\n';
    return 1;
  }
}
