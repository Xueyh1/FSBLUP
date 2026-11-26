#include "cli.hpp"
using namespace fsb;
namespace {
const char *help = R"(fsblup 0.1.0
  build --ped PED --snp SNP --omics OMICS [--omics-type generic|fpkm] --out DIR
  fuse --A A --G G --O O --alpha AWT --beta BWT --out DIR
  fit --pheno TABLE --id ID --trait TRAIT --A A --G G --O O --out DIR
      [--alpha AWT --beta BWT | --folds N --repeats N --seed N | --folds-file FILE
       | --year CUTOFF | --ngen CUTOFF] [--metric cor|rmse|auc] [--score-col COL]
      [--covar COL,...] [--factor COL,...] [--strict-simplex] [--write-k]
  All commands accept --threads N (default 1) and --memory-mb N (dense estimate).
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
      std::cout << "fsblup 0.1.0\n";
      return 0;
    }
    string cmd = argv[1];
    Options opt(argc, argv, cmd);
    int threads = integer(opt.get("threads", "1"), 1);
    openblas_set_num_threads(threads);
    double cap = number(opt.get("memory-mb", "8192"));
    Destination dest(opt.get("out"));
    Timing tm;
    try {
      run(cmd, opt, dest.work, cap, tm);
    } catch (const TuningFailure &e) {
      output(dest.work / "failure.txt") << e.what() << '\n';
      tm.mark("failed_search");
      tm.save(dest.work);
      dest.publish();
      throw;
    }
    tm.save(dest.work);
    dest.publish();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "fsblup: " << e.what() << '\n';
    return 1;
  }
}
