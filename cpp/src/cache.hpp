#pragma once
#include "core.hpp"
#include <atomic>
namespace fsb {
struct PedigreeOperator {
  Ids ids;
  std::vector<long> sire, dam;
  vec diag, d;
  explicit PedigreeOperator(const Table &);
  mat apply(mat x) const;
  std::function<mat(const mat &)> restricted(const uvec &rows) const;
};
struct CacheIO {
  std::atomic<uint64_t> bytes{0}, reads{0}, read_ns{0}, decode_ns{0};
};
struct FeatureFile {
  fs::path path;
  Ids ids;
  size_t columns;
  vec diag;
  mat lookup;
  size_t stride = 0;
  void read(std::ifstream &, size_t rows, double *dest, CacheIO &) const;
  mat apply(const mat &x, size_t block, CacheIO &io,
            const uvec *subset = nullptr) const;
};
struct PopulationKernel {
  Ids ids;
  vec diagonal;
  std::function<mat(const mat &)> apply;
  std::vector<PdBlock> diagnostics;
  std::function<std::function<mat(const mat &)>(const uvec &, double)>
      train_apply;
  double retained_mib = 0;
  TrainKernel restrict(const Ids &targets, const uvec &train, double cap,
                       double solver_mib = 0) const;
};
struct Cache {
  PedigreeOperator ped;
  FeatureFile g, o;
  CacheIO io;
  uint64_t verified_bytes = 0;
  bool streaming = false, resident = false;
  explicit Cache(const fs::path &, double cap);
  string coverage;
  const string &mode() const { return coverage; }
  Layers materialize(double cap);
  PopulationKernel prepare(const Ids &population, double a, double b,
                           PdPolicy policy, double cap);
};
void prepare_cache(const fs::path &ped, const fs::path &snp,
                   const fs::path &omics, bool fpkm, const fs::path &dir,
                   double cap, bool two_bit = false);
void scan_genotypes(
    const fs::path &, const std::function<void(const Ids &)> &,
    const std::function<void(const string &, const std::vector<uint8_t> &)> &);
void scan_table(const fs::path &,
                const std::function<void(const Ids &)> &header,
                const std::function<void(const Ids &)> &row);
} // namespace fsb
