#pragma once
#include "core.hpp"
#include <atomic>
namespace fsb {
struct PedigreeOperator {
  Ids ids;
  std::vector<long> sire,dam;
  vec diag,d;
  explicit PedigreeOperator(const Table&);
  mat apply(mat x) const;
};
struct FeatureFile {
  fs::path path;
  Ids ids;
  size_t columns;
  vec diag;
  mat apply(const mat &x,size_t block,std::atomic<uint64_t>& bytes) const;
};
struct Cache {
  PedigreeOperator ped;
  FeatureFile g,o;
  std::atomic<uint64_t> bytes{0};
  explicit Cache(const fs::path&);
  string mode() const;
  Layers materialize(double cap);
  TrainKernel kernel(const Ids& population,const uvec& train,double a,double b,double cap);
};
void prepare_cache(const fs::path& ped,const fs::path& snp,const fs::path& omics,bool fpkm,const fs::path& dir,double cap);
void scan_table(const fs::path&,const std::function<void(const Ids&)>& header,const std::function<void(const Ids&)>& row);
}
