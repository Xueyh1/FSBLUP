#pragma once
#include <armadillo>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>
namespace fsb {
using Ids = std::vector<std::string>;
using arma::mat;
using arma::uvec;
using arma::vec;
using std::string;
namespace fs = std::filesystem;
void require(bool ok, const string &message);
bool missing(const string &s);
double number(const string &s);
int integer(const string &s, int minimum);
void unique_ids(const Ids &ids);
uvec positions(const Ids &source, const Ids &wanted);
Ids intersect(const Ids &first, const Ids &second);
struct Table {
  Ids header;
  std::vector<Ids> rows;
  size_t col(const string &) const;
  Ids column(const string &) const;
};
Table read_table(const fs::path &);
void write_file(const fs::path &, const std::function<void(std::ostream &)> &);
struct Kernel {
  Ids ids;
  mat values;
  Kernel subset(const Ids &) const;
};
Kernel read_features(const fs::path &, double memory_mb);
Kernel read_kernel(const fs::path &, double memory_mb);
string field(const string &);
void write_matrix(const fs::path &, const Ids &, const mat &);
double budget(size_t n, size_t m, double memory_mb);
struct Layers {
  Kernel a, g, o;
  string mode() const;
  Layers subset(const Ids &) const;
  string counts() const;
};
struct BlockAudit {
  mat base, calibrated, replacement, propagated;
};
enum class PdPolicy { strict, near_pd };
// Tuning records projection counts and relative changes. Final output also
// reports the exact minimum eigenvalue of each projected block after rescaling.
enum class PdDiagnostics { full, tuning };
struct PdBlock {
  string block;
  bool projected = false;
  int iterations = 0;
  double min_raw = NAN, min_used = NAN, rel_tol = NAN;
  double norm_f = 0., relative_norm_f = 0., max_abs_change = 0.;
};
struct Fusion {
  mat raw, k;
  string correction;
  std::vector<PdBlock> pd_blocks;
};
// Bound to one immutable ID universe. Reference and full cohorts each own a
// plan.
class FusionPlan {
public:
  explicit FusionPlan(const Layers &layers)
      : layers_(layers), mode_(layers.mode()) {}
  FusionPlan(Layers &&) = delete;
  Fusion combine(double alpha, double beta, BlockAudit *audit = nullptr,
                 PdPolicy policy = PdPolicy::strict,
                 PdDiagnostics diagnostics = PdDiagnostics::full);

private:
  struct Nested {
    explicit Nested(const Layers &layers);
    uvec u, g, s;
    mat a44, g44, g22, r14, r24, k12;
  };
  const Layers &layers_;
  string mode_;
  std::optional<Nested> nested_;
};
Fusion fuse(const Layers &, double alpha, double beta,
            BlockAudit *audit = nullptr, PdPolicy policy = PdPolicy::strict,
            PdDiagnostics diagnostics = PdDiagnostics::full);
struct Features {
  mat k;
  size_t kept = 0, filtered = 0, constant = 0, imputed = 0;
};
Kernel pedigree(const Ids &, const Ids &, const Ids &, double memory_mb);
Features genomic(mat);
Features omics(mat, bool fpkm);
struct Model {
  vec b, u, yhat;
  double lambda, vu, ve, objective;
  string status;
};
Model solve(const mat &, const mat &, const vec &, const uvec &,
            double fixed_lambda = 0);
struct Fold {
  int repeat, fold;
  uvec train, valid;
  string reason;
};
using Labels = std::map<int, std::vector<int>>;
Labels random_labels(const Ids &, int folds, int repeats, unsigned seed);
std::vector<Fold> split(const Labels &, bool temporal, const vec &truth,
                        const string &metric);
double score(const vec &, const vec &, const string &);
struct Evaluation {
  double value = NAN;
  string reason, correction;
  std::vector<double> scores;
  size_t nearpd_blocks = 0;
  double max_relative_norm_f = 0.;
};
Evaluation evaluate_weight(FusionPlan &plan, double a, double b, const mat &x,
                           const vec &y, const vec &truth,
                           const std::vector<Fold> &folds, const string &metric,
                           PdPolicy policy = PdPolicy::strict);
struct Trial {
  string stage;
  double alpha, beta;
  Evaluation result;
};
struct Search {
  std::vector<Trial> trials;
  double alpha = 0, beta = 0, best = NAN;
  string stop;
};
Search search(bool strict, bool minimize,
              const std::function<Evaluation(double, double)> &evaluate);
} // namespace fsb
