#include "core.hpp"
#include <queue>
namespace fsb {
Kernel pedigree(const Ids &ids, const Ids &sire, const Ids &dam, double cap) {
  unique_ids(ids);
  require(ids.size() == sire.size() && ids.size() == dam.size(),
          "pedigree column lengths");
  Ids all = ids;
  std::map<string, size_t> index;
  for (size_t i = 0; i < ids.size(); ++i) {
    require(ids[i] != "0", "0 is reserved for unknown parent");
    index[ids[i]] = i;
  }
  for (auto parents : {&sire, &dam})
    for (auto &p : *parents) {
      require(!missing(p), "use 0 for an unknown parent");
      if (p != "0" && !index.count(p)) {
        index[p] = all.size();
        all.push_back(p);
      }
    }
  size_t n = all.size();
  budget(n, 0, cap);
  std::vector<std::vector<size_t>> children(n), parents(n);
  std::vector<int> degree(n);
  for (size_t i = 0; i < ids.size(); ++i)
    for (auto &p : {sire[i], dam[i]})
      if (p != "0") {
        size_t j = index.at(p);
        parents[i].push_back(j);
        children[j].push_back(i);
        ++degree[i];
      }
  std::queue<size_t> ready;
  for (size_t i = 0; i < n; ++i)
    if (!degree[i])
      ready.push(i);
  mat a(n, n, arma::fill::zeros);
  std::vector<size_t> done;
  while (!ready.empty()) {
    size_t i = ready.front();
    ready.pop();
    for (size_t j : done) {
      double v = 0;
      for (auto p : parents[i])
        v += a(p, j) / 2;
      a(i, j) = a(j, i) = v;
    }
    a(i, i) = 1;
    if (parents[i].size() == 2)
      a(i, i) += a(parents[i][0], parents[i][1]) / 2;
    done.push_back(i);
    for (auto j : children[i])
      if (--degree[j] == 0)
        ready.push(j);
  }
  require(done.size() == n, "cycle in pedigree");
  return {all, std::move(a)};
}
Features genomic(mat x) {
  Features out;
  double denom = 0;
  for (size_t j = 0; j < x.n_cols; ++j) {
    vec v = x.col(j);
    v.replace(-9, arma::datum::nan);
    auto valid = arma::find_finite(v).eval();
    require(!valid.empty() || v.has_nan(), "invalid genotype");
    for (auto i : valid)
      require(v[i] == 0 || v[i] == 1 || v[i] == 2,
              "genotypes must be 0, 1, 2, NA or -9");
    if (valid.n_elem * 2 < v.n_elem) {
      ++out.filtered;
      continue;
    }
    out.imputed += v.n_elem - valid.n_elem;
    v.replace(arma::datum::nan, arma::mean(v.elem(valid)));
    double het = arma::accu(v == 1);
    double p = (2 * arma::accu(v == 2) + het) / (2 * v.n_elem),
           q = (2 * arma::accu(v == 0) + het) / (2 * v.n_elem);
    denom += 2 * p * q;
    x.col(out.kept++) = v - 2 * p;
  }
  require(out.kept && denom > 0, "no usable genomic denominator");
  auto w = x.head_cols(out.kept);
  out.k = w * w.t() / denom;
  return out;
}
Features omics(mat x, bool fpkm) {
  require(x.n_rows >= 2, "omics requires at least two samples");
  Features out;
  for (size_t j = 0; j < x.n_cols; ++j) {
    vec v = x.col(j);
    out.imputed += arma::find_nonfinite(v).eval().n_elem;
    v.replace(arma::datum::nan, 0);
    if (fpkm && arma::accu(v < 0.1) * 100 >= 95 * v.n_elem) {
      ++out.filtered;
      continue;
    }
    if (v.max() == v.min()) {
      ++out.constant;
      continue;
    }
    x.col(out.kept++) = (v - arma::mean(v)) / arma::stddev(v);
  }
  require(out.kept > 0, "no remaining omics features");
  auto w = x.head_cols(out.kept);
  out.k = w * w.t() / out.kept;
  return out;
}
string Layers::counts() const {
  std::map<string, int> membership;
  for (auto &id : a.ids)
    membership[id] |= 4;
  for (auto &id : g.ids)
    membership[id] |= 2;
  for (auto &id : o.ids)
    membership[id] |= 1;
  size_t counts[8] = {};
  string examples;
  for (auto &[id, mask] : membership) {
    if (!counts[mask])
      examples += " example_" + std::to_string(mask) + "=" + id;
    ++counts[mask];
  }
  std::ostringstream s;
  s << "A=" << a.ids.size() << " G=" << g.ids.size() << " O=" << o.ids.size()
    << " union=" << membership.size() << " intersection=" << counts[7];
  const char *labels[] = {"",       "O_only",  "G_only",  "GO_only",
                          "A_only", "AO_only", "AG_only", "AGO"};
  for (int i = 1; i < 8; ++i)
    s << " " << labels[i] << "=" << counts[i];
  return s.str() + examples;
}
string Layers::mode() const {
  auto ag = intersect(a.ids, g.ids).size(), go = intersect(g.ids, o.ids).size();
  if (ag != g.ids.size() || go != o.ids.size() || o.ids.empty())
    throw std::runtime_error("unsupported coverage: " + counts() +
                             "; supply nonempty O subset of G subset of A");
  if (ag == a.ids.size())
    return go == ag ? "A" : "B";
  require(go < ag, "unsupported coverage: " + counts() +
                       "; supply full A/G/O or a strict O subset of G");
  return "C";
}
Layers Layers::subset(const Ids &ids) const {
  return {a.subset(intersect(ids, a.ids)), g.subset(intersect(ids, g.ids)),
          o.subset(intersect(ids, o.ids))};
}
namespace {
// Positive semidefinite inputs receive the baseline diagonal shift. Assess the
// unshifted spectrum on Cholesky failure so the shift cannot hide
// indefiniteness.
mat adjusted(const mat &raw, string &note) {
  require(raw.is_finite(), "nonfinite covariance");
  mat chol;
  if (!arma::chol(chol, raw)) {
    vec eigen = arma::eig_sym(raw);
    double tolerance = 64 * std::numeric_limits<double>::epsilon() *
                       raw.n_rows * std::max(1., arma::abs(eigen).max());
    require(eigen.min() >= -tolerance,
            "indefinite covariance, min eigenvalue=" +
                std::to_string(eigen.min()));
    note += "semidefinite/roundoff,";
  }
  mat k = raw;
  k.diag() += 1e-6;
  note += "jitter=1e-6";
  return k;
}
mat regression(const mat &cross, const mat &base) {
  mat lower;
  require(arma::chol(lower, base, "lower"),
          "singular pedigree conditioning block");
  return arma::solve(arma::trimatu(lower.t()),
                     arma::solve(arma::trimatl(lower), cross.t()))
      .t();
}
uvec outside(size_t n, const uvec &s) {
  uvec mask(n, arma::fill::ones);
  mask.elem(s).zeros();
  return arma::find(mask);
}
} // namespace
Fusion fuse(const Layers &l, double alpha, double beta, BlockAudit *audit) {
  const mat &a = l.a.values;
  string mode = l.mode(), note;
  mat raw;
  if (mode == "A") {
    if (l.g.ids == l.a.ids && l.o.ids == l.a.ids) {
      raw = (1 - alpha - beta) * a + beta * l.g.values + alpha * l.o.values;
    } else {
      raw = (1 - alpha - beta) * a + beta * l.g.subset(l.a.ids).values +
            alpha * l.o.subset(l.a.ids).values;
    }
  } else if (mode == "B") {
    mat g = l.g.subset(l.a.ids).values;
    auto s = positions(l.a.ids, l.o.ids), u = outside(a.n_rows, s);
    note = "Gss:";
    mat base = adjusted(g.submat(s, s), note);
    auto &o = l.o.values;
    double od = arma::mean(o.diag()), gd = arma::mean(base.diag());
    mat t = base;
    if (alpha != 0) {
      double contrast = arma::mean(arma::mean(o)) - od;
      require(std::abs(contrast) > 1e-14 * std::max(1., arma::abs(o).max()),
              "unidentifiable BB000 omics calibration");
      double slope = (arma::mean(arma::mean(base)) - gd) / contrast;
      mat calibrated = slope * o + (od - slope * gd);
      t = alpha * calibrated + (1 - alpha) * base;
      if (audit)
        audit->calibrated = std::move(calibrated);
    }
    mat reg = regression(g.submat(u, s), base);
    mat h = g;
    h.submat(s, s) = t;
    h.submat(u, u) += reg * (t - base) * reg.t();
    h.submat(u, s) = reg * t;
    h.submat(s, u) = h.submat(u, s).t();
    if (audit) {
      audit->base = base;
      audit->replacement = t;
      audit->propagated = h;
    }
    double scale = arma::mean(h.diag());
    require(scale > 0, "nonpositive BB000 scale");
    raw = beta * h / scale + (1 - beta) * a;
    note += ";K:";
  } else {
    // Equation 3 with group 3 empty. Groups 1/2/4: pedigree only,
    // genotyped only, and jointly observed. All regressions use original A.
    auto s = positions(l.a.ids, l.o.ids);
    Ids group2;
    std::set<string> observed(l.o.ids.begin(), l.o.ids.end());
    for (auto &id : l.g.ids)
      if (!observed.count(id))
        group2.push_back(id);
    auto g = positions(l.a.ids, group2),
         u = outside(a.n_rows, positions(l.a.ids, l.g.ids));
    mat a44 = a.submat(s, s), c = alpha * l.o.values +
                                  beta * l.g.subset(l.o.ids).values +
                                  (1 - alpha - beta) * a44;
    mat g22 = l.g.subset(group2).values, r14 = regression(a.submat(u, s), a44);
    raw = a;
    raw.submat(u, u) += r14 * (c - a44) * r14.t();
    raw.submat(s, s) = c;
    raw.submat(g, g) = g22;
    raw.submat(u, g) = regression(a.submat(u, g), a.submat(g, g)) * g22;
    raw.submat(u, s) = r14 * c;
    raw.submat(g, s) = regression(a.submat(g, s), a44) * c;
    raw.submat(g, u) = raw.submat(u, g).t();
    raw.submat(s, u) = raw.submat(u, s).t();
    raw.submat(s, g) = raw.submat(g, s).t();
  }
  raw = (raw + raw.t()) / 2;
  mat k = adjusted(raw, note);
  return {std::move(raw), std::move(k), note};
}
} // namespace fsb
