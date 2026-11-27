#include "cache.hpp"
#include "omics.hpp"
#include <fstream>
#include <iomanip>
#include <numeric>
#include <openssl/evp.h>
#include <queue>
namespace fsb {
namespace {
enum class ReadPhase { calibration, preload, training, cross };
thread_local ReadPhase read_phase = ReadPhase::calibration;
struct ScopedReadPhase {
  ReadPhase old = read_phase;
  explicit ScopedReadPhase(ReadPhase next) { read_phase = next; }
  ~ScopedReadPhase() { read_phase = old; }
};
string digest(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  require(bool(in), "cannot read " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),
                                                              EVP_MD_CTX_free);
  require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1,
          "SHA256 init");
  char buf[65536];
  while (in) {
    in.read(buf, sizeof(buf));
    require(EVP_DigestUpdate(ctx.get(), buf, in.gcount()) == 1,
            "SHA256 update");
  }
  require(in.eof(), "cache read error");
  unsigned char bytes[EVP_MAX_MD_SIZE];
  unsigned len = 0;
  require(EVP_DigestFinal_ex(ctx.get(), bytes, &len) == 1, "SHA256 finalize");
  std::ostringstream out;
  for (unsigned j = 0; j < len; ++j)
    out << std::hex << std::setfill('0') << std::setw(2) << unsigned(bytes[j]);
  return out.str();
}
using Weights = std::map<size_t, double, std::greater<size_t>>;
Weights ancestors(long parent, const std::vector<long> &sire,
                  const std::vector<long> &dam) {
  Weights result, pending;
  pending[parent] = 1;
  while (!pending.empty()) {
    auto [i, w] = *pending.begin();
    pending.erase(pending.begin());
    result[i] = w;
    for (long p : {sire[i], dam[i]})
      if (p >= 0)
        pending[p] += .5 * w;
  }
  return result;
}
struct Moment {
  size_t count[3]{}, missing = 0;
  OmicsMoment omics;
};
void pack_features(const fs::path &input, const fs::path &dir, string name,
                   bool snp, bool fpkm, double cap, bool bits) {
  Ids ids, header;
  std::vector<Moment> moments;
  auto scan = [&](auto head, auto process) {
    if (snp)
      scan_genotypes(input, head,
                     [&](const string &id, const std::vector<uint8_t> &code) {
                       process(id, [&](size_t j) {
                         return code[j] == 3 ? double(NAN) : double(code[j]);
                       });
                     });
    else
      scan_table(input, head, [&](const Ids &row) {
        process(row[0], [&](size_t j) {
          return missing(row[j + 1]) ? double(NAN) : number(row[j + 1]);
        });
      });
  };
  scan(
      [&](auto &h) {
        header = h;
        require(h.size() > 1 && h.size() * 160. / 1048576. < cap,
                "feature header exceeds memory budget");
        moments.resize(h.size() - 1);
        unique_ids(Ids(h.begin() + 1, h.end()));
      },
      [&](const string &id, auto value) {
        ids.push_back(id);
        size_t n = ids.size();
        require((n * 160. + header.size() * 160.) / 1048576. < cap,
                "feature metadata exceeds memory budget");
        for (size_t j = 0; j < moments.size(); ++j) {
          auto &m = moments[j];
          double v = value(j);
          if (std::isnan(v) || (snp && v == -9)) {
            ++m.missing;
            if (snp)
              continue;
            v = 0;
          }
          if (snp) {
            require(v == 0 || v == 1 || v == 2, "invalid genotype");
            ++m.count[int(v)];
          } else {
            m.omics.add(v);
          }
        }
      });
  unique_ids(ids);
  size_t n = ids.size();
  require(n >= 2, "two or more feature rows required");
  std::vector<size_t> keep;
  std::vector<double> impute, center;
  double denom = 0;
  write_file(dir / (name + ".stats.tsv"), [&](auto &out) {
    out << "feature\tkeep\tmissing\timpute\tcenter\tdivisor\tp\tq\n";
    for (size_t j = 0; j < moments.size(); ++j) {
      auto &m = moments[j];
      bool retain = true;
      double fill = 0, mu = snp ? 0 : m.omics.center(), sd = 1, p = NAN,
             q = NAN;
      if (snp) {
        retain = m.missing <= n / 2;
        if (retain) {
          fill = (2. * m.count[2] + m.count[1]) / (n - m.missing);
          if (fill == 0 || fill == 1 || fill == 2)
            m.count[int(fill)] += m.missing;
          p = (2. * m.count[2] + m.count[1]) / (2 * n);
          q = (2. * m.count[0] + m.count[1]) / (2 * n);
          mu = 2 * p;
          denom += 2 * p * q;
        }
      } else {
        sd = m.omics.sd();
        retain = sd > 0 && !m.omics.filtered(fpkm);
      }
      if (retain) {
        keep.push_back(j);
        impute.push_back(fill);
        center.push_back(mu);
      }
      out << field(header[j + 1]) << '\t' << retain << '\t' << m.missing << '\t'
          << fill << '\t' << mu << '\t' << sd << '\t' << p << '\t' << q << '\n';
    }
  });
  if (!snp)
    denom = keep.size();
  require(!keep.empty() && denom > 0, "no usable " + name + " features");
  vec diag(n);
  size_t i = 0;
  std::vector<double> packed(keep.size());
  std::vector<uint8_t> codes((keep.size() + 3) / 4);
  double normalization = std::sqrt(denom);
  // Store normalized row-major doubles. Linux binary output uses the same
  // checked close as TSV.
  write_file(dir / (name + ".bin"), [&](auto &out) {
    scan([&](auto &h) { require(h == header, "feature header changed"); },
         [&](const string &id, auto value) {
           require(i < n && id == ids[i], "feature IDs changed during prepare");
           double sum = 0;
           if (bits)
             std::fill(codes.begin(), codes.end(), 0);
           for (size_t j = 0; j < keep.size(); ++j) {
             double v = value(keep[j]);
             if (bits)
               codes[j / 4] |= (std::isnan(v) ? 3 : int(v)) << (2 * (j % 4));
             if (std::isnan(v) || (snp && v == -9))
               v = impute[j];
             v = snp ? (v - center[j]) / normalization
                     : moments[keep[j]].omics.standardized(v) / normalization;
             if (!std::isfinite(v))
               throw std::runtime_error("nonfinite standardized feature");
             packed[j] = v;
             sum += v * v;
           }
           diag[i++] = sum;
           if (bits)
             out.write(reinterpret_cast<const char *>(codes.data()),
                       codes.size());
           else
             out.write(reinterpret_cast<const char *>(packed.data()),
                       packed.size() * sizeof(double));
         });
    require(i == n, "feature input truncated");
  });
  write_file(dir / (name + ".ids.tsv"), [&](auto &f) {
    f << "ID\tdiagonal\n";
    for (size_t j = 0; j < n; ++j)
      f << field(ids[j]) << '\t' << diag[j] << '\n';
  });
  write_file(dir / (name + ".shape"), [&](auto &f) {
    f << n << ' ' << keep.size() << ' ' << denom << ' '
      << (bits ? "2bit" : "f64") << '\n';
  });
}
const std::vector<string> files = {
    "ped.tsv", "G.bin",   "G.shape",   "G.ids.tsv",   "G.stats.tsv",
    "O.bin",   "O.shape", "O.ids.tsv", "O.stats.tsv", "sources.tsv"};
PedigreeOperator load_pedigree(const fs::path &dir, double cap,
                               uint64_t &verified) {
  size_t rows = 0;
  scan_table(
      dir / "ped.tsv", [](auto &) {},
      [&](auto &) {
        require(++rows * 1024. / 1048576. < cap,
                "cache metadata exceeds --memory-mb");
      });

  std::ifstream f(dir / "manifest");
  string line;
  require(std::getline(f, line) &&
              (line == "fsblup-cache 1 little-endian f64" ||
               line == "fsblup-cache 2 little-endian" ||
               line == "fsblup-cache 3 little-endian"),
          "invalid cache version");
  bool topology = line == "fsblup-cache 3 little-endian";
  auto verify = [&](const string &file) {
    string hash, key;
    require(bool(f >> hash >> key) && key == file && hash == digest(dir / file),
            "cache integrity failure: " + file);
    verified += fs::file_size(dir / file);
  };
  for (const auto &file : files)
    verify(file);
  if (topology)
    verify("topology.tsv");
  require(!(f >> line), "unexpected cache manifest entries");
  if (!topology)
    return PedigreeOperator(read_table(dir / "ped.tsv"));
  PedigreeOperator ped;
  ped.diag.set_size(rows);
  ped.d.set_size(rows);
  scan_table(
      dir / "topology.tsv",
      [](const Ids &h) {
        require(h == Ids{"ID", "sire_index", "dam_index", "D", "Aii"},
                "invalid topology header");
      },
      [&](const Ids &r) {
        size_t j = ped.ids.size();
        long sire = integer(r[1], -1), dam = integer(r[2], -1);
        require(j < rows && sire < long(j) && dam < long(j),
                "invalid topology parent index");
        ped.ids.push_back(r[0]);
        ped.sire.push_back(sire);
        ped.dam.push_back(dam);
        ped.d[j] = number(r[3]);
        ped.diag[j] = number(r[4]);
        double expected = 1 - .25 * ((sire < 0 ? 0 : ped.diag[sire]) +
                                     (dam < 0 ? 0 : ped.diag[dam]));
        require(ped.diag[j] >= 1 && ped.diag[j] <= 2 && ped.d[j] >= 0 &&
                    ped.d[j] == expected,
                "invalid cached pedigree diagonal or Mendelian variance");
      });
  require(ped.ids.size() == rows, "incomplete topology");
  unique_ids(ped.ids);
  return ped;
}
FeatureFile load_features(const fs::path &dir, string layer, double cap) {
  FeatureFile f;
  f.path = dir / (layer + ".bin");
  std::ifstream shape(dir / (layer + ".shape"));
  size_t n = 0;
  double denom;
  require(bool(shape >> n >> f.columns >> denom) && n && f.columns && denom > 0,
          "invalid cache shape");
  string codec = "f64";
  shape >> codec;
  require(codec == "f64" || (codec == "2bit" && layer == "G"),
          "invalid cache codec");
  f.stride = codec == "2bit" ? (f.columns + 3) / 4 : f.columns * 8;
  require(fs::file_size(f.path) / n == f.stride &&
              fs::file_size(f.path) % n == 0,
          "invalid cached dimensions");
  if (codec == "2bit") {
    require((32. * f.columns + 160. * n) / 1048576. < cap,
            "SNP dictionary exceeds --memory-mb");
    f.lookup.set_size(4, f.columns);
    size_t j = 0, keep = 0, impute = 0, center = 0;
    scan_table(
        dir / "G.stats.tsv",
        [&](const Ids &h) {
          Table header{h, {}};
          keep = header.col("keep");
          impute = header.col("impute");
          center = header.col("center");
        },
        [&](const Ids &row) {
          if (row[keep] == "0")
            return;
          require(row[keep] == "1" && j < f.columns, "invalid SNP dictionary");
          double fill = number(row[impute]), mu = number(row[center]);
          for (size_t k = 0; k < 4; ++k)
            f.lookup(k, j) =
                ((k == 3 ? fill : double(k)) - mu) / std::sqrt(denom);
          ++j;
        });
    require(j == f.columns, "incomplete SNP dictionary");
  }
  auto t = read_table(dir / (layer + ".ids.tsv"));
  f.ids = t.column("ID");
  unique_ids(f.ids);
  require(f.ids.size() == n, "invalid cache ID count");
  f.diag.set_size(n);
  for (size_t j = 0; j < n; ++j)
    f.diag[j] = number(t.rows[j][1]);
  return f;
}
} // namespace
PedigreeOperator::PedigreeOperator(const Table &input) {
  Table p = input;
  for (auto &h : p.header)
    for (char &c : h)
      c = std::tolower(static_cast<unsigned char>(c));
  auto names = p.column("id"), ss = p.column("sire"), dd = p.column("dam");
  unique_ids(names);
  std::map<string, size_t> index;
  for (size_t i = 0; i < names.size(); ++i) {
    require(names[i] != "0", "0 reserved");
    index[names[i]] = i;
  }
  for (auto parents : {&ss, &dd})
    for (auto &id : *parents) {
      require(!missing(id), "unknown parent must be 0");
      if (id != "0" && !index.count(id)) {
        index[id] = names.size();
        names.push_back(id);
      }
    }
  size_t n = names.size();
  std::vector<std::vector<size_t>> child(n);
  std::vector<int> deg(n);
  for (size_t i = 0; i < ss.size(); ++i)
    for (auto parent : {ss[i], dd[i]})
      if (parent != "0") {
        child[index.at(parent)].push_back(i);
        ++deg[i];
      }
  std::queue<size_t> ready;
  for (size_t i = 0; i < n; ++i)
    if (!deg[i])
      ready.push(i);
  std::vector<long> topo(n, -1);
  diag.set_size(n);
  d.set_size(n);
  while (!ready.empty()) {
    size_t orig = ready.front();
    ready.pop();
    size_t i = ids.size();
    topo[orig] = i;
    ids.push_back(names[orig]);
    long s = -1, m = -1;
    if (orig < ss.size()) {
      if (ss[orig] != "0")
        s = topo[index.at(ss[orig])];
      if (dd[orig] != "0")
        m = topo[index.at(dd[orig])];
    }
    sire.push_back(s);
    dam.push_back(m);
    d[i] = 1 - .25 * ((s >= 0 ? diag[s] : 0) + (m >= 0 ? diag[m] : 0));
    diag[i] = 1;
    if (s >= 0 && m >= 0) {
      auto u = ancestors(s, sire, dam), v = ancestors(m, sire, dam);
      double r = 0;
      for (auto [j, w] : u) {
        auto it = v.find(j);
        if (it != v.end())
          r += w * d[j] * it->second;
      }
      diag[i] += .5 * r;
    }
    require(d[i] >= 0, "negative Mendelian sampling variance");
    for (auto c : child[orig])
      if (--deg[c] == 0)
        ready.push(c);
  }
  require(ids.size() == n, "cycle in pedigree");
}
mat PedigreeOperator::apply(mat x) const {
  for (size_t col = 0; col < x.n_cols; ++col) {
    auto v = x.colptr(col);
    for (size_t i = ids.size(); i-- > 0;) {
      double h = .5 * v[i];
      if (sire[i] >= 0)
        v[sire[i]] += h;
      if (dam[i] >= 0)
        v[dam[i]] += h;
    }
    for (size_t i = 0; i < ids.size(); ++i) {
      v[i] *= d[i];
      if (sire[i] >= 0)
        v[i] += .5 * v[sire[i]];
      if (dam[i] >= 0)
        v[i] += .5 * v[dam[i]];
    }
  }
  return x;
}
std::function<mat(const mat &)>
PedigreeOperator::restricted(const uvec &rows) const {
  if (rows.n_elem == ids.size() &&
      arma::all(rows == arma::regspace<uvec>(0, ids.size() - 1)))
    return [this](const mat &v) { return apply(v); };
  std::vector<long> index(ids.size(), -1);
  for (auto j : rows)
    index[j] = 0;
  for (size_t j = ids.size(); j-- > 0;)
    if (index[j] >= 0) {
      if (sire[j] >= 0)
        index[sire[j]] = 0;
      if (dam[j] >= 0)
        index[dam[j]] = 0;
    }
  std::vector<long> ss, dd;
  std::vector<double> var;
  ss.reserve(rows.n_elem);
  dd.reserve(rows.n_elem);
  for (size_t j = 0; j < ids.size(); ++j)
    if (index[j] >= 0) {
      index[j] = ss.size();
      ss.push_back(sire[j] < 0 ? -1 : index[sire[j]]);
      dd.push_back(dam[j] < 0 ? -1 : index[dam[j]]);
      var.push_back(d[j]);
    }
  uvec ix(rows.n_elem);
  for (size_t j = 0; j < rows.n_elem; ++j)
    ix[j] = index[rows[j]];
  return [ix, ss = std::move(ss), dd = std::move(dd),
          var = std::move(var)](const mat &v) -> mat {
    mat z(ss.size(), v.n_cols, arma::fill::zeros);
    z.rows(ix) = v;
    for (size_t c = 0; c < v.n_cols; ++c) {
      auto p = z.colptr(c);
      for (size_t i = ss.size(); i-- > 0;) {
        double h = p[i] / 2;
        if (ss[i] >= 0)
          p[ss[i]] += h;
        if (dd[i] >= 0)
          p[dd[i]] += h;
      }
      for (size_t i = 0; i < ss.size(); ++i) {
        p[i] *= var[i];
        if (ss[i] >= 0)
          p[i] += .5 * p[ss[i]];
        if (dd[i] >= 0)
          p[i] += .5 * p[dd[i]];
      }
    }
    return z.rows(ix);
  };
}
void FeatureFile::read(std::ifstream &f, size_t rows, double *dest,
                       CacheIO &io) const {
  using Clock = std::chrono::steady_clock;
  std::vector<uint8_t> buf;
  if (!lookup.empty())
    buf.resize(rows * stride);
  auto tick = Clock::now();
  f.read(lookup.empty() ? reinterpret_cast<char *>(dest)
                        : reinterpret_cast<char *>(buf.data()),
         rows * stride);
  require(bool(f), "truncated cached features");
  auto split = Clock::now();
  io.read_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(split - tick)
          .count(),
      std::memory_order_relaxed);
  io.bytes.fetch_add(rows * stride, std::memory_order_relaxed);
  io.phase_bytes[static_cast<size_t>(read_phase)].fetch_add(
      rows * stride, std::memory_order_relaxed);
  ++io.reads;
  if (!lookup.empty()) {
    for (size_t i = 0; i < rows; ++i)
      for (size_t j = 0; j < columns; ++j)
        dest[i * columns + j] =
            lookup((buf[i * stride + j / 4] >> (2 * (j % 4))) & 3, j);
    io.decode_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               Clock::now() - split)
                               .count(),
                           std::memory_order_relaxed);
  }
}
mat FeatureFile::apply(const mat &x, size_t block, CacheIO &io,
                       const uvec *subset) const {
  std::ifstream f(path, std::ios::binary);
  require(bool(f), "unreadable feature cache");
  mat coeff(columns, x.n_cols, arma::fill::zeros), result(x.n_rows, x.n_cols);
  // A scattered training vector can have very few nonzero rows. Only those
  // rows contribute to W'v. The cross pass still returns every target row.
  uvec nz = arma::find(arma::any(x != 0, 1));
  for (int pass = 0; pass < 2; ++pass) {
    const bool sparse = pass == 0 && nz.n_elem < x.n_rows / 4;
    const size_t count = sparse ? nz.n_elem : x.n_rows;
    for (size_t first = 0; first < count; first += block) {
      size_t len = std::min(block, count - first);
      auto source = [&](size_t i) {
        auto j = sparse ? nz[first + i] : first + i;
        return subset ? (*subset)[j] : j;
      };
      mat w(columns, len, arma::fill::none);
      for (size_t j = 0; j < len;) {
        size_t end = j + 1;
        auto start = source(j);
        while (end < len && source(end) == start + end - j)
          ++end;
        f.seekg(start * stride);
        read(f, end - j, w.colptr(j), io);
        j = end;
      }
      if (pass == 1)
        result.rows(first, first + len - 1) = w.t() * coeff;
      else if (sparse)
        coeff += w * x.rows(nz.subvec(first, first + len - 1));
      else
        coeff += w * x.rows(first, first + len - 1);
    }
  }
  return result;
}
void prepare_cache(const fs::path &pedigree, const fs::path &snp,
                   const fs::path &omics, bool fpkm, const fs::path &dir,
                   double cap, bool two_bit) {
  require(cap > 0, "positive memory budget required");
  auto orig = std::vector<fs::path>{pedigree, snp, omics};
  std::vector<string> hashes;
  for (auto &f : orig)
    hashes.push_back(digest(f));
  size_t lines = 0;
  scan_table(
      pedigree, [](auto &) {},
      [&](auto &) {
        require(++lines * 1024. / 1048576. < cap,
                "pedigree metadata exceeds --memory-mb");
      });
  auto p = read_table(pedigree);
  PedigreeOperator op(p);
  write_file(dir / "ped.tsv", [&](auto &f) {
    f << "ID\tSire\tDam\n";
    for (size_t i = 0; i < op.ids.size(); ++i)
      f << field(op.ids[i]) << '\t'
        << (op.sire[i] < 0 ? "0" : field(op.ids[op.sire[i]])) << '\t'
        << (op.dam[i] < 0 ? "0" : field(op.ids[op.dam[i]])) << '\n';
  });
  write_file(dir / "topology.tsv", [&](auto &f) {
    f << "ID\tsire_index\tdam_index\tD\tAii\n";
    for (size_t i = 0; i < op.ids.size(); ++i)
      f << field(op.ids[i]) << '\t' << op.sire[i] << '\t' << op.dam[i] << '\t'
        << op.d[i] << '\t' << op.diag[i] << '\n';
  });
  pack_features(snp, dir, "G", true, false, cap, two_bit);
  pack_features(omics, dir, "O", false, fpkm, cap, false);
  write_file(dir / "sources.tsv", [&](auto &f) {
    f << "path\tsha256\tomics_type\n";
    for (size_t i = 0; i < orig.size(); ++i) {
      require(digest(orig[i]) == hashes[i], "source changed during prepare");
      f << field(orig[i].string()) << '\t' << hashes[i] << '\t'
        << (fpkm ? "fpkm" : "generic") << '\n';
    }
  });
  write_file(dir / "manifest", [&](auto &f) {
    f << "fsblup-cache 3 little-endian\n";
    for (auto &file : files)
      f << digest(dir / file) << ' ' << file << '\n';
    f << digest(dir / "topology.tsv") << " topology.tsv\n";
  });
}
Cache::Cache(const fs::path &dir, double cap)
    : ped(load_pedigree(dir, cap, verified_bytes)),
      g(load_features(dir, "G", cap)), o(load_features(dir, "O", cap)) {
  coverage = Layers{{ped.ids, {}}, {g.ids, {}}, {o.ids, {}}}.mode();
}
Layers Cache::materialize(double cap) {
  budget(ped.ids.size(), std::max(g.columns, o.columns), cap);
  return {{ped.ids, ped.apply(arma::eye(ped.ids.size(), ped.ids.size()))},
          {g.ids, g.apply(arma::eye(g.ids.size(), g.ids.size()), 64, io)},
          {o.ids, o.apply(arma::eye(o.ids.size(), o.ids.size()), 64, io)}};
}
namespace {
using Apply = std::function<mat(const mat &)>;
Apply subset_apply(size_t whole, uvec ix, Apply apply) {
  if (ix.n_elem == whole && arma::all(ix == arma::regspace<uvec>(0, whole - 1)))
    return apply;
  return [whole, ix, apply](const mat &x) -> mat {
    mat padded(whole, x.n_cols, arma::fill::zeros);
    padded.rows(ix) = x;
    mat y = apply(padded);
    return y.rows(ix);
  };
}
// The cache is row-major on disk. Coalesce runs without reordering IDs.
mat feature_rows(const FeatureFile &f, std::ifstream &in, const uvec &ix,
                 CacheIO &io) {
  mat w(f.columns, ix.n_elem);
  for (size_t j = 0; j < ix.n_elem;) {
    size_t end = j + 1;
    while (end < ix.n_elem && ix[end] == ix[j] + end - j)
      ++end;
    in.seekg(ix[j] * f.stride);
    f.read(in, end - j, w.colptr(j), io);
    j = end;
  }
  return w;
}
ProductPair feature_products(const FeatureFile &f, const uvec &train,
                             const uvec &targets, const uvec &local_train,
                             size_t block, bool keep, CacheIO &io) {
  std::shared_ptr<const mat> packed;
  if (keep) {
    std::ifstream in(f.path, std::ios::binary);
    auto stored = std::make_shared<mat>(f.columns, train.n_elem);
    for (size_t j = 0; j < train.n_elem; j += block) {
      size_t end = std::min(j + block, size_t(train.n_elem)) - 1;
      stored->cols(j, end) = feature_rows(f, in, train.subvec(j, end), io);
    }
    packed = std::move(stored);
  }
  auto blocks = [&f, block, &io](const uvec &ix, const auto &visit) {
    std::ifstream in(f.path, std::ios::binary);
    require(bool(in), "cannot open feature cache");
    for (size_t i = 0; i < ix.n_elem; i += block) {
      size_t end = std::min(i + block, size_t(ix.n_elem)) - 1;
      visit(i, end, feature_rows(f, in, ix.subvec(i, end), io));
    }
  };
  auto coefficients = [packed, blocks, train,
                       m = f.columns](const mat &v) -> mat {
    if (packed)
      return (*packed) * v;
    mat z(m, v.n_cols, arma::fill::zeros);
    blocks(train, [&](size_t i, size_t end, const mat &w) {
      z += w * v.rows(i, end);
    });
    return z;
  };
  auto product = [packed, blocks, train](const mat &z) -> mat {
    if (packed)
      return packed->t() * z;
    mat result(train.n_elem, z.n_cols);
    blocks(train, [&](size_t i, size_t end, const mat &w) {
      result.rows(i, end) = w.t() * z;
    });
    return result;
  };
  uvec mask(targets.n_elem, arma::fill::ones);
  mask.elem(local_train).zeros();
  uvec other = arma::find(mask), rows = targets.elem(other);
  return {[coefficients, product](const mat &v) {
            return product(coefficients(v));
          },
          [coefficients, product, local_train, other, rows, blocks,
           n = targets.n_elem](const mat &v) -> mat {
            mat z = coefficients(v), out(n, v.n_cols);
            out.rows(local_train) = product(z);
            blocks(rows, [&](size_t i, size_t end, const mat &w) {
              out.rows(other.subvec(i, end)) = w.t() * z;
            });
            return out;
          }};
}
mat columns(const Apply &apply, size_t n, const uvec &idx, size_t batch) {
  mat result(n, idx.n_elem);
  for (size_t j = 0; j < idx.n_elem; j += batch) {
    size_t w = std::min(batch, size_t(idx.n_elem) - j);
    mat x(n, w, arma::fill::zeros);
    for (size_t k = 0; k < w; ++k)
      x(idx[j + k], k) = 1;
    result.cols(j, j + w - 1) = apply(x);
  }
  return result;
}
void check_space(double bytes, double cap, const string &purpose) {
  if (bytes / 1048576. > cap)
    throw CertificationUnavailable(
        purpose + " needs " + std::to_string(bytes / 1048576.) +
        " MiB; raise --memory-mb or reduce the conditioning group");
}
} // namespace
FeatureWorkspace feature_workspace(size_t genomic, size_t omics, size_t train,
                                   size_t targets, size_t rhs, bool stream,
                                   bool resident, double available) {
  size_t width = std::max(genomic, omics);
  double stored = 8. * train * (double(genomic) + omics);
  double fixed = 8. * rhs * (width + 4. * targets) + 64. * targets;
  double row = 24. * width;
  double minimum = fixed + row + (resident ? stored : 0.);
  bool keep = resident || (!stream && stored <= available / 4 &&
                           fixed + row + stored <= available);
  double base = fixed + (keep ? stored : 0.);
  size_t block = width && available >= base + row
                     ? size_t(std::min({2048., double(targets),
                                        std::floor((available - base) / row)}))
                     : 1;
  return {block, keep, minimum, base + row * block};
}
TrainKernel PopulationKernel::restrict(const Ids &targets, const uvec &train,
                                       double cap, double solver_mib,
                                       size_t rhs_panel) const {
  ScopedReadPhase setup(ReadPhase::preload);
  cap -= retained_mib;
  if (cap <= solver_mib)
    throw CertificationUnavailable(
        "training and feature workspaces exceed --memory-mb");
  uvec ix = positions(ids, targets), tr = ix.elem(train);
  Product training, cross;
  if (common_products) {
    auto pair = common_products(tr, ix, train, cap - solver_mib, rhs_panel);
    training = std::move(pair.train);
    cross = std::move(pair.cross);
  } else {
    training = train_apply ? train_apply(tr, cap - solver_mib)
                           : subset_apply(ids.size(), tr, apply);
    cross = [fn = apply, ix, tr, n = ids.size()](const mat &x) -> mat {
      mat z(n, x.n_cols, arma::fill::zeros);
      z.rows(tr) = x;
      mat res = fn(z);
      return res.rows(ix);
    };
  }
  training = [apply = std::move(training)](const mat &v) {
    ScopedReadPhase stage(ReadPhase::training);
    return apply(v);
  };
  cross = [apply = std::move(cross)](const mat &v) {
    ScopedReadPhase stage(ReadPhase::cross);
    return apply(v);
  };
  auto dense = [training, n = train.n_elem, whole = ids.size(), cap]() -> mat {
    check_space(160. * n * n + 256. * whole, cap,
                "exact training spectrum/direct factor; alternatively use "
                "fixed PCG or Hutchinson");
    size_t batch = std::max(
        1., std::min(32., (cap * 1048576. - 160. * n * n) / (96. * whole)));
    mat k = columns(training, n, arma::regspace<uvec>(0, n - 1), batch);
    return .5 * k + .5 * k.t();
  };
  Ids keys;
  for (auto i : train)
    keys.push_back(targets[i]);
  return {diagonal.elem(tr), training, cross,       dense,
          std::move(keys),   nullptr,  train.n_elem};
}
PopulationKernel Cache::prepare(const Ids &pop, double a, double b,
                                PdPolicy policy, double cap) {
  auto pa = positions(ped.ids, pop);
  auto mode_name = mode();
  size_t n = pop.size(), whole = ped.ids.size(),
         m = std::max(g.columns, o.columns);
  check_space(200. * whole + 128. * n + 64. * m, cap, "feature operator");
  size_t block = std::max(1., std::min(2048., cap * 1048576. / (64. * m)));
  Apply ap = ped.restricted(pa);
  vec ad = ped.diag.elem(pa);
  auto finish = [&](vec diagonal, Apply raw,
                    std::vector<PdBlock> pd = std::vector<PdBlock>{{"K"}}) {
    auto immutable = std::make_shared<const Apply>(std::move(raw));
    PopulationKernel result{
        pop, diagonal + 1e-6,
        [immutable](const mat &x) -> mat { return (*immutable)(x) + 1e-6 * x; },
        std::move(pd)};
    result.retained_mib = (320. * whole + 128. * n + 64. * m) / 1048576.;
    return result;
  };
  if (mode_name == "B" && b == 0)
    return finish(ad, ap);
  Ids gids = intersect(pop, g.ids), oids = intersect(pop, o.ids);
  require(!oids.empty(), "reference coverage degenerates: no omics");
  auto pg = positions(g.ids, gids), po = positions(o.ids, oids);
  Apply gp = [this, pg, block](const mat &x) {
    return g.apply(x, block, io, &pg);
  };
  Apply op = [this, po, block](const mat &x) {
    return o.apply(x, block, io, &po);
  };
  vec gd = g.diag.elem(pg), od = o.diag.elem(po);
  auto indexed = [&](double pedigree_weight, double b, double a) {
    return [this, pa, pg, po, pedigree_weight, a,
            b](const uvec &tr, const uvec &ix, const uvec &local, double limit,
               size_t rhs_panel) {
      auto plan = feature_workspace(b ? g.columns : 0, a ? o.columns : 0,
                                    tr.n_elem, ix.n_elem, rhs_panel, streaming,
                                    resident, limit * 1048576.);
      check_space(plan.planned_bytes, limit,
                  "training features and cross workspace");
      size_t chunk = plan.block;
      bool keep = plan.keep;
      ProductPair A, G, O;
      if (pedigree_weight != 0) {
        A.train = ped.restricted(pa.elem(tr));
        auto ap = ped.restricted(pa.elem(ix));
        A.cross = [ap, local, n = ix.n_elem](const mat &v) -> mat {
          mat z(n, v.n_cols, arma::fill::zeros);
          z.rows(local) = v;
          return ap(z);
        };
      }
      if (b)
        G = feature_products(g, pg.elem(tr), pg.elem(ix), local, chunk, keep,
                             io);
      if (a)
        O = feature_products(o, po.elem(tr), po.elem(ix), local, chunk, keep,
                             io);
      auto combine = [a, b, pedigree_weight](const Apply &p, const Apply &g,
                                             const Apply &o, const mat &v,
                                             mat z) -> mat {
        if (p)
          z += (pedigree_weight)*p(v);
        if (g)
          z += b * g(v);
        if (o)
          z += a * o(v);
        return z;
      };
      return ProductPair{
          [combine, A, G, O](const mat &v) {
            return combine(A.train, G.train, O.train, v, 1e-6 * v);
          },
          [combine, A, G, O, local, n = ix.n_elem](const mat &v) {
            mat z(n, v.n_cols, arma::fill::zeros);
            z.rows(local) = 1e-6 * v;
            return combine(A.cross, G.cross, O.cross, v, std::move(z));
          }};
    };
  };
  if (mode_name == "A") {
    if (a < 0 || b < 0 || a + b > 1) {
      // Materialize only the requested weighted covariance. The three input
      // layers remain operators, including on the full-population check.
      double bytes = (policy == PdPolicy::strict ? 40. : 144.) * n * n +
                     320. * whole + 128. * n + 64. * m;
      if (bytes > cap * 1048576.)
        throw CertificationUnavailable(
            "signed-weight certification exceeds --memory-mb");
      size_t batch = std::max(
          1., std::min(32., (cap * 1048576. - bytes) / (128. * n + 64. * m)));
      size_t chunk = std::max(
          1., std::min(double(block), (cap * 1048576. - bytes) / (32. * m)));
      auto raw = [&](const mat &v) -> mat {
        mat y = (1 - a - b) * ap(v);
        if (b)
          y += b * g.apply(v, chunk, io, &pg);
        if (a)
          y += a * o.apply(v, chunk, io, &po);
        return y;
      };
      mat covariance = columns(raw, n, arma::regspace<uvec>(0, n - 1), batch);
      symmetrize(covariance, arma::abs(covariance).max());
      PdBlock pd{"K"};
      auto k = std::make_shared<const mat>(
          adjust_covariance(covariance, policy, PdDiagnostics::full, pd));
      PopulationKernel result{
          pop, k->diag(), [k](const mat &v) -> mat { return *k * v; }, {pd}};
      result.retained_mib =
          (8. * n * n + 320. * whole + 128. * n + 64. * m) / 1048576.;
      return result;
    }
    auto result = finish((1 - a - b) * ad + b * gd + a * od,
                         [ap, gp, op, a, b](const mat &v) -> mat {
                           mat z(v.n_rows, v.n_cols, arma::fill::zeros);
                           if (1 - a - b != 0)
                             z += (1 - a - b) * ap(v);
                           if (b != 0)
                             z += b * gp(v);
                           if (a != 0)
                             z += a * op(v);
                           return z;
                         });
    result.common_products = indexed(1 - a - b, b, a);
    return result;
  }
  auto s = positions(pop, oids);
  size_t ns = s.n_elem, ng = gids.size();
  auto dense_certificate = [&]() {
    check_space(192. * n * n + 320. * whole + 64. * m, cap,
                "whole-kernel certification");
    auto materialize = [&](const Apply &fn, size_t count) {
      return columns(fn, count, arma::regspace<uvec>(0, count - 1), 1);
    };
    Layers local{{pop, materialize(ap, n)},
                 {gids, materialize(gp, ng)},
                 {oids, materialize(op, ns)}};
    auto fused = FusionPlan(local).combine(a, b, nullptr, policy);
    auto k = std::make_shared<const mat>(std::move(fused.k));
    PopulationKernel result{pop, k->diag(),
                            [k](const mat &v) -> mat { return *k * v; },
                            std::move(fused.pd_blocks)};
    result.retained_mib =
        (8. * n * n + 320. * whole + 128. * n + 64. * m) / 1048576.;
    return result;
  };
  if (mode_name == "B") {
    if (b < 0 || b > 1)
      return dense_certificate();
    auto loading = std::make_shared<mat>(), delta = std::make_shared<mat>();
    vec joint(n, arma::fill::zeros);
    joint.elem(s).ones();
    vec hd = gd + 1e-6 * joint;
    if (a != 0) {
      check_space(96. * n * ns + 128. * ns * ns + 256. * whole, cap,
                  "B calibration workspace");
      mat cols = columns(gp, n, s, std::min<size_t>(8, ns));
      mat base = cols.rows(s);
      base.diag() += 1e-6;
      mat lower;
      if (!arma::chol(lower, base, "lower"))
        return dense_certificate();
      mat omics = columns(op, ns, arma::regspace<uvec>(0, ns - 1),
                          std::min<size_t>(8, ns));
      double om = arma::mean(od), gm = arma::mean(base.diag());
      double contrast = arma::mean(arma::mean(omics)) - om;
      require(std::abs(contrast) > 1e-14 * std::max(1., arma::abs(omics).max()),
              "unidentifiable BB000 omics calibration");
      double slope = (arma::mean(arma::mean(base)) - gm) / contrast;
      mat replacement =
              a * (slope * omics + (om - slope * gm)) + (1 - a) * base,
          check;
      if (!arma::chol(check, replacement))
        return dense_certificate();
      *delta = replacement - base;
      *loading = arma::solve(arma::trimatu(lower.t()),
                             arma::solve(arma::trimatl(lower), cols.t(),
                                         arma::solve_opts::no_approx),
                             arma::solve_opts::no_approx)
                     .t();
      loading->rows(s) = arma::eye(ns, ns);
      hd += arma::sum(((*loading) * (*delta)) % (*loading), 1);
    }
    double scale = arma::mean(hd);
    require(scale > 0, "nonpositive BB000 scale");
    double gw = b / scale;
    auto result =
        finish((1 - b) * ad + gw * hd,
               [ap, gp, b, gw, loading, delta, joint](const mat &v) -> mat {
                 mat h = gp(v) + 1e-6 * (v.each_col() % joint);
                 if (!delta->empty())
                   h += (*loading) * ((*delta) * (loading->t() * v));
                 return (1 - b) * ap(v) + gw * h;
               },
               {{"Gss"}, {"K"}});
    auto base_products = indexed(1 - b, gw, 0);
    result.retained_mib +=
        (8. * (loading->n_elem + delta->n_elem + n)) / 1048576.;
    result.common_products = [base_products, loading, delta, joint,
                              gw](const uvec &tr, const uvec &ix,
                                  const uvec &local, double limit, size_t rhs) {
      double scratch =
          8. * (tr.n_elem + ix.n_elem + 2. * rhs) * loading->n_cols;
      check_space(scratch, limit, "restricted B calibration");
      auto base = base_products(tr, ix, local, limit - scratch / 1048576., rhs);
      vec diag = 1e-6 * gw * joint.elem(tr);
      auto left = std::make_shared<mat>(), target = std::make_shared<mat>();
      if (!delta->empty()) {
        *left = loading->rows(tr);
        *target = loading->rows(ix);
      }
      return ProductPair{[fn = std::move(base.train), left, delta, diag,
                          gw](const mat &v) -> mat {
                           mat out = fn(v) + v.each_col() % diag;
                           if (!delta->empty())
                             out += gw * (*left) * ((*delta) * (left->t() * v));
                           return out;
                         },
                         [fn = std::move(base.cross), left, target, delta, diag,
                          gw, local](const mat &v) -> mat {
                           mat out = fn(v);
                           out.rows(local) += v.each_col() % diag;
                           if (!delta->empty())
                             out +=
                                 gw * (*target) * ((*delta) * (left->t() * v));
                           return out;
                         }};
    };
    return result;
  }
  // Pedigree-only connected components have zero A cross-covariance with
  // every measured individual. Equation 3 leaves those components equal to A.
  std::vector<size_t> root(whole);
  std::iota(root.begin(), root.end(), 0);
  auto find = [&](size_t i) {
    while (root[i] != i) {
      root[i] = root[root[i]];
      i = root[i];
    }
    return i;
  };
  for (size_t i = 0; i < whole; ++i)
    for (long parent : {ped.sire[i], ped.dam[i]})
      if (parent >= 0)
        root[find(i)] = find(parent);
  std::set<size_t> included;
  for (auto i : positions(ped.ids, gids))
    included.insert(find(i));
  std::vector<arma::uword> active;
  Ids names;
  for (size_t j = 0; j < n; ++j)
    if (included.count(find(pa[j]))) {
      active.push_back(j);
      names.push_back(pop[j]);
    }
  size_t na = names.size();
  check_space(192. * na * na + 320. * whole + 128. * ng * ng, cap,
              "mode C measured connected components");
  uvec ia(active);
  auto sub = subset_apply(n, ia, ap);
  mat ac = columns(sub, na, arma::regspace<uvec>(0, na - 1),
                   std::min<size_t>(4, na));
  mat gg = columns(gp, ng, arma::regspace<uvec>(0, ng - 1), 4),
      oo = columns(op, ns, arma::regspace<uvec>(0, ns - 1), 4);
  Layers layers{
      {names, std::move(ac)}, {gids, std::move(gg)}, {oids, std::move(oo)}};
  Fusion fusion;
  try {
    fusion = FusionPlan(layers, true)
                 .combine(a, b, nullptr, policy,
                          na == n ? PdDiagnostics::full
                                  : PdDiagnostics::certification);
  } catch (const std::runtime_error &ex) {
    if (string(ex.what()).find("complete-population nearPD required") !=
        string::npos)
      return dense_certificate();
    throw std::runtime_error(
        "mode C: " + string(ex.what()) +
        "; full dense MiB=" + std::to_string(160. * n * n / 1048576.) +
        "; use dense backend within budget");
  }
  require(
      !fusion.pd_blocks.back().projected || na == n,
      "mode C requires nearPD on complete population, estimated dense MiB=" +
          std::to_string(160. * n * n / 1048576.) +
          "; use dense backend within budget");
  mat difference = fusion.k - layers.a.values;
  difference.diag() -= 1e-6;
  vec diag = ad;
  diag.elem(ia) += difference.diag();
  auto delta = std::make_shared<const mat>(std::move(difference));
  auto result = finish(
      diag,
      [ap, ia, delta](const mat &x) -> mat {
        mat v = ap(x);
        v.rows(ia) += (*delta) * x.rows(ia);
        return v;
      },
      std::move(fusion.pd_blocks));
  result.active_count = na;
  result.retained_mib += 8. * na * na / 1048576.;
  result.train_apply = [this, pa, ia, delta](const uvec &tr, double) -> Apply {
    auto aop = ped.restricted(pa.elem(tr));
    std::map<size_t, size_t> local;
    for (size_t i = 0; i < ia.n_elem; ++i)
      local[ia[i]] = i;
    std::vector<arma::uword> active_rows, block_rows;
    for (size_t i = 0; i < tr.n_elem; ++i) {
      auto it = local.find(tr[i]);
      if (it != local.end()) {
        active_rows.push_back(i);
        block_rows.push_back(it->second);
      }
    }
    uvec ix(active_rows), br(block_rows);
    mat sub = delta->submat(br, br);
    return [aop, ix, sub = std::move(sub)](const mat &x) -> mat {
      mat y = aop(x) + 1e-6 * x;
      if (!ix.empty())
        y.rows(ix) += sub * x.rows(ix);
      return y;
    };
  };
  return result;
}
} // namespace fsb
