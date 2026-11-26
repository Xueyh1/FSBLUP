#include "core.hpp"
#include <charconv>
#include <fstream>
#include <iomanip>
#include <limits>
#include <unordered_map>
namespace fsb {
void require(bool ok, const string &message) {
  if (!ok)
    throw std::runtime_error(message);
}
bool missing(const string &s) { return s.empty() || s == "NA"; }
double number(const string &s) {
  size_t used = 0;
  double x;
  try {
    x = std::stod(s, &used);
  } catch (const std::exception &) {
    throw std::runtime_error("expected finite number: " + s);
  }
  if (used != s.size() || !std::isfinite(x))
    throw std::runtime_error("expected finite number: " + s);
  return x;
}
int integer(const string &s, int minimum) {
  double x = number(s);
  require(x == std::floor(x) && x >= minimum && x <= INT_MAX,
          "invalid integer: " + s);
  return static_cast<int>(x);
}
void unique_ids(const Ids &ids) {
  std::set<string> seen;
  for (auto &id : ids)
    require(!missing(id) && seen.insert(id).second,
            "missing or duplicate ID/name: " + id);
}
uvec positions(const Ids &source, const Ids &wanted) {
  std::unordered_map<string, size_t> index;
  for (size_t i = 0; i < source.size(); ++i)
    index.emplace(source[i], i);
  uvec pos(wanted.size());
  for (size_t i = 0; i < wanted.size(); ++i) {
    auto it = index.find(wanted[i]);
    require(it != index.end(), "ID outside support: " + wanted[i]);
    pos(i) = it->second;
  }
  return pos;
}
Ids intersect(const Ids &first, const Ids &second) {
  std::set<string> s(second.begin(), second.end());
  Ids out;
  for (auto &id : first)
    if (s.count(id))
      out.push_back(id);
  return out;
}
Kernel Kernel::subset(const Ids &wanted) const {
  auto idx = positions(ids, wanted);
  return {wanted, values.submat(idx, idx)};
}
namespace {
struct Reader {
  std::ifstream in;
  char delim;
  Ids header;
  size_t line_no = 0;
  string filename;
  Reader(const fs::path &path) : in(path), filename(path.string()) {
    string line;
    require(bool(std::getline(in, line)), "cannot read header: " + filename);
    delim = line.find('\t') != string::npos ? '\t' : ',';
    header = parse(line);
  }
  Ids parse(const string &line) {
    ++line_no;
    size_t length = line.size();
    if (length && line.back() == '\r')
      --length;
    Ids row;
    row.reserve(header.size());
    string cell;
    bool quoted = false, closed = false;
    for (size_t i = 0; i < length; ++i) {
      char c = line[i];
      if (quoted && c == '"' && i + 1 < length && line[i + 1] == '"') {
        cell += '"';
        ++i;
      } else if (c == '"' && !closed && (quoted || cell.empty())) {
        closed = quoted;
        quoted = !quoted;
      } else if (!quoted && c == delim) {
        row.push_back(cell);
        cell.clear();
        closed = false;
      } else {
        if (closed || (!quoted && c == '"'))
          throw std::runtime_error("invalid quoted field: " + filename);
        cell += c;
      }
    }
    require(!quoted, "unclosed quote: " + filename);
    row.push_back(cell);
    return row;
  }
  bool next(Ids &row) {
    string line;
    if (!std::getline(in, line)) {
      require(in.eof(), "read failure: " + filename);
      return false;
    }
    row = parse(line);
    require(row.size() == header.size(),
            "column count: " + filename + " line " + std::to_string(line_no));
    return true;
  }
};
} // namespace
size_t Table::col(const string &name) const {
  auto i = std::find(header.begin(), header.end(), name);
  require(i != header.end(), "unknown column: " + name);
  return i - header.begin();
}
Ids Table::column(const string &name) const {
  auto j = col(name);
  Ids v;
  for (auto &r : rows)
    v.push_back(r[j]);
  return v;
}
Table read_table(const fs::path &p) {
  Reader r(p);
  Table t{r.header, {}};
  if (t.header[0].empty())
    t.header[0] = "ID";
  unique_ids(t.header);
  Ids row;
  while (r.next(row))
    t.rows.push_back(std::move(row));
  require(!t.rows.empty(), "empty table: " + p.string());
  return t;
}
void write_file(const fs::path &path,
                const std::function<void(std::ostream &)> &write) {
  try {
    std::ofstream stream;
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.open(path);
    stream << std::setprecision(17);
    write(stream);
    // A disk error can first appear while closing the last buffered block.
    stream.close();
  } catch (const std::ios_base::failure &error) {
    throw std::runtime_error("cannot write " + path.string() + ": " + error.what());
  }
}
double budget(size_t n, size_t m, double cap) {
  double estimate = (20. * n * n * 8 + 64. * n * m) / 1048576.;
  require(cap > 0 && estimate <= cap,
          "estimated dense working set " + std::to_string(estimate) +
              " MiB exceeds --memory-mb " + std::to_string(cap));
  return estimate;
}
Kernel read_features(const fs::path &path, double cap) {
  Reader reader(path);
  size_t m = reader.header.size() - 1, n = 0;
  require(m > 0, "feature table must include feature columns");
  unique_ids(Ids(reader.header.begin() + 1, reader.header.end()));
  // Count samples before allocating a dense matrix. Only a single text row is
  // resident.
  string line;
  while (std::getline(reader.in, line))
    ++n;
  require(n > 0 && reader.in.eof(), "empty or unreadable features");
  budget(n, m, cap);
  reader.in.clear();
  reader.in.seekg(0);
  std::getline(reader.in, line);
  Ids ids, row;
  mat x(n, m);
  while (reader.next(row)) {
    size_t i = ids.size();
    require(i < n, "feature table changed while reading");
    ids.push_back(row[0]);
    for (size_t j = 0; j < m; ++j)
      x(i, j) = missing(row[j + 1]) ? NAN : number(row[j + 1]);
  }
  require(ids.size() == n, "feature table changed while reading");
  unique_ids(ids);
  return {ids, std::move(x)};
}
Kernel read_kernel(const fs::path &p, double cap) {
  Reader r(p);
  Ids cols(r.header.begin() + 1, r.header.end()), rows;
  unique_ids(cols);
  auto n = cols.size();
  require(n > 0, "empty kernel");
  budget(n, 0, cap);
  mat k(n, n);
  Ids row;
  while (r.next(row)) {
    size_t i = rows.size();
    require(i < n, "too many matrix rows");
    rows.push_back(row[0]);
    for (size_t j = 0; j < n; ++j)
      k(i, j) = number(row[j + 1]);
  }
  unique_ids(rows);
  require(rows.size() == n, "matrix is not square");
  if (cols != rows)
    k = k.cols(positions(cols, rows));
  double scale = std::max(1., arma::abs(k).max());
  require(arma::abs(k - k.t()).max() <= 1e-10 * scale,
          "asymmetric kernel: " + p.string());
  return {rows, (k + k.t()) / 2};
}
string field(const string &s) {
  if (s.find_first_of("\"\t,") == string::npos)
    return s;
  string out = "\"";
  for (auto c : s) {
    out += c;
    if (c == '\"')
      out += c;
  }
  return out + '\"';
}
void write_matrix(const fs::path &p, const Ids &ids, const mat &k) {
  write_file(p, [&](std::ostream &f) {
    f << "ID";
    for (const auto &id : ids)
      f << '\t' << field(id);
    f << '\n';
    // Format one row at a time, then reuse its capacity for the next row.
    string row;
    char number[64];
    for (size_t i = 0; i < ids.size(); ++i) {
      row.clear();
      row += field(ids[i]);
      for (size_t j = 0; j < ids.size(); ++j) {
        row.push_back('\t');
        const auto result = std::to_chars(number, number + sizeof(number), k(i, j),
                                          std::chars_format::general,
                                          std::numeric_limits<double>::max_digits10);
        if (result.ec != std::errc{})
          throw std::runtime_error("cannot format matrix value for " + p.string());
        row.append(number, result.ptr);
      }
      row.push_back('\n');
      f.write(row.data(), static_cast<std::streamsize>(row.size()));
    }
  });
}
} // namespace fsb
