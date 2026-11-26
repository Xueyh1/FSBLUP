#include "core.hpp"
#include <iostream>
using namespace fsb;
namespace {
void close(double x, double y, double tol = 1e-9) {
  require(std::abs(x - y) < tol,
          "numeric mismatch " + std::to_string(x) + " vs " + std::to_string(y));
}
template <class F> void rejects(F f) {
  try {
    f();
  } catch (const std::runtime_error &) {
    return;
  }
  throw std::runtime_error("expected rejection");
}
vec numbers(const Table &t, const string &col) {
  auto ss = t.column(col);
  vec out(ss.size());
  for (size_t i = 0; i < ss.size(); ++i)
    out[i] = missing(ss[i]) ? NAN : number(ss[i]);
  return out;
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 2, "need fixture directory");
    fs::path d = argv[1];
    auto load = [&](string s) { return read_kernel(d / (s + ".tsv"), 1024); };
    auto k = load("K");
    auto p = read_table(d / "pheno.tsv");
    vec y = numbers(p, "trait"), age = numbers(p, "age");
    auto herd = p.column("herd");
    mat x(y.n_elem, 3, arma::fill::ones);
    x.col(1) = age;
    for (size_t i = 0; i < herd.size(); ++i)
      x(i, 2) = herd[i] == "B";
    auto m = solve(k.values, x, y, arma::find_finite(y), .7);
    auto expected = read_table(d / "fixed_lambda.tsv");
    double err = arma::abs(m.u - numbers(expected, "u")).max();
    close(err, 0, 1e-8);
    close(
        arma::abs(m.b - numbers(read_table(d / "fixed_lambda_b.tsv"), "value"))
            .max(),
        0, 1e-8);
    std::cout << "fixed lambda max |u error| " << err << " ID "
              << k.ids[arma::abs(m.u - numbers(expected, "u")).index_max()]
              << "\n";
    // With K=I and a balanced group design, GLS estimates each group mean.
    // This exercises the high-rank design path using a closed-form BLUP.
    {
      constexpr size_t n = 120, groups = 40;
      mat design(n, groups, arma::fill::zeros);
      design.col(0).ones();
      vec response(n), deviation(n);
      for (size_t i = 0; i < n; ++i) {
        size_t group = i % groups;
        if (group)
          design(i, group) = 1.;
        deviation(i) = double(i / groups) - 1.;
        response(i) = double(group) + deviation(i);
      }
      auto wide = solve(arma::eye<mat>(n, n), design, response,
                        arma::regspace<uvec>(0, n - 1), .7);
      close(arma::abs(wide.u - deviation / 1.7).max(), 0, 1e-10);
      close(arma::abs(wide.yhat - (response - deviation + deviation / 1.7)).max(),
            0, 1e-10);
    }
    // Verify every BB000 intermediate using the reconstructed propagation and
    // mixture.
    Layers l{load("A"), load("G"), load("Ob")};
    auto s = positions(l.a.ids, l.o.ids);
    mat base = l.g.values.submat(s, s);
    base.diag() += 1e-6;
    close(arma::abs(base - load("B_base").values).max(), 0);
    auto h = load("B_H").values;
    BlockAudit audit;
    auto result = fuse(l, .2, .5, &audit);
    close(arma::abs(audit.base - load("B_base").values).max(), 0);
    close(arma::abs(audit.calibrated - load("B_cal").values).max(), 0);
    close(arma::abs(audit.replacement - load("B_T").values).max(), 0);
    close(arma::abs(audit.propagated - h).max(), 0, 1e-8);
    close(arma::abs(result.raw -
                    (.5 * h / arma::mean(h.diag()) + .5 * l.a.values))
              .max(),
          0);
    mat t = h.submat(s, s);
    close(arma::abs(t - load("B_T").values).max(), 0);
    close(arma::abs((t - .8 * base) / .2 - load("B_cal").values).max(), 0);
    auto refs = intersect(p.column("ID"), l.a.ids);
    refs.resize(20);
    auto rk = fuse(l.subset(refs), .2, .5);
    close(arma::abs(rk.k - load("B_Kref").values).max(), 0, 1e-8);
    auto idx = positions(l.a.ids, refs);
    require(arma::abs(rk.k - result.k.submat(idx, idx)).max() > 1e-5,
            "restriction before fusion");
    Layers singleton{l.a, l.g, l.o.subset({l.o.ids.front()})};
    auto endpoint = fuse(singleton, .5, 0, &audit);
    close(arma::abs(endpoint.raw - l.a.values).max(), 0);
    mat expected_endpoint = l.a.values;
    expected_endpoint.diag() += 1e-6;
    close(arma::abs(endpoint.k - expected_endpoint).max(), 0);
    require(audit.base.is_empty() && audit.calibrated.is_empty() &&
                audit.replacement.is_empty() && audit.propagated.is_empty(),
            "zero weight skips the propagated layer");
    rejects([&] { fuse(singleton, .5, .5); });
    mat v(20, 3, arma::fill::zeros);
    v(0, 0) = 2;
    v.col(1) = arma::linspace(1, 20, 20);
    v.col(2).fill(3);
    auto o = omics(v, true);
    require(o.filtered == 1 && o.constant == 1 && o.kept == 1,
            "95 percent filtering or single remaining feature");
    require(omics(v.col(1), true).kept == 1 && omics(v, false).kept == 2,
            "zero filtered features");
    rejects([&] { omics(v.col(0), true); });
    auto ped = pedigree({"c", "m"}, {"f", "0"}, {"m", "0"}, 100);
    close(ped.values(0, 2), .5);
    require(ped.ids[2] == "f", "implicit founder");
    rejects([] { pedigree({"1", "2"}, {"2", "1"}, {"0", "0"}, 100); });
    close(score({0, 1, 0, 1}, {.1, .1, .4, .8}, "auc"), .625);
    rejects([] { score({0, 2}, {1, 2}, "auc"); });
    auto folds = split({{1, {1, 1, 2, 2}}}, false, {0, 0, 0, 1}, "auc");
    require(!folds[0].reason.empty() && folds[1].reason.empty(),
            "undefined fold policy");
    FusionPlan plan(l);
    require(evaluate_weight(plan, .2, .5, x, y, y, {}, "rmse").reason ==
                "no validation folds",
            "empty folds should produce an ineligible candidate");
    auto search_result = search(false, true, [](double a, double b) {
      Evaluation e;
      e.value = (a - .375) * (a - .375) + (b - .375) * (b - .375);
      return e;
    });
    close(search_result.best, 0);
    close(search_result.alpha, .375);
    close(search_result.beta, .375);
    auto tied = search(true, false, [](double, double) {
      Evaluation e;
      e.value = 1;
      return e;
    });
    close(tied.alpha, 0);
    close(tied.beta, 0);
    for (auto &trial : tied.trials)
      require(trial.alpha >= 0 && trial.beta >= 0 &&
                  trial.alpha + trial.beta <= 1,
              "strict simplex");
    auto legacy = load("legacy_solver_K");
    auto old = solve(legacy.values, x.col(0), y, arma::find_finite(y));
    close(arma::abs(old.u - numbers(read_table(d / "legacy_u.tsv"), "u")).max(),
          0, 1e-5);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
