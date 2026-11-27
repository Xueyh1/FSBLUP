#include "cache.hpp"
#include <iostream>
using namespace fsb;
int main(int argc, char **argv) {
  try {
    require(argc == 4, "arguments: cache, built matrices, mode");
    Cache cache(argv[1], 256);
    fs::path d = argv[2];
    Layers l{read_kernel(d / "A.tsv", 256), read_kernel(d / "G.tsv", 256),
             read_kernel(d / "O.tsv", 256)};
    require(cache.mode() == argv[3], "coverage");
    Ids ids = l.a.ids;
    std::reverse(ids.begin(), ids.end());
    std::vector<std::pair<double, double>> cases = {{.02, .4}, {0, 0}};
    if (cache.mode() == "A")
      cases.emplace_back(-.01, -.01);
    for (auto weights : cases) {
      auto ref =
          fuse(l, weights.first, weights.second, nullptr, PdPolicy::near_pd);
      auto order = positions(l.a.ids, ids);
      mat expected = ref.k.submat(order, order);
      auto op = cache.prepare(ids, weights.first, weights.second,
                              PdPolicy::near_pd, 256);
      mat v(ids.size(), 3);
      for (size_t i = 0; i < v.n_elem; ++i)
        v[i] = std::sin(i + .37);
      require(arma::abs(expected * v - op.apply(v)).max() < 2e-8,
              "operator matrix product " + cache.mode());
      require(arma::abs(expected.diag() - op.diagonal).max() < 1e-8,
              "operator diagonal");
      auto train = arma::regspace<uvec>(0, 2, ids.size() - 1);
      auto sys = op.restrict(ids, train, 256);
      mat b = v.rows(train);
      require(
          arma::abs(sys.apply(b) - expected.submat(train, train) * b).max() <
              2e-8,
          "training product");
      require(arma::abs(sys.cross(b) - expected.cols(train) * b).max() < 2e-8,
              "cross product");
      if (cache.mode() == "A") {
        uvec subset = {2, 7, 0, 5, 3}, local = {3, 1, 0};
        Ids targets;
        for (auto row : subset)
          targets.push_back(ids[row]);
        for (bool stream : {false, true}) {
          cache.streaming = stream;
          auto partial = cache.prepare(ids, weights.first, weights.second,
                                       PdPolicy::near_pd, 256);
          auto test = partial.restrict(targets, local, 256);
          uvec t = subset.elem(local);
          require(arma::abs(test.cross(v.rows(t)) -
                            expected.submat(subset, t) * v.rows(t))
                          .max() < 2e-8,
                  "sparse permuted cross product");
        }
        cache.streaming = false;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
