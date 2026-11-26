#include "core.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <unistd.h>
using namespace fsb;
int main() {
  auto tmp = fs::temp_directory_path() /
             ("fsblup-roundtrip-" + std::to_string(getpid()));
  try {
    Ids ids = {"001", string(12000, 'x') + "\t\"", "003"};
    mat input(3, 3, arma::fill::zeros);
    input(0, 0) = std::numeric_limits<double>::max();
    input(1, 1) = -0.;
    input(2, 2) = std::numeric_limits<double>::min();
    input(0, 2) = input(2, 0) = std::numeric_limits<double>::denorm_min();
    write_matrix(tmp, ids, input);
    auto k = read_kernel(tmp, 100);
    require(k.ids == ids && arma::all(arma::vectorise(k.values == input)) &&
                std::signbit(k.values(1, 1)),
            "matrix roundtrip");
    auto fused = fuse({k, k, k}, 0, 0);
    require(fused.raw.is_finite() && fused.raw(0, 0) == input(0, 0),
            "finite symmetric endpoint");
    fs::remove(tmp);
    return 0;
  } catch (const std::exception &e) {
    fs::remove(tmp);
    std::cerr << e.what() << '\n';
    return 1;
  }
}
