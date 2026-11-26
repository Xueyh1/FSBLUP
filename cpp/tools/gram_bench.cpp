// Standalone BLAS experiment. Compile with the same OpenBLAS as fsblup.
#include <algorithm>
#include <cblas.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
int main() {
  std::cout << "samples\tfeatures\tmethod\trepeat\tseconds\tmax_abs_error\n";
  for (auto shape : {std::pair{24, 17}, std::pair{74, 5000},
                     std::pair{198, 5000}, std::pair{513, 1025}}) {
    auto [n, m] = shape;
    std::vector<double> x(n * m), k(n * n), ref(n * n);
    for (int i = 0; i < n * m; ++i)
      x[i] = std::sin(.017 * i);
    for (int rep = 0; rep <= 5; ++rep)
      for (int j = 0; j < 3; ++j) {
        int method = (rep + j) % 3;
        auto start = std::chrono::steady_clock::now();
        if (method == 0)
          cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, m, 1,
                      x.data(), n, x.data(), n, 0, k.data(), n);
        else {
          int block = method == 1 ? m : 256;
          for (int col = 0; col < m; col += block)
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n,
                        std::min(block, m - col), 1, x.data() + col * n, n,
                        col ? 1 : 0, k.data(), n);
          for (int c = 0; c < n; ++c)
            for (int r = c + 1; r < n; ++r)
              k[c + r * n] = k[r + c * n];
        }
        auto s = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count();
        if (rep == 0 && method == 0)
          ref = k;
        double err = 0;
        for (size_t i = 0; i < k.size(); ++i)
          err = std::max(err, std::abs(k[i] - ref[i]));
        if (err > 1e-9)
          throw std::runtime_error("Gram mismatch");
        std::cout << n << '\t' << m << '\t' << method << '\t' << rep << '\t'
                  << s << '\t' << err << '\n';
      }
  }
}
