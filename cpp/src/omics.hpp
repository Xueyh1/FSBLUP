#pragma once
#include "core.hpp"
#include <cmath>
namespace fsb {
// Welford on differences from the first observation. Rescale the running
// moments by powers of two when the observed dynamic range grows.
struct OmicsMoment {
  size_t count = 0, low = 0;
  double anchor = 0;
  int exponent = 0;
  long double mean = 0, m2 = 0;
  void add(double x) {
    require(std::isfinite(x), "nonfinite omics feature");
    if (!count)
      anchor = x;
    low += x < .1;
    ++count;
    long double difference = static_cast<long double>(x) - anchor;
    if (difference != 0) {
      int next;
      std::frexp(difference, &next);
      if (next > exponent || (!m2 && !mean)) {
        mean = std::scalbn(mean, exponent - next);
        m2 = std::scalbn(m2, 2 * (exponent - next));
        exponent = next;
      }
    }
    long double v = std::scalbn(difference, -exponent), delta = v - mean;
    mean += delta / count;
    m2 += delta * (v - mean);
  }
  double center() const {
    return static_cast<double>(anchor + std::scalbn(mean, exponent));
  }
  double sd() const {
    double value = std::scalbn(std::sqrt(m2 / (count - 1)), exponent);
    require(std::isfinite(value) && (value > 0 || m2 == 0),
            "omics SD outside representable range");
    return value;
  }
  bool filtered(bool fpkm) const { return fpkm && low * 100 >= 95 * count; }
  double standardized(double x) const {
    long double z =
        std::scalbn(static_cast<long double>(x) - anchor, -exponent) - mean;
    return z / std::sqrt(m2 / (count - 1));
  }
};
} // namespace fsb
