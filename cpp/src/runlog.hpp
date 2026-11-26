#pragma once
#include <functional>
#include <memory>
#include <string>
namespace fsb {
struct Options;
class RunLog {
  struct Impl;
  std::unique_ptr<Impl> impl;

public:
  RunLog(const Options &, const std::string &file, const std::string &command);
  ~RunLog();
  void info(const std::string &);
  bool debug_enabled() const;
  bool trace_enabled() const;
  void error(const std::string &);
  void debug(const std::string &);
  void verify();
};
} // namespace fsb
