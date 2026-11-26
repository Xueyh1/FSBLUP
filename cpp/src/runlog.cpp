#include "cli.hpp"
#include "runlog.hpp"
#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <mutex>

namespace fsb {
struct RunLog::Impl {
  std::mutex mutex;
  string error;
  std::unique_ptr<spdlog::logger> logger;
};
RunLog::RunLog(const Options &opt,const string &file,const string &command):impl(std::make_unique<Impl>()) {
  const string level=opt.get("log-level","info");
  const std::map<string,spdlog::level::level_enum> levels={{"off",spdlog::level::off},{"error",spdlog::level::err},{"warn",spdlog::level::warn},{"info",spdlog::level::info},{"debug",spdlog::level::debug},{"trace",spdlog::level::trace}};
  require(levels.count(level),"unknown --log-level: "+level);
  std::vector<spdlog::sink_ptr> sinks={std::make_shared<spdlog::sinks::stderr_color_sink_mt>()};
  if(!file.empty())sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(file,false));
  impl->logger=std::make_unique<spdlog::logger>(std::to_string(getpid()),sinks.begin(),sinks.end());
  impl->logger->set_pattern("run=%n thread=%t %l %v");
  impl->logger->set_level(levels.at(level));
  impl->logger->set_error_handler([this](const string& msg){std::lock_guard<std::mutex> lock(impl->mutex);impl->error=msg;});
  info("command="+command+" stage=start");
}
RunLog::~RunLog()=default;
void RunLog::info(const string& s){impl->logger->info("{}",s);}
void RunLog::debug(const string& s){impl->logger->debug("{}",s);}
bool RunLog::debug_enabled()const{return impl->logger->should_log(spdlog::level::debug);}
void RunLog::verify(){impl->logger->flush();std::lock_guard<std::mutex> lock(impl->mutex);require(impl->error.empty(),"log write/flush failed: "+impl->error);}
}
