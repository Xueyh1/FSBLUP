// Standalone positive and negative controls for dynamic analysis, never linked
// into fsblup.
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
volatile int shared_value;
__attribute__((noinline)) void allocation(bool lose) {
  auto p = static_cast<volatile char *>(std::malloc(79));
  *p = 1;
  if (!lose)
    std::free(const_cast<char *>(p));
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  if (std::strcmp(argv[1], "leak") == 0 || std::strcmp(argv[1], "clean") == 0) {
    allocation(std::strcmp(argv[1], "leak") == 0);
    return 0;
  }
  bool race = std::strcmp(argv[1], "race") == 0;
  std::atomic<int> safe{0};
  auto work = [&] {
    for (int i = 0; i < 10000; ++i) {
      if (race)
        ++shared_value;
      else
        ++safe;
    }
  };
  std::thread t(work);
  work();
  t.join();
}
