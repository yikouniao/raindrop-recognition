#include "directory.h"

static void MakeDir(const std::string& dir) {
#if defined WIN32 || defined _WIN32
  CreateDirectoryA(dir.c_str(), 0);
#else
  mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}

void MakeUsedDirs() {
  MakeDir(results_dir);
}