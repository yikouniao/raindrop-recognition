#pragma once
#include <string>

#if defined WIN32 || defined _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#include "sys/types.h"
#endif
#include <sys/stat.h>

const std::string images_dir = "images";
const std::string results_dir = "results";

static void MakeDir(const std::string& dir);
void MakeUsedDirs();