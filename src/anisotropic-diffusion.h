#pragma once
#include "opencv2/opencv.hpp"

void anisotropicDiffusion(cv::InputArray _src, cv::OutputArray _dst,
                          int k, int iterate);