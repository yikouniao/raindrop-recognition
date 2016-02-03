#pragma once
#include "opencv2/opencv.hpp"

static int findLabelNeighbors(const cv::Mat& labels, int i, int j,
                              std::vector<int>& neighbors);
static int getMaxInVector(const std::vector<int>& v);