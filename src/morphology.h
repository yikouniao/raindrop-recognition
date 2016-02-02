#pragma once
#include "opencv2/opencv.hpp"

void morphologyOperation(const cv::Mat& src, cv::Mat& dst);
static void open(const cv::Mat& src, cv::Mat& dst);