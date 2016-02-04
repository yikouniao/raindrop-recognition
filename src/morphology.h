#pragma once
#include "opencv2/opencv.hpp"

void open(const cv::Mat& src, cv::Mat& dst);
void close(const cv::Mat& src, cv::Mat& dst);
void removeSmallConnectedComponents(const cv::Mat& src, cv::Mat& dst,
                                    int minArea);
static void clearAConnectedComponent(
  const cv::Mat_<uchar>& src, cv::Mat_<uchar>& dst,
  cv::Mat_<int> labels, int label_to_clear);
void clearImgEdgeInterference(const cv::Mat& _src, cv::Mat& _dst,
                              double ratio);