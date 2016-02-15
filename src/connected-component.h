#pragma once
#include "opencv2/opencv.hpp"

int findConnectedComponent(const cv::Mat& src, cv::Mat& labels);
static int findLabelNeighbors(const cv::Mat& labels, int i, int j,
                              std::vector<int>& neighbors);
static void unionLinkedNeighbors(std::vector<int>& linked,
                                 const std::vector<int>& neighbors);