#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include <set>

int findConnectedComponent(const cv::Mat& src, cv::Mat& labels);
static int findLabelNeighbors(const cv::Mat& labels, int i, int j,
                              std::vector<int>& neighbors);
static void unionLinkedNeighbors(std::vector<int>& linked,
                                 const std::vector<int>& neighbors);
static int getconverted(const std::vector<std::vector<int>>& linkeds,
                        std::vector<int>& converted);
static void getlabelSet(const std::vector<std::vector<int>>& linkeds,
                        int labelRoot, std::set<int>& labelSet);