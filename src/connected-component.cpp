#include "connected-component.h"
#include <vector>

using namespace cv;
using namespace std;

int findConnectedComponent(const cv::Mat& src, cv::Mat& labels) {
  // Two pass
  labels = Mat(src.rows, src.cols, CV_32SC1, Scalar(0));
  vector<vector<int>> linked;
  int next_label = 1;

  // First pass
  for (int i = 0; i < src.rows; ++i) {
    for (int j = 0; j < src.cols; ++j) {
      if (src.at<uchar>(i, j) == 255) {
        vector<int> neighbors;
        int min_neighbor_label = findLabelNeighbors(labels, i, j, neighbors);
        if (min_neighbor_label) { // If neighbors is not empty
          labels.at<int>(i, j) = min_neighbor_label;
          for (int k = 0; k < neighbors.size(); ++k) {
            if (!neighbors[k]) {
              // union(linked[neighbors[k]], neighbors)
              
            }
          }
        }
        else { // If neighbors is empty
          linked[next_label].push_back(next_label);
          labels.at<int>(i, j) = next_label;
          ++next_label;
        }
      }
    }
  }
}

// Find neighbors of a label, and return the minimum label
// If return 0, it means the neighbors is empty
static int findLabelNeighbors(const cv::Mat& labels, int i, int j,
                              std::vector<int>& neighbors) {
  CV_Assert(i >= 0 && i < labels.rows && j >= 0 && j < labels.cols);
  if (j != 0)
    neighbors.push_back(labels.at<int>(i, j - 1));
  if (j != labels.cols - 1)
    neighbors.push_back(labels.at<int>(i, j + 1));
  if (i != 0) {
    neighbors.push_back(labels.at<int>(i - 1, j));
    if (j != 0)
      neighbors.push_back(labels.at<int>(i - 1, j - 1));
    if (j != labels.cols - 1)
      neighbors.push_back(labels.at<int>(i - 1, j + 1));
  }
  if (i != labels.rows - 1) {
    neighbors.push_back(labels.at<int>(i + 1, j));
    if (j != 0)
      neighbors.push_back(labels.at<int>(i + 1, j - 1));
    if (j != labels.cols - 1)
      neighbors.push_back(labels.at<int>(i + 1, j + 1));
  }

  // Erase redundant elements
  sort(neighbors.begin(), neighbors.end());
  auto last = unique(neighbors.begin(), neighbors.end());
  neighbors.erase(last, neighbors.end());

  // Erase zero elements(background label)
  // After sorting, only the first element can be 0
  if (!neighbors[0]) {
    neighbors.erase(neighbors.begin());
  }

  if (neighbors.empty()) {
    return 0;
  }
  return neighbors[0];
}

// union(linked, neighbors), linked += neighbors
static void unionLinkedNeighbors(std::vector<int>& linked,
                                 const std::vector<int>& neighbors) {

}