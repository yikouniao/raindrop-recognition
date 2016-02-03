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
        if(min_neighbor_label) { // If neighbors is not empty
          labels.at<int>(i, j) = min_neighbor_label;
          for (int k = 0; k < neighbors.size(); ++k) {
            if (!neighbors[k]) {
              linked[neighbors[k]].push_back(min_neighbor_label);
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
  return getMinInVector(neighbors);
}

static int getMinInVector(const std::vector<int>& v) {
  CV_Assert(!v.empty());
  int min = v[0];
  for (int i = 0; i < v.size(); ++i) {
    if (v[i] < min) {
      min = v[i];
    }
  }
  return min;
}

vector<int>& vector<int>::operator+=(const vector<int>& a) {

}