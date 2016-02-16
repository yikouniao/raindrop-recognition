#include "connected-component.h"

using namespace cv;
using namespace std;

// The function realizes almost the same things
// as cv::connectedComponentsWithStats besides the choices
// of connectivity and ltype.
// return: the number of labels
// stats: statistics output for each label, including the background label.
//   Statistics are accessed via stats(label, COLUMN) where COLUMN is one of
//   cv::ConnectedComponentsTypes. The data type is CV_32S(int).
// centroids: centroid output for each label, including the background label.
//   Centroids are accessed via centroids(label, 0) for x
//   and centroids(label, 1) for y.The data type CV_64F(double).
int findConnectedComponent(const cv::Mat& src, cv::Mat& labels,
                           cv::Mat_<int>& stats, cv::Mat_<double> centroids) {
  // Two pass
  labels = Mat(src.rows, src.cols, CV_32SC1, Scalar(0));
  vector<vector<int>> linkeds;
  int next_label = 1;

  // First pass
  for (int i = 0; i < src.rows; ++i) {
    for (int j = 0; j < src.cols; ++j) {
      if (src.at<uchar>(i, j) == 255) {
        vector<int> neighbors;
        int min_neighbor_label = findLabelNeighbors(labels, i, j, neighbors);
        if (min_neighbor_label) { // If neighbors is not empty
          labels.at<int>(i, j) = min_neighbor_label;
          for (size_t k = 0; k < neighbors.size(); ++k) {
            unionLinkedNeighbors(linkeds[neighbors[k]], neighbors);
          }
        } else {                  // If neighbors is empty
          linkeds[next_label].push_back(next_label);
          labels.at<int>(i, j) = next_label;
          ++next_label;
        }
      }
    }
  }

  // Get converted
  vector<int> converted(linkeds.size(), 0);
  int nLabels = getconverted(linkeds, converted);

  // Second pass
  for (int i = 0; i < labels.rows; ++i) {
    for (int j = 0; j < labels.cols; ++j) {
      if (labels.at<int>(i, j)) {
        labels.at<int>(i, j) = converted[labels.at<int>(i, j)];
      }
    }
  }

  // get number area...

  return nLabels;
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
  for (size_t i = 0; i < neighbors.size(); ++i) {
    linked.push_back(neighbors[i]);
  }
  sort(linked.begin(), linked.end());
  auto last = unique(linked.begin(), linked.end());
  linked.erase(last, linked.end());
}

// return: the number of labels
static int getconverted(const std::vector<std::vector<int>>& linkeds,
                        std::vector<int>& converted) {
  CV_Assert(converted.size() == linkeds.size());
  vector<set<int>> labelSets;

  // Start form i = 1 because i = 0 responds to background label
  for (size_t i = 1; i < converted.size(); ++i) {
    if (!converted[i]) {
      set<int> labelSet;
      getlabelSet(linkeds, i, labelSet);
      labelSets.push_back(labelSet);
      for (const auto& e : labelSet) {
        // The first element in the set will be the smallest one
        converted[e] = *labelSet.begin();
      }
    }
  }

  // Modify labels into 1, 2, 3...
  // There must be a label 1 in converted, so begin from label 2
  for (size_t i = 2; i < labelSets.size(); ++i) {
    size_t j = i;
    while (find(converted.begin(), converted.end(), j++) == converted.end()) {}
    --j;
    if (i != j) {
      for (auto& e : converted) {
        e = (e == j) ? i : e;
      }
    }
  }

  return labelSets.size();
}

static void getlabelSet(const std::vector<std::vector<int>>& linkeds,
                        int labelRoot, std::set<int>& labelSet) {
  for (size_t i = 0; i < linkeds[labelRoot].size(); ++i) {
    auto insert_result = labelSet.insert(linkeds[labelRoot][i]);
    if (insert_result.second) {
      getlabelSet(linkeds, linkeds[labelRoot][i], labelSet);
    }
  }
}