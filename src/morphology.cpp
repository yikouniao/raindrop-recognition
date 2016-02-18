#include "morphology.h"
#include "connected-component.h"

using namespace cv;

// Use the matrix to implement open and close operation
unsigned char m[7][7] = {{0, 0, 0, 1, 0, 0, 0},
                         {0, 1, 1, 1, 1, 1, 0},
                         {0, 1, 1, 1, 1, 1, 0},
                         {1, 1, 1, 1, 1, 1, 1},
                         {0, 1, 1, 1, 1, 1, 0},
                         {0, 1, 1, 1, 1, 1, 0},
                         {0, 0, 0, 1, 0, 0, 0}};

void open(const cv::Mat& src, cv::Mat& dst) {
  Mat element = Mat(7, 7, CV_8U, m);
  morphologyEx(src, dst, MORPH_OPEN, element);
}

void close(const cv::Mat& src, cv::Mat& dst) {
  Mat element = Mat(7, 7, CV_8U, m);
  morphologyEx(src, dst, MORPH_CLOSE, element);
}

void removeSmallConnectedComponents(const cv::Mat& src, cv::Mat& _dst,
                                    int minArea) {
  CV_Assert(src.depth() == CV_8UC1);
  Mat_<int> labels, stats;
  Mat_<double> centroids;
  int nLabels = connectedComponentsWithStats(src, labels, stats,
                                             centroids, 8, CV_32S);
  // findConnectedComponent is a function I wrote to realize almost the same
  // things as connectedComponentsWithStats, but it's much slower and
  // unpractical.
  //int nLabels = findConnectedComponent(src, labels, stats, centroids);
  Mat_<uchar>& dst = (Mat_<uchar>&)_dst;
  for (int i = 1; i < nLabels; ++i) {
    if (stats(i, CC_STAT_AREA) < minArea) {
      clearAConnectedComponent(src, dst, labels, i);
    }
  }
}

static void clearAConnectedComponent(
    const cv::Mat_<uchar>& src, cv::Mat_<uchar>& dst,
    cv::Mat_<int> labels, int label_to_clear) {
  CV_Assert(src.depth() == CV_8UC1);
  dst = Mat_<uchar>(src.size());
  for (int i = 0; i < src.rows; ++i) {
    for (int j = 0; j < src.cols; ++j) {
      if (labels(i, j) == label_to_clear) {
        dst(i, j) = 0;
      } else {
        dst(i, j) = src(i, j);
      }
    }
  }
}

void clearImgEdgeInterference(const cv::Mat& _src, cv::Mat& _dst,
                              double ratio) {
  CV_Assert(_src.depth() == CV_8UC1);
  _dst = _src.clone();
  Mat_<uchar>& src = (Mat_<uchar>&)_src;
  Mat_<uchar>& dst = (Mat_<uchar>&)_dst;
  for (int i = 0; i < src.rows; ++i) {
    int pos_cnt = 0;
    for (int j = 0; j < src.cols; ++j) {
      pos_cnt = src(i, j) ? pos_cnt + 1 : pos_cnt;
    }
    if (static_cast<double>(pos_cnt) / src.cols > ratio) {
      dst.row(i) = Mat::zeros(1, dst.cols, CV_8UC1);
    }
  }
  for (int j = 0; j < src.cols; ++j) {
    int pos_cnt = 0;
    for (int i = 0; i < src.rows; ++i) {
      pos_cnt = src(i, j) ? pos_cnt + 1 : pos_cnt;
    }
    if (static_cast<double>(pos_cnt) / src.rows > ratio) {
      dst.col(j) = Mat::zeros(dst.rows, 1, CV_8UC1);
    }
  }
}