#include "morphology.h"

using namespace cv;

void morphologyOperation(const cv::Mat& src, cv::Mat& dst) {
  open(src, dst);
}

static void open(const cv::Mat& src, cv::Mat& dst) {
  unsigned char m[7][7] = {{0, 0, 0, 1, 0, 0, 0},
                           {0, 1, 1, 1, 1, 1, 0},
                           {0, 1, 1, 1, 1, 1, 0},
                           {1, 1, 1, 1, 1, 1, 1},
                           {0, 1, 1, 1, 1, 1, 0},
                           {0, 1, 1, 1, 1, 1, 0},
                           {0, 0, 0, 1, 0, 0, 0}};
  Mat element = Mat(7, 7, CV_8U, m);
  morphologyEx(src, dst, MORPH_OPEN, element);
}