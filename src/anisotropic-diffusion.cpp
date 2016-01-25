#include "anisotropic-diffusion.h"

using namespace cv;

void anisotropicDiffusion(cv::InputArray _src, cv::OutputArray _dst,
                          int k, int iterate) {
  Mat src = _src.getMat();
  _dst.create(src.size(), src.type());
  Mat dst = _dst.getMat();
  dst = src.clone();
  //for (int i = 0; i < src.rows; ++i) {
  //  for (int j = 0; j < src.cols; ++j) {
  //    dst.at<uchar>(i, j) = 255;
  //  }
  //}
  for (size_t i = 0; i < iterate; ++i) {
    Mat fluxN = dst.clone();
    fluxN.row(0) = Mat::zeros(1, fluxN.cols, fluxN.depth());
    for (int j = 1; j < fluxN.rows; ++j) {
      fluxN.row(j) = dst.row(j - 1) - dst.row(j);
    }
  }
  dst.copyTo(_dst);
}