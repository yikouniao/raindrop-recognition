#include "sobel.h"

using namespace cv;

void sobel(const cv::Mat& src, cv::Mat& dst) {
  Mat grad_x, grad_y;
  const int ddepth = -1; // the dst image will have the same depth as the src
  double scale = 1, delta = 0;

  // [-1 0 1]     [-1 -2 -1]
  // [-2 0 2]     [ 0  0  0]
  // [-1 0 1] and [ 1  2  1]
  Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
  Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
  Mat abs_grad_x, abs_grad_y;
  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);
  // Approximate the gradient by adding both directional gradients.
  // This is not an exact calculation, but it is good for our purposes.
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
}