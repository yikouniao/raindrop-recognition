#include "anisotropic-diffusion.h"

using namespace cv;

void anisotropicDiffusion(cv::InputArray _src, cv::OutputArray _dst,
                          int k, int iterate) {
  Mat src = _src.getMat();
  _dst.create(src.size(), src.type());
  Mat dst = _dst.getMat();
  dst = src.clone(); // the pointer data will change
  for (size_t i = 0; i < iterate; ++i) {
    Mat fluxN;
  }
}