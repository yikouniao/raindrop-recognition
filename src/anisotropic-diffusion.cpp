#include "anisotropic-diffusion.h"

using namespace cv;

void anisotropicDiffusion(cv::InputArray _src, cv::OutputArray _dst,
                          int k, int iterate) {
  Mat src = _src.getMat();
  CV_Assert(src.depth() == CV_32F || src.depth() == CV_64F);
  _dst.create(src.size(), src.type());
  Mat dst = _dst.getMat();
  dst = src.clone();

  for (size_t i = 0; i < iterate; ++i) {
    Mat fluxN, fluxS, fluxE, fluxW;
    calcfluxAll(dst, fluxN, fluxS, fluxE, fluxW, k);
  }
  dst.copyTo(_dst);
}

static void calcfluxAll(cv::InputArray _src, cv::OutputArray _fluxN,
                        cv::OutputArray _fluxS, cv::OutputArray _fluxE,
                        cv::OutputArray _fluxW, int k) {
  Mat src = _src.getMat();
  _fluxN.create(src.size(), src.type());
  Mat fluxN = _fluxN.getMat();
  _fluxS.create(src.size(), src.type());
  Mat fluxS = _fluxS.getMat();
  _fluxE.create(src.size(), src.type());
  Mat fluxE = _fluxE.getMat();
  _fluxW.create(src.size(), src.type());
  Mat fluxW = _fluxW.getMat();
  calcfluxN(src, fluxN, k);
  calcfluxS(src, fluxS, k);
  calcfluxE(src, fluxE, k);
  calcfluxW(src, fluxW, k);
}

static void calcfluxN(cv::InputArray _src, cv::OutputArray _fluxN, int k) {
  Mat src = _src.getMat();
  _fluxN.create(src.size(), src.type());
  Mat fluxN = _fluxN.getMat();
  // Calculate deltaN
  Mat deltaN = src.clone();
  deltaN.row(0) = Mat::zeros(1, deltaN.cols, deltaN.depth());
  for (int i = 1; i < deltaN.rows; ++i) {
    deltaN.row(i) = src.row(i - 1) - src.row(i);
  }
  // Calculate fluxN
  calcflux(deltaN, fluxN, k);
}

static void calcfluxS(cv::InputArray _src, cv::OutputArray _fluxS, int k) {
  Mat src = _src.getMat();
  _fluxS.create(src.size(), src.type());
  Mat fluxS = _fluxS.getMat();
  // Calculate deltaS
  Mat deltaS = src.clone();
  deltaS.row(deltaS.rows - 1) = Mat::zeros(1, deltaS.cols, deltaS.depth());
  for (int i = 0; i < deltaS.rows - 1; ++i) {
    deltaS.row(i) = src.row(i + 1) - src.row(i);
  }
  // Calculate fluxS
  calcflux(deltaS, fluxS, k);
}

static void calcfluxE(cv::InputArray _src, cv::OutputArray _fluxE, int k) {
  Mat src = _src.getMat();
  _fluxE.create(src.size(), src.type());
  Mat fluxE = _fluxE.getMat();
  // Calculate deltaE
  Mat deltaE = src.clone();
  deltaE.col(deltaE.cols - 1) = Mat::zeros(deltaE.rows, 1, deltaE.depth());
  for (int i = 0; i < deltaE.cols - 1; ++i) {
    deltaE.col(i) = src.col(i + 1) - src.col(i);
  }
  // Calculate fluxE
  calcflux(deltaE, fluxE, k);
}

static void calcfluxW(cv::InputArray _src, cv::OutputArray _fluxW, int k) {
  Mat src = _src.getMat();
  _fluxW.create(src.size(), src.type());
  Mat fluxW = _fluxW.getMat();
  // Calculate deltaW
  Mat deltaW = src.clone();
  deltaW.col(0) = Mat::zeros(deltaW.rows, 1, deltaW.depth());
  for (int i = 1; i < deltaW.cols; ++i) {
    deltaW.col(i) = src.col(i - 1) - src.col(i);
  }
  // Calculate fluxN
  calcflux(deltaW, fluxW, k);
}

static void calcflux(cv::InputArray _delta, cv::OutputArray _flux, int k) {
  Mat delta = _delta.getMat();
  _flux.create(delta.size(), delta.type());
  Mat flux = _flux.getMat();
  // flux = delta .* exp(-((abs(delta) ./ k) .^ 2))
  Mat flux_temp;
  pow(abs(delta) / k, 2, flux_temp);
  exp(-flux_temp, flux_temp);
  flux = delta.mul(flux_temp);
}