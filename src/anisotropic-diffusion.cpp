#include "anisotropic-diffusion.h"

using namespace cv;

void anisotropicDiffusion(cv::InputArray _src, cv::OutputArray _dst,
                          int k, int iterate) {
  std::cout << "Anisotropic diffusion...\n";
  Mat src = _src.getMat();
  CV_Assert(src.depth() == CV_32F || src.depth() == CV_64F);
  _dst.create(src.size(), src.type());
  Mat dst = _dst.getMat();
  dst = src.clone();

  double lambda = .025;
  for (int i = 0; i < iterate; ++i) {
    Mat fluxN, fluxS, fluxE, fluxW;
    calcfluxAll(dst, fluxN, fluxS, fluxE, fluxW, k);
    dst += lambda * (fluxN + fluxS + fluxE + fluxW);
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
  calcflux(src, fluxN, k, 'N');
  calcflux(src, fluxS, k, 'S');
  calcflux(src, fluxE, k, 'E');
  calcflux(src, fluxW, k, 'W');
}

static void calcflux(cv::InputArray _src, cv::OutputArray _flux,
                     int k, char dir) {
  Mat src = _src.getMat();
  _flux.create(src.size(), src.type());
  Mat flux = _flux.getMat();
  Mat delta = src.clone();
  switch (dir) {
    case 'N': // Calculate deltaN
      delta.row(0) = Mat::zeros(1, delta.cols, delta.depth());
      for (int i = 1; i < delta.rows; ++i) {
        delta.row(i) = src.row(i - 1) - src.row(i);
      }
      break;
    case 'S': // Calculate deltaS
      delta.row(delta.rows - 1) = Mat::zeros(1, delta.cols, delta.depth());
      for (int i = 0; i < delta.rows - 1; ++i) {
        delta.row(i) = src.row(i + 1) - src.row(i);
      }
      break;
    case 'E': // Calculate deltaE
      delta.col(delta.cols - 1) = Mat::zeros(delta.rows, 1, delta.depth());
      for (int i = 0; i < delta.cols - 1; ++i) {
        delta.col(i) = src.col(i + 1) - src.col(i);
      }
      break;
    case 'W': // Calculate deltaW
      delta.col(0) = Mat::zeros(delta.rows, 1, delta.depth());
      for (int i = 1; i < delta.cols; ++i) {
        delta.col(i) = src.col(i - 1) - src.col(i);
      }
      break;
    default:
      exit(1);
  }
  // Calculate flux
  calcfluxBydelta(delta, flux, k);
}

static void calcfluxBydelta(cv::InputArray _delta, cv::OutputArray _flux,
                            int k) {
  Mat delta = _delta.getMat();
  _flux.create(delta.size(), delta.type());
  Mat flux = _flux.getMat();
  // flux = delta .* exp(-((abs(delta) ./ k) .^ 2))
  Mat flux_temp;
  pow(abs(delta) / k, 2, flux_temp);
  exp(-flux_temp, flux_temp);
  flux = delta.mul(flux_temp);
}