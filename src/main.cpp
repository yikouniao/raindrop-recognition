#include "opencv2/opencv.hpp"
#include "anisotropic-diffusion.h"
#include "sobel.h"

using namespace std;
using namespace cv;

static void help() {
  cout << "raindrop-recognition\n\n"
       << "Waiting for update...\n"
       << "The code is based on OpenCV3.10.\n";
}

int main(int argc, char** argv) {
  help();
  Mat img = imread("images/20090423-194552-01-P.jpg");
  cvtColor(img, img, CV_BGR2GRAY);
  namedWindow("Original image");
  imshow("Original image", img);
  img.convertTo(img, CV_32FC1);
  Mat dst;
  int k = 16, iterate = 64;
  anisotropicDiffusion(img, dst, k, iterate);
  dst.convertTo(dst, CV_8UC1);
  namedWindow("dst");
  imshow("dst", dst);
  waitKey(0);
  return 0;
}