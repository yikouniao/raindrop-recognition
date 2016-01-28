#include "opencv2/opencv.hpp"
#include "anisotropic-diffusion.h"

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
  img.convertTo(img, CV_32FC1);
  Mat dst;
  anisotropicDiffusion(img, dst, 16, 64);
  namedWindow("fu");
  imshow("fu", dst);
  waitKey(0);
  return 0;
}