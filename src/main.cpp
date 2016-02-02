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
  img = img(Rect(0, 8, 690, img.rows - 8)); // Tailor invalid area
  namedWindow("Original image");
  imshow("Original image", img);
  img.convertTo(img, CV_32FC1);

  // Anisotropic diffusion
  int k = 16, iterate = 64;
  anisotropicDiffusion(img, img, k, iterate);

  // Sobel Derivatives
  sobel(img, img);

  // binarization
  double thresh = 20, maxval = 255;
  threshold(img, img, thresh, maxval, THRESH_BINARY);

  img.convertTo(img, CV_8UC1);
  namedWindow("dst");
  imshow("dst", img);
  waitKey(0);
  return 0;
}