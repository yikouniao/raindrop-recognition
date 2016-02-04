#include "opencv2/opencv.hpp"
#include "anisotropic.h"
#include "sobel.h"
#include "morphology.h"
#include <vector>
#include <array>

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
  const int k = 16, iterate = 2;
  anisotropicDiffusion(img, img, k, iterate);

  // Sobel Derivatives
  sobel(img, img);

  // Binarization
  const double thresh = 12, max_val = 255;
  threshold(img, img, thresh, max_val, THRESH_BINARY);
  img.convertTo(img, CV_8UC1);

  // Seperate interference from raindrops and weaken it
  open(img, img);

  // Remove some interference from raindrops
  const int minArea = 10;
  removeSmallConnectedComponents(img, img, minArea);

  // Make the edges of raindrops more continuous
  close(img, img);

  // Remove the long straight edges interference of the image
  const double ratio = 0.75;
  clearImgEdgeInterference(img, img, ratio);

  // Make the edges of raindrops more continuous
  close(img, img);

  // Fill the holes inside the raindrops
  Mat img_temp = img.clone();
  vector<vector<Point>> contours;
  findContours(img_temp, contours, noArray(),
               CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
  for (size_t i = 0; i < contours.size(); i++) {
    drawContours(img, contours, i, Scalar(255), CV_FILLED, 8, noArray(), 0);
  }

  // Remove some interference from raindrops
  removeSmallConnectedComponents(img, img, minArea);

  namedWindow("dst");
  imshow("dst", img);
  waitKey(0);
  return 0;
}