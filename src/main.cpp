#include "opencv2/opencv.hpp"
#include "anisotropic.h"
#include "sobel.h"
#include "morphology.h"
#include "raindrop-recog.h"
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
  Mat img_original = imread("images/20090423-194552-01-P.jpg");
  // Tailor invalid area
  img_original = img_original(Rect(0, 8, 690, img_original.rows - 8));
  namedWindow("Original image");
  imshow("Original image", img_original);
  Mat img_grey;
  cvtColor(img_original, img_grey, CV_BGR2GRAY);
  img_grey.convertTo(img_grey, CV_32FC1);

  // Anisotropic diffusion
  const int k = 16, iterate = 2;
  anisotropicDiffusion(img_grey, img_grey, k, iterate);

  // Sobel Derivatives
  sobel(img_grey, img_grey);

  // Binarization
  Mat img_binary;
  const double thresh = 12, max_val = 255;
  threshold(img_grey, img_binary, thresh, max_val, THRESH_BINARY);
  img_binary.convertTo(img_binary, CV_8UC1);

  // Seperate interference from raindrops and weaken it
  open(img_binary, img_binary);

  // Remove some interference from raindrops
  const int minArea = 10;
  removeSmallConnectedComponents(img_binary, img_binary, minArea);

  // Make the edges of raindrops more continuous
  close(img_binary, img_binary);

  // Remove the long straight edges interference of the image
  const double ratio = 0.75;
  clearImgEdgeInterference(img_binary, img_binary, ratio);

  // Clear the numbers and characters on the top left corner
  img_binary(Rect(20, 12, 416, 32)) = Mat::zeros(32, 416, CV_8UC1);

  // Make the edges of raindrops more continuous
  close(img_binary, img_binary);

  // Fill the holes inside the raindrops
  Mat img_temp = img_binary.clone();
  vector<vector<Point>> contours;
  findContours(img_temp, contours, noArray(),
               CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
  for (size_t i = 0; i < contours.size(); i++) {
    drawContours(img_binary, contours, i, Scalar(255),
                 CV_FILLED, 8, noArray(), 0);
  }

  // Remove some interference from raindrops
  removeSmallConnectedComponents(img_binary, img_binary, minArea);

  // Raindrops recognition
  Mat img_dst;
  raindropRecognition(img_original, img_binary, img_dst);

  namedWindow("Result");
  imshow("Result", img_dst);
  waitKey(0);
  return 0;
}