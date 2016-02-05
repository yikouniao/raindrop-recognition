#include "raindrop-recog.h"
#include <vector>

using namespace cv;
using namespace std;

void raindropRecognition(Mat& img, Mat& img_binary) {
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(img_binary, contours, hierarchy,
               CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
  vector<RotatedRect> minEllipse(contours.size());
  for (size_t i = 0; i < contours.size(); i++) {
    minEllipse[i] = fitEllipse(Mat(contours[i]));
  }
  for (size_t i = 0; i< contours.size(); i++) {
    ellipse(img, minEllipse[i], Scalar(0, 255, 0), 2, LINE_8);
  }
}