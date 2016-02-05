#include "raindrop-recog.h"
#include <vector>

using namespace cv;
using namespace std;

void raindropRecognition(cv::Mat& img_original, cv::Mat& img_binary,
                         cv::Mat& img_dst) {
  img_dst = img_original.clone();
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;

  // Find contours
  findContours(img_binary, contours, hierarchy,
               CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
  vector<RotatedRect> minEllipse(contours.size());

  // Calculate fittest ellipses for contours
  for (size_t i = 0; i < contours.size(); ++i) {
    minEllipse[i] = fitEllipse(Mat(contours[i]));
  }

  // Draw ellipses of raindrops on the image
  for (size_t i = 0; i < contours.size(); ++i) {
    ellipse(img_dst, minEllipse[i], Scalar(0, 255, 0), 1, LINE_8);
  }

  // Draw series numbers of raindrops on the image
  int font_face = FONT_HERSHEY_SIMPLEX;
  double font_scale = 0.5;
  int thickness = 1;
  for (size_t i = 0; i < contours.size(); ++i) {
    string num = to_string(i + 1);
    Point org = contours[i][0];

    // Pull it back if the origin is close to the edges of the image
    int thresh_to_edge = 12;
    if (org.x < thresh_to_edge) {
      org.x += thresh_to_edge;
    } else if (org.x > img_dst.cols - thresh_to_edge) {
      org.x -= thresh_to_edge;
    }
    if (org.y < thresh_to_edge) {
      org.y += thresh_to_edge;
    } else if (org.y > img_dst.rows - thresh_to_edge) {
      org.y -= thresh_to_edge;
    }
    putText(img_dst, num, org, font_face, font_scale,
            Scalar(0, 255, 0), thickness, 8);
  }

  // Draw sum numbers of raindrops on upper center of the image
  string text_sum = to_string(contours.size()) + " raindrops in sum";
  int baseline = 0;
  font_scale = 1;
  Size text_size = getTextSize(text_sum, font_face,
                               font_scale, thickness, &baseline);
  Point text_sum_org((img_dst.cols - text_size.width) / 2,
                     60 + text_size.height / 2);
  putText(img_dst, text_sum, text_sum_org, font_face, font_scale,
          Scalar(0, 0, 255), thickness, 8);
}