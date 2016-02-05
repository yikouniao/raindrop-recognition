#pragma once
#include "opencv2/opencv.hpp"

void raindropRecognition(cv::Mat& img_original, cv::Mat& img_binary,
                         cv::Mat& img_dst);