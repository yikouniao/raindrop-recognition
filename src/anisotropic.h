#pragma once
#include "opencv2/opencv.hpp"

void anisotropicDiffusion(cv::InputArray _src, cv::OutputArray _dst,
                          int k, int iterate);
static void calcfluxAll(cv::InputArray _src, cv::OutputArray _fluxN,
                        cv::OutputArray _fluxS, cv::OutputArray _fluxE,
                        cv::OutputArray _fluxW, int k);
static void calcflux(cv::InputArray _src, cv::OutputArray _flux,
                     int k, char dir);
static void calcfluxBydelta(cv::InputArray _delta, cv::OutputArray _flux,
                            int k);