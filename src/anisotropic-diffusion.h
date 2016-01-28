#pragma once
#include "opencv2/opencv.hpp"

void anisotropicDiffusion(cv::InputArray _src, cv::OutputArray _dst,
                          int k, int iterate);
static void calcfluxAll(cv::InputArray _src, cv::OutputArray _fluxN,
                        cv::OutputArray _fluxS, cv::OutputArray _fluxE,
                        cv::OutputArray _fluxW, int k);
static void calcfluxN(cv::InputArray _src, cv::OutputArray _fluxN, int k);
static void calcfluxS(cv::InputArray _src, cv::OutputArray _fluxS, int k);
static void calcfluxE(cv::InputArray _src, cv::OutputArray _fluxE, int k);
static void calcfluxW(cv::InputArray _src, cv::OutputArray _fluxW, int k);
static void calcflux(cv::InputArray _delta, cv::OutputArray _flux, int k);