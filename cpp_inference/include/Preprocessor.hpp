#pragma once

#include <opencv2/opencv.hpp>

class Preprocessor
{
public:
    static void process(const cv::Mat& imageBgr, int inputW, int inputH, std::vector<float>& outputTensor);
};
