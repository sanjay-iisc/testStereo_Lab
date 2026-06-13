#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

class Postprocessor
{
public:
    static void createSkyMask(
        const std::vector<float>& skyLogits,
        int inputW,
        int inputH,
        cv::Mat& skyMask);

    static void createDepthMap(
        const std::vector<float>& depthOutput,
        int inputW,
        int inputH,
        cv::Mat& depthMap);

    static cv::Mat createVisualization(
        const cv::Mat& resizedBgr,
        const cv::Mat& skyMask,
        const cv::Mat& depthMap,
        double inferenceMs);
};