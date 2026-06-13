#include "Postprocessor.hpp"

#include <stdexcept>
#include <string>

void Postprocessor::createSkyMask(const std::vector<float>& skyLogits, int inputW, int inputH, cv::Mat& skyMask)
{
    const size_t expectedSize = static_cast<size_t>(2 * inputH * inputW);

    if(skyLogits.size() != expectedSize)
    {
        throw std::runtime_error("Invalid sky logits size");
    }

    skyMask = cv::Mat(inputH, inputW, CV_8UC1);

    const int planeSize = inputH * inputW;

    for(int h = 0; h < inputH; ++h)
    {
        for(int w = 0; w < inputW; ++w)
        {
            const int idx = h * inputW + w;

            const float class0 =
                skyLogits[idx];

            const float class1 =
                skyLogits[planeSize + idx];

            skyMask.at<uchar>(h, w) =
                class1 > class0 ? 255 : 0;
        }
    }
}

void Postprocessor::createDepthMap(const std::vector<float>& depthOutput, int inputW, int inputH, cv::Mat& depthMap)
{
    const size_t expectedSize =
        static_cast<size_t>(inputH * inputW);

    if(depthOutput.size() != expectedSize)
    {
        throw std::runtime_error("Invalid depth output size");
    }

    depthMap = cv::Mat(inputH, inputW, CV_32FC1);

    std::memcpy(depthMap.data, depthOutput.data(), depthOutput.size() * sizeof(float));
}

cv::Mat Postprocessor::createVisualization(const cv::Mat& resizedBgr, const cv::Mat& skyMask, const cv::Mat& depthMap, double inferenceMs)
{
    if(resizedBgr.empty() || skyMask.empty() || depthMap.empty())
    {
        throw std::runtime_error("Empty input to visualization");
    }

    cv::Mat depthNorm;
    cv::normalize(depthMap, depthNorm, 0, 255, cv::NORM_MINMAX);

    depthNorm.convertTo(depthNorm, CV_8UC1);

    cv::Mat depthColor;
    cv::applyColorMap(depthNorm, depthColor, cv::COLORMAP_JET);

    cv::Mat overlay = resizedBgr.clone();

    overlay.setTo(cv::Scalar(255, 0, 0),skyMask);

    cv::addWeighted(overlay, 0.4, resizedBgr, 0.6, 0.0, overlay);

    const double fps = inferenceMs > 0.0 ? 1000.0 / inferenceMs : 0.0;

    cv::putText(overlay, "Inference: " + std::to_string(inferenceMs) + " ms", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65, 
    cv::Scalar(0, 255, 0),2);

    cv::putText(overlay,"FPS: " + std::to_string(fps), cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);

    cv::Mat display;

    cv::hconcat(std::vector<cv::Mat>{resizedBgr, overlay, depthColor},display);

    return display;
}