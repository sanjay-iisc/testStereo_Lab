#include "Preprocessor.hpp"

#include <stdexcept>

// void Preprocessor::process(const cv::Mat& imageBgr, int inputW, int inputH, std::vector<float>& outputTensor)
// {
//     if(imageBgr.empty())
//     {
//         throw std::runtime_error("Input image is empty");
//     }

//     cv::Mat resized;
//     cv::resize(imageBgr, resized, cv::Size(inputW, inputH));

//     cv::Mat imageRgb;
//     cv::cvtColor(resized, imageRgb, cv::COLOR_BGR2RGB);

//     imageRgb.convertTo(imageRgb,CV_32FC3,1.0 / 255.0);

//     outputTensor.resize(3 * inputH * inputW);

//     int index = 0;

//     for(int c = 0; c < 3; ++c)
//     {
//         for(int h = 0; h < inputH; ++h)
//         {
//             for(int w = 0; w < inputW; ++w)
//             {
//                 outputTensor[index++] =
//                     imageRgb.at<cv::Vec3f>(h, w)[c];
//             }
//         }
//     }
// }


void Preprocessor::process(const cv::Mat& imageBgr, int inputW, int inputH, std::vector<float>& outputTensor)
{
    if(imageBgr.empty())
    {
        throw std::runtime_error("Input image is empty");
    }

    cv::Mat blob = cv::dnn::blobFromImage(
        imageBgr,                    // input image
        1.0 / 255.0,                 // scale
        cv::Size(inputW, inputH),    // resize
        cv::Scalar(),                // mean
        true,                        // BGR -> RGB
        false,                       // don't crop
        CV_32F);                     // float32

    const size_t tensorSize = static_cast<size_t>(blob.total());

    outputTensor.resize(tensorSize);

    std::memcpy(outputTensor.data(), blob.ptr<float>(), tensorSize * sizeof(float));
}