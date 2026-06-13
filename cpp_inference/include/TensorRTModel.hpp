#pragma once

#include "DeviceBuffer.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

class TensorRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class TensorRTModel
{
public:
    
    explicit TensorRTModel(const std::string& modelPath);
    
    ~TensorRTModel();

    TensorRTModel(const TensorRTModel&) = delete;
    TensorRTModel& operator=(const TensorRTModel&) = delete;

    bool infer(const cv::Mat& inputImage, 
        std::vector<float>& skyOutput, 
        std::vector<float>& depthOutput,double& inferenceMs);

    int inputC() const;
    int inputH() const;
    int inputW() const;

private:
    void loadModel(const std::string& modelPath);
    void loadEngine(const std::string& enginePath);
    void loadOnnx(const std::string& onnxPath);
    void allocateBuffers();
    void readTensorShapes();

    std::vector<char> readBinaryFile(const std::string& path);

private:
    TensorRTLogger logger_;

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_ = nullptr;

    std::unique_ptr<DeviceBuffer<float>> inputDevice_;
    std::unique_ptr<DeviceBuffer<float>> skyDevice_;
    std::unique_ptr<DeviceBuffer<float>> depthDevice_;

    std::vector<float> inputHost_;

    int inputC_ = 0;
    int inputH_ = 0;
    int inputW_ = 0;

    size_t inputSize_ = 0;
    size_t skySize_ = 0;
    size_t depthSize_ = 0;

    std::string inputName_ = "input";
    std::string skyName_ = "sky_logits";
    std::string depthName_ = "depth";
};
