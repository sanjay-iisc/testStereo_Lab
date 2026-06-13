#include "TensorRTModel.hpp"
#include "Preprocessor.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace
{
void checkCuda(cudaError_t status, const std::string& message)
{
    if(status != cudaSuccess)
    {
        throw std::runtime_error(
            message + ": " + cudaGetErrorString(status));
    }
}

size_t volume(const nvinfer1::Dims& dims)
{
    size_t v = 1;

    for(int i = 0; i < dims.nbDims; ++i)
    {
        v *= static_cast<size_t>(dims.d[i]);
    }

    return v;
}
}

void TensorRTLogger::log(Severity severity, const char* msg) noexcept
{
    if(severity <= Severity::kWARNING)
    {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

TensorRTModel::TensorRTModel(const std::string& enginePath)
{
    loadEngine(enginePath);
    readTensorShapes();
    allocateBuffers();

    checkCuda(cudaStreamCreate(&stream_),"cudaStreamCreate");

    if(!context_->setTensorAddress(inputName_.c_str(), inputDevice_->data()))
    {
        throw std::runtime_error("Failed to set input tensor address");
    }

    if(!context_->setTensorAddress(skyName_.c_str(), skyDevice_->data()))
    {
        throw std::runtime_error("Failed to set sky tensor address");
    }

    if(!context_->setTensorAddress(depthName_.c_str(), depthDevice_->data()))
    {
        throw std::runtime_error("Failed to set depth tensor address");
    }
}

TensorRTModel::~TensorRTModel()
{
    if(stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

std::vector<char> TensorRTModel::readBinaryFile(
    const std::string& path)
{
    std::ifstream file(path, std::ios::binary);

    if(!file)
    {
        throw std::runtime_error("Could not open file: " + path);
    }

    file.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    return buffer;
}

void TensorRTModel::loadEngine(
    const std::string& enginePath)
{
    std::vector<char> engineData = readBinaryFile(enginePath);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));

    if(!runtime_)
    {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_.reset(
        runtime_->deserializeCudaEngine(
            engineData.data(),
            engineData.size()));

    if(!engine_)
    {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    context_.reset(engine_->createExecutionContext());

    if(!context_)
    {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }
}

void TensorRTModel::readTensorShapes()
{
    nvinfer1::Dims inputDims =
        engine_->getTensorShape(inputName_.c_str());

    nvinfer1::Dims skyDims =
        engine_->getTensorShape(skyName_.c_str());

    nvinfer1::Dims depthDims =
        engine_->getTensorShape(depthName_.c_str());

    if(inputDims.nbDims != 4)
    {
        throw std::runtime_error("Expected input tensor shape NCHW");
    }

    inputC_ = inputDims.d[1];
    inputH_ = inputDims.d[2];
    inputW_ = inputDims.d[3];

    inputSize_ = volume(inputDims);
    skySize_ = volume(skyDims);
    depthSize_ = volume(depthDims);

    std::cout << "Model input shape: "
              << inputDims.d[0] << "x"
              << inputDims.d[1] << "x"
              << inputDims.d[2] << "x"
              << inputDims.d[3] << std::endl;

    std::cout << "Sky output elements: " << skySize_ << std::endl;
    std::cout << "Depth output elements: " << depthSize_ << std::endl;
}

void TensorRTModel::allocateBuffers()
{
    inputDevice_ = std::make_unique<DeviceBuffer<float>>(inputSize_);
    skyDevice_ = std::make_unique<DeviceBuffer<float>>(skySize_);
    depthDevice_ = std::make_unique<DeviceBuffer<float>>(depthSize_);

    inputHost_.resize(inputSize_);
}

bool TensorRTModel::infer(
    const cv::Mat& inputImage,
    std::vector<float>& skyOutput,
    std::vector<float>& depthOutput,
    double& inferenceMs)
{
    Preprocessor::process(inputImage, inputW_, inputH_, inputHost_);

    skyOutput.resize(skySize_);
    depthOutput.resize(depthSize_);

    checkCuda(cudaMemcpyAsync(inputDevice_->data(), inputHost_.data(), inputSize_ * sizeof(float), cudaMemcpyHostToDevice, stream_),"cudaMemcpyAsync input");

    auto start = std::chrono::high_resolution_clock::now();

    bool success = context_->enqueueV3(stream_);

    if(!success)
    {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }

    checkCuda(cudaMemcpyAsync(skyOutput.data(), skyDevice_->data(), skySize_ * sizeof(float), cudaMemcpyDeviceToHost,stream_),"cudaMemcpyAsync sky output");

    checkCuda(cudaMemcpyAsync(depthOutput.data(),depthDevice_->data(), depthSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream_),"cudaMemcpyAsync depth output");

    checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    auto end = std::chrono::high_resolution_clock::now();

    inferenceMs =
        std::chrono::duration<double, std::milli>(end - start).count();

    return true;
}

int TensorRTModel::inputC() const
{
    return inputC_;
}

int TensorRTModel::inputH() const
{
    return inputH_;
}

int TensorRTModel::inputW() const
{
    return inputW_;
}