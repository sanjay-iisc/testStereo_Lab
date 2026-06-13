#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

template<typename T>
class DeviceBuffer
{

public:

    explicit DeviceBuffer(size_t count): ptr_(nullptr), count_(count)
    {
        cudaError_t err = cudaMalloc(&ptr_, count_ * sizeof(T));

        if(err != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    ~DeviceBuffer()
    {
        if(ptr_)
        {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // move constrcutor
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    // move assignment
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if(this != &other)
        {
            if(ptr_)
            {
                cudaFree(ptr_);
            }

            ptr_ = other.ptr_;
            count_ = other.count_;

            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        return *this;
    }

    T* data()
    {
        return ptr_;
    }

    const T* data() const
    {
        return ptr_;
    }

    size_t size() const
    {
        return count_;
    }

private:
    T* ptr_; // Points to GPU
    size_t count_; // How many T elements
};
