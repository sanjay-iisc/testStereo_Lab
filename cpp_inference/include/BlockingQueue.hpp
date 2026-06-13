#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

template<typename T>
class BlockingQueue
{
public:
    explicit BlockingQueue(size_t maxSize)
        : maxSize_(maxSize)
    {}

    void push(T item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        notFull_.wait(lock, [&] {
            return queue_.size() < maxSize_ || closed_;
        });

        if(closed_)
        {
            return;
        }

        queue_.push(std::move(item));
        notEmpty_.notify_one();
    }

    bool pop(T& item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        notEmpty_.wait(lock, [&] {
            return !queue_.empty() || closed_;
        });

        if(queue_.empty())
        {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop();

        notFull_.notify_one();
        return true;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

private:
    std::queue<T> queue_;
    size_t maxSize_;
    bool closed_ = false;

    std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
};