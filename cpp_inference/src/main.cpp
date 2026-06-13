#include "TensorRTModel.hpp"
#include "Postprocessor.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

bool isCameraIndex(const std::string& input)
{
    if(input.empty())
    {
        return false;
    }

    for(char c : input)
    {
        if(!std::isdigit(static_cast<unsigned char>(c)))
        {
            return false;
        }
    }

    return true;
}

bool isImageFile(const std::string& path)
{
    cv::Mat image = cv::imread(path);
    return !image.empty();
}

cv::Mat runOneFrame(TensorRTModel& model, const cv::Mat& frame)
{
    const auto totalStart = std::chrono::high_resolution_clock::now();

    std::vector<float> skyOutput;
    std::vector<float> depthOutput;
    double inferenceMs = 0.0;

    model.infer(frame, skyOutput, depthOutput, inferenceMs);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(model.inputW(), model.inputH()));

    cv::Mat skyMask;
    cv::Mat depthMap;

    Postprocessor::createSkyMask(skyOutput, model.inputW(), model.inputH(),skyMask);

    Postprocessor::createDepthMap(depthOutput, model.inputW(), model.inputH(),depthMap);

    cv::Mat display =
        Postprocessor::createVisualization(resized, skyMask, depthMap, inferenceMs);

    const auto totalEnd = std::chrono::high_resolution_clock::now();

    const double totalMs =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    const double realFps = totalMs > 0.0 ? 1000.0 / totalMs : 0.0;
    const int overlayTextX = model.inputW() + 20;

    cv::putText(display,
        "Total: " + std::to_string(totalMs) + " ms",
        cv::Point(overlayTextX, 90),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 255),
        2);

    cv::putText(display,
        "Pipeline FPS: " + std::to_string(realFps),
        cv::Point(overlayTextX, 120),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 255),
        2);

    return display;
}

void drawDisplayLoopStats(
    cv::Mat& display,
    double captureMs,
    double processMs,
    double guiMs,
    double displayLoopMs,
    int frameIndex,
    int frameStep)
{
    const double displayFps =
        displayLoopMs > 0.0 ? 1000.0 / displayLoopMs : 0.0;

    cv::putText(display,
        "Capture: " + std::to_string(captureMs) + " ms",
        cv::Point(20, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 255),
        2);

    cv::putText(display,
        "Process: " + std::to_string(processMs) + " ms",
        cv::Point(20, 60),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 255),
        2);

    cv::putText(display,
        "GUI: " + std::to_string(guiMs) + " ms",
        cv::Point(20, 90),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 255),
        2);

    cv::putText(display,
        "Loop FPS: " + std::to_string(displayFps),
        cv::Point(20, 120),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 255),
        2);

    cv::putText(display,
        "Frame: " + std::to_string(frameIndex),
        cv::Point(20, 150),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 255),
        2);

    cv::putText(display,
        "Step: " + std::to_string(frameStep),
        cv::Point(20, 180),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 255),
        2);
}

int main(int argc, char** argv)
{
    try
    {
        if(argc < 3)
        {
            std::cerr << "Usage:\n";
            std::cerr << argv[0]<< " <onnx_or_engine_path> <image_or_video_or_camera_id> [output_video]\n\n";

            std::cerr << "Examples:\n";
            std::cerr << argv[0]
                      << " ../models/multitask.onnx ../data/test.jpg\n";
            std::cerr << argv[0]
                      << " ../models/multitask.onnx ../data/test.mp4 result.mp4\n";
            std::cerr << argv[0]
                      << " ../models/multitask_fp16.engine 0\n";

            return 1;
        }

        const std::string modelPath = argv[1];
        const std::string inputPath = argv[2];

        const bool saveOutput = argc >= 4;
        const std::string outputPath = saveOutput ? argv[3] : "";

        TensorRTModel model(modelPath);

        cv::Mat image = cv::imread(inputPath);

        if(!image.empty() && !isCameraIndex(inputPath))
        {
            cv::Mat display = runOneFrame(model, image);

            cv::imshow("RGB | Sky Overlay | Depth", display);

            cv::imwrite("result.png", display);
            
            std::cout << "Saved image result: result.png\n";
            std::cout << "Press q or ESC to exit\n";

            while(true)
            {
                cv::imshow(
                    "RGB | Sky Overlay | Depth",
                    display);

                const int key = cv::waitKey(30);

                if(key == 'q' || key == 'Q' || key == 27)
                {
                    break;
                }
            }

        cv::destroyAllWindows();
        return 0;
        }

        cv::VideoCapture cap;

        if(isCameraIndex(inputPath))
        {
            cap.open(std::stoi(inputPath));
        }
        else
        {
            cap.open(inputPath);
        }

        if(!cap.isOpened())
        {
            throw std::runtime_error("Could not open input: " + inputPath);
        }

        const double sourceFps = cap.get(cv::CAP_PROP_FPS);
        const double frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
        const bool cameraInput = isCameraIndex(inputPath);

        std::cout << "Source FPS: " << sourceFps << std::endl;
        std::cout << "Source frames: " << frameCount << std::endl;
        std::cout << "Use + / - to change video frame step" << std::endl;

        cv::VideoWriter writer;
        bool writerInitialized = false;

        cv::Mat frame;
        double previousCaptureMs = 0.0;
        double previousProcessMs = 0.0;
        double previousGuiMs = 0.0;
        double previousLoopMs = 0.0;
        int frameIndex = 0;
        int frameStep = cameraInput ? 1 : 30;


        while(true)
        {
            const auto loopStart = std::chrono::high_resolution_clock::now();

            const auto captureStart = std::chrono::high_resolution_clock::now();

            if(!cap.read(frame))
            {
                break;
            }

            const auto captureEnd = std::chrono::high_resolution_clock::now();

            ++frameIndex;

            const auto processStart = std::chrono::high_resolution_clock::now();

            cv::Mat display = runOneFrame(model, frame);

            const auto processEnd = std::chrono::high_resolution_clock::now();

            drawDisplayLoopStats(
                display,
                previousCaptureMs,
                previousProcessMs,
                previousGuiMs,
                previousLoopMs,
                frameIndex,
                frameStep);

            if(saveOutput && !writerInitialized)
            {
                double fps = cap.get(cv::CAP_PROP_FPS);

                if(fps <= 1.0 || fps > 240.0)
                {
                    fps = 30.0;
                }

                int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

                writer.open(outputPath, fourcc, fps, display.size());

                if(!writer.isOpened())
                {
                    throw std::runtime_error("Could not open output video: " + outputPath);
                }

                writerInitialized = true;
            }

            if(writerInitialized)
            {
                writer.write(display);
            }

            const auto guiStart = std::chrono::high_resolution_clock::now();

            cv::imshow("RGB | Sky Overlay | Depth", display);

            int key = cv::waitKey(30);

            const auto guiEnd = std::chrono::high_resolution_clock::now();

            if(key == 27 || key == 'q')
            {
                break;
            }

            if(key == '+' || key == '=')
            {
                ++frameStep;
            }
            else if(key == '-' && frameStep > 1)
            {
                --frameStep;
            }

            if(!cameraInput)
            {
                for(int skipped = 1; skipped < frameStep; ++skipped)
                {
                    if(!cap.grab())
                    {
                        break;
                    }

                    ++frameIndex;
                }
            }

            const auto loopEnd = std::chrono::high_resolution_clock::now();

            previousCaptureMs = std::chrono::duration<double, std::milli>(captureEnd - captureStart).count();

            previousProcessMs = std::chrono::duration<double, std::milli>(processEnd - processStart).count();

            previousGuiMs = std::chrono::duration<double, std::milli>(guiEnd - guiStart).count();

            previousLoopMs = std::chrono::duration<double, std::milli>(loopEnd - loopStart).count();
        }

        cap.release();

        if(writerInitialized)
        {
            writer.release(); 
            std::cout << "Saved video result: " << outputPath << std::endl;
        }

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
