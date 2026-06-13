#include "TensorRTModel.hpp"

int main()
{
    TensorRTModel model("model.engine");

    cv::VideoCapture cap("video.mp4");

    while(cap.read(frame))
    {
        model.infer(frame, sky, depth);

        auto display = Postprocessor::createOverlay(frame, sky, depth);

        cv::imshow("result", display);
    }
}