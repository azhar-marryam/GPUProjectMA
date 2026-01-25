#include <opencv2/opencv.hpp>

void cpu_grayscale(const cv::Mat& input, cv::Mat& output) {
    output.create(input.rows, input.cols, CV_8UC1);

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
            output.at<uchar>(y, x) =
                static_cast<uchar>(0.299 * pixel[2] +
                                   0.587 * pixel[1] +
                                   0.114 * pixel[0]);
        }
    }
}
