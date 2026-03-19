#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class Picture {
private:
    cv::Mat image;

public:
    Picture(const std::string& filename);
    Picture(const cv::Mat& mat);

    int width() const;
    int height() const;

    cv::Vec3b getPixel(int row, int col) const;
    void setPixel(int row, int col, const cv::Vec3b& color);

    void display();
};