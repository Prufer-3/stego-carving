#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class Picture {
private:
    cv::Mat image;

public:
    Picture(const std::string& filename);
    Picture(const cv::Mat& mat);

    Picture(const Picture&) = delete;
    Picture& operator=(const Picture&) = delete;

    Picture(Picture&&) noexcept = delete;
    Picture& operator=(Picture&&) noexcept = delete;

    int width() const;
    int height() const;

    cv::Vec3b getPixel(int row, int col) const;
    void setPixel(int row, int col, const cv::Vec3b& color);

    void display();
};