#pragma once
#include <string>

#include <opencv2/opencv.hpp>

class Picture {
private:
    cv::Mat image;

public:
    // Rule of 5
    // Copy constructor
    Picture(const Picture& other);
    // Move constructor
    Picture(Picture&& other) noexcept;
    // Copy assignment
    Picture& operator=(const Picture& other);
    // Move assignment
    Picture& operator=(Picture&& other) noexcept;
    // Destructor
    ~Picture() = default;

    Picture(const std::string& filename);
    Picture(const cv::Mat& mat);

    int width() const;
    int height() const;

    cv::Vec3b getPixel(int row, int col) const;
    void setPixel(int row, int col, const cv::Vec3b& color);

    void display() const;
    const cv::Mat& toMat() const;
};