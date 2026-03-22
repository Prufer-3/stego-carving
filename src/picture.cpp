#include "picture.h"

#include <utility>
#include <stdexcept>

#include <opencv2/core.hpp>

Picture::Picture(const Picture& other) 
    : image(other.image.clone()) {}

Picture::Picture(Picture&& other) noexcept 
    : image(std::move(other.image)) {}

Picture& Picture::operator=(const Picture& other) {
    if (this != &other) {
        image = other.image.clone();
    }
    return *this;
}

Picture& Picture::operator=(Picture&& other) noexcept {
    if (this != &other) {
        image = std::move(other.image);
    }
    return *this;
}

Picture::Picture(const std::string& filename) {
    image = cv::imread(filename, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::invalid_argument("Could not open or find the image: " + filename);
    }
}

Picture::Picture(const cv::Mat& mat) {
    if (mat.empty()) {
        throw std::invalid_argument("Picture: input cv::Mat is empty");
    }
    if (mat.type() != CV_8UC3) {
        throw std::invalid_argument("Picture: input cv::Mat must be of type CV_8UC3 (8-bit 3-channel BGR image)");
    }
    image = mat.clone();
}

int Picture::width() const {
    return image.cols;
}

int Picture::height() const {
    return image.rows;
}

cv::Vec3b Picture::getPixel(int row, int col) const {
    if (row < 0 || row >= height() || col < 0 || col >= width()) {
        throw std::out_of_range("getPixel: Coordinates out of bounds");
    }
    return image.at<cv::Vec3b>(row, col);
}

void Picture::setPixel(int row, int col, const cv::Vec3b& color) {
    if (row < 0 || row >= height() || col < 0 || col >= width()) {
        throw std::out_of_range("setPixel: Coordinates out of bounds");
    }
    image.at<cv::Vec3b>(row, col) = color;
}

void Picture::display() const {
    cv::imshow("", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

const cv::Mat& Picture::toMat() const {
    return image;
}