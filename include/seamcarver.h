#pragma once
#include <stack>

#include <picture.h>
#include <opencv2/opencv.hpp>

class SeamCarver {
private:
    Picture picture;
    cv::Mat energy_matrix;
    // TODO: Consider transposition for horizontal seam finding
    // bool transposed = false;

    float calculateEnergy(int row, int col);
    void aggregateEnergy();

public:
    SeamCarver(Picture&& pic);
    const cv::Mat& energy() const;
    std::stack<int> findVerticalSeam() const;
    std::stack<int> findHorizontalSeam() const;
};