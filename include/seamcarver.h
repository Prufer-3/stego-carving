#pragma once

#include <opencv2/opencv.hpp>
#include <picture.h>

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
};