#pragma once
#include <stack>

#include <picture.h>
#include <opencv2/opencv.hpp>

class SeamCarver {
private:
    // Picture picture;
    int width, height;
    cv::Mat energy_matrix;
    bool transposed = false;

    float calculateEnergy(const Picture& pic, int row, int col);
    void aggregateEnergy(const Picture& pic);
    std::stack<int> findSeam() const;
    void transpose();

public:
    SeamCarver(const Picture& pic);
    const cv::Mat energy() const;
    std::stack<int> findVerticalSeam();
    std::stack<int> findHorizontalSeam();
};