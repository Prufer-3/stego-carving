#pragma once
#include <stack>

#include <picture.h>
#include <opencv2/opencv.hpp>

class SeamCarver {
private:
    Picture pic;
    int width, height;
    cv::Mat energy_matrix;
    bool transposed = false;

    float calculateEnergy(int row, int col);
    void aggregateEnergy();
    std::stack<int> findSeam() const;
    void transpose();
    void removeSeam(std::stack<int>& seam);
    void checkSeam(const std::stack<int>& seam) const;

public:
    SeamCarver(Picture pic);
    const cv::Mat energy() const;
    std::stack<int> findVerticalSeam();
    void removeVerticalSeam(std::stack<int> seam);
    std::stack<int> findHorizontalSeam();
    void removeHorizontalSeam(std::stack<int> seam);
    const Picture picture() const;
};