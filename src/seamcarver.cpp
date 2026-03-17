#include <seamcarver.h>

#include <cmath>
#include <utility>
#include <algorithm>

// Add private helper functions here
namespace {
    // BGR values in Vec3b for pixels
    int gradientSquared(cv::Vec3b px1, cv::Vec3b px2) {
        int blueDelta = static_cast<int>(px1[0]) - static_cast<int>(px2[0]);
        int greenDelta = static_cast<int>(px1[1]) - static_cast<int>(px2[1]);
        int redDelta = static_cast<int>(px1[2]) - static_cast<int>(px2[2]);
        
        return (blueDelta * blueDelta) + 
               (greenDelta * greenDelta) + 
               (redDelta * redDelta);
    }
}

SeamCarver::SeamCarver(Picture&& pic) : picture(std::move(pic)) {
    aggregateEnergy();
}

void SeamCarver::aggregateEnergy() {
    energy_matrix = cv::Mat(picture.height(), picture.width(), CV_32FC1);
    for (int row = 0; row < picture.height(); ++row) {
        for (int col = 0; col < picture.width(); ++col) {
            energy_matrix.at<float>(row, col) = calculateEnergy(row, col);
        }
    }
}

// Calculate dual-gradient energy for a specific pixel
float SeamCarver::calculateEnergy(int row, int col) {
    if (col == 0 || row == 0) return 1000.0f;
    if (col == picture.width() - 1 || row == picture.height() - 1) return 1000.0f;

    int xGradSquared = gradientSquared(
        picture.getPixel(row, col - 1), picture.getPixel(row, col + 1)
    );
    int yGradSquared = gradientSquared(
        picture.getPixel(row - 1, col), picture.getPixel(row + 1, col)
    );

    return std::sqrt(static_cast<float> (xGradSquared + yGradSquared));
}

const cv::Mat& SeamCarver::energy() const {
    return energy_matrix;
}

std::stack<int> SeamCarver::findVerticalSeam() const {
    cv::Mat energy_sum(picture.height(), picture.width(), CV_32FC1);
    cv::Mat parent(picture.height(), picture.width(), CV_32SC1);
    energy_sum.row(0).setTo(cv::Scalar(1000.0f));
    parent.row(0).setTo(cv::Scalar(-1));

    for (int row = 1; row < picture.height(); ++row) {
        for (int col = 0; col < picture.width(); ++col) {
            int parent_index = col;
            float min_parent_sum = energy_sum.at<float>(row - 1, parent_index);
            if (col != 0 && energy_sum.at<float>(row - 1, col - 1) < min_parent_sum) {
                min_parent_sum = energy_sum.at<float>(row - 1, col - 1);
                parent_index = col - 1;
            }
            if (col != picture.width() - 1 && energy_sum.at<float>(row - 1, col + 1) < min_parent_sum) {
                min_parent_sum = energy_sum.at<float>(row - 1, col + 1);
                parent_index = col + 1;
            }

            parent.at<int>(row, col) = parent_index;

            energy_sum.at<float>(row, col) = energy_matrix.at<float>(row, col) + min_parent_sum;
        }
    }
    
    // Finding the location of the smallest energy sum
    cv::Point minLoc;
    cv::minMaxLoc(energy_sum.row(energy_sum.rows - 1), nullptr, nullptr, &minLoc, nullptr);
    int min_sum_index = minLoc.x;

    // Backtracking to find indices which make up seam. Top to bottom of Picture.
    std::stack<int> seam;
    for (int row = picture.height() - 1; row >= 0; --row) {
        seam.push(min_sum_index);
        min_sum_index = parent.at<int>(row, min_sum_index);
    }

    return seam;
}