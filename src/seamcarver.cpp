#include <seamcarver.h>
#include <cmath>
#include <utility>

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