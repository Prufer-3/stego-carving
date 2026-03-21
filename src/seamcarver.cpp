#include <seamcarver.h>

#include <cmath>
#include <utility>
#include <algorithm>

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

SeamCarver::SeamCarver(const Picture pic) : pic(pic) {
    width = pic.width();
    height = pic.height();
    aggregateEnergy();
}

void SeamCarver::aggregateEnergy() {
    energy_matrix = cv::Mat(height, width, CV_32FC1);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            energy_matrix.at<float>(row, col) = calculateEnergy(row, col);
        }
    }
}

// Calculate dual-gradient energy for a specific pixel
float SeamCarver::calculateEnergy(int row, int col) {
    if (col == 0 || row == 0) return 1000.0f;
    if (col == width - 1 || row == height - 1) return 1000.0f;

    int xGradSquared = gradientSquared(
        pic.getPixel(row, col - 1), pic.getPixel(row, col + 1)
    );
    int yGradSquared = gradientSquared(
        pic.getPixel(row - 1, col), pic.getPixel(row + 1, col)
    );

    return std::sqrt(static_cast<float> (xGradSquared + yGradSquared));
}

void SeamCarver::transpose() {
    energy_matrix = energy_matrix.t();
    transposed = !transposed;
    std::swap(width, height);
}

const cv::Mat SeamCarver::energy() const {
    if (transposed) {
        return energy_matrix.t();
    }
    return energy_matrix;
}

std::stack<int> SeamCarver::findSeam() const {
    cv::Mat energy_sum(height, width, CV_32FC1);
    cv::Mat parent(height, width, CV_32SC1);
    energy_sum.row(0).setTo(cv::Scalar(1000.0f));
    parent.row(0).setTo(cv::Scalar(-1));

    // DP algorithm to calculate all possible seam energy values.
    for (int row = 1; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int parent_index = col;
            float min_parent_sum = energy_sum.at<float>(row - 1, parent_index);
            if (col != 0 && energy_sum.at<float>(row - 1, col - 1) < min_parent_sum) {
                min_parent_sum = energy_sum.at<float>(row - 1, col - 1);
                parent_index = col - 1;
            }
            if (col != width - 1 && energy_sum.at<float>(row - 1, col + 1) < min_parent_sum) {
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
    for (int row = height - 1; row >= 0; --row) {
        seam.push(min_sum_index);
        min_sum_index = parent.at<int>(row, min_sum_index);
    }

    return seam;
}

std::stack<int> SeamCarver::findVerticalSeam() {
    if (transposed) transpose();
    return findSeam();
}

void SeamCarver::removeVerticalSeam(std::stack<int> seam) {
    if (transposed) transpose();
    checkSeam(seam);
    removeSeam(seam);
}

std::stack<int> SeamCarver::findHorizontalSeam() {
    if (!transposed) transpose();
    return findSeam();
}

void SeamCarver::removeHorizontalSeam(std::stack<int> seam) {
    if (!transposed) transpose();
    checkSeam(seam);
    removeSeam(seam);
}

void SeamCarver::checkSeam(const std::stack<int>& seam) const {
    if (width == 1) {
        throw std::domain_error("Deletion of the last seam in image is not allowed");
    }
    // Width and height are a bit of a misnomer here
    // But they are coupled with the current orientation of the Picture
    if (static_cast<int>(seam.size()) != height) {
        throw std::invalid_argument("Mismatched seam length");
    }

    std::stack<int> seam_copy = seam;

    int prev = seam_copy.top();
    while (!seam_copy.empty()) {
        int curr = seam_copy.top();
        if (curr < 0 || curr > width - 1) {
            throw std::invalid_argument("Seam index out of bounds");
        }
        if (std::abs(curr - prev) > 1) {
            throw std::invalid_argument("Seam must be continuous");
        }
        prev = curr;
        seam_copy.pop();
    }
}

void SeamCarver::removeSeam(std::stack<int> seam) {
    cv::Mat new_image(height, width - 1, CV_8UC3);

    for (int row = 0; row < height; ++row) {
        int i = seam.top(); seam.pop();
        const cv::Vec3b* src = pic.toMat().ptr<cv::Vec3b>(row);
        cv::Vec3b* dst = new_image.ptr<cv::Vec3b>(row);

        std::copy(src, src + i, dst);
        std::copy(src + i + 1, src + width, dst + i);
    }

    --width;
    pic = Picture(new_image);
    aggregateEnergy();
}

const Picture SeamCarver::picture() const {
    return pic;
}