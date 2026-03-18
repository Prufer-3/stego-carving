#include <stack>
#include <utility>
#include <stdexcept>
#include <algorithm>

#include <picture.h>
#include <seamcarver.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace Catch::Matchers;

TEST_CASE("Boundary energy calculation is correct", "[energy]") {
    Mat image = Mat::zeros(3, 1, CV_8UC3);
    
    // opencv uses BGR order by default.
    image.at<Vec3b>(0, 0) = Vec3b(10, 0, 0);
    image.at<Vec3b>(1, 0) = Vec3b(0, 20, 0);
    image.at<Vec3b>(2, 0) = Vec3b(0, 50, 0);

    Picture pic(image);
    SeamCarver sc(pic);

    Mat energy_matrix = sc.energy();
    for (int row = 0; row < energy_matrix.rows; ++row) {
        for (int col = 0; col < energy_matrix.cols; ++col) {
            REQUIRE_THAT(energy_matrix.at<float>(row, col), WithinAbs(1000.0, 0.1));
        }
    }
}

TEST_CASE("Dual-gradient energy calculation is correct", "[energy]") {    
    Picture pic("../images/6x5.png");
    SeamCarver sc(pic);

    Mat energy_matrix = sc.energy();
    // Values from https://coursera.cs.princeton.edu/algs4/assignments/seam/specification.php
    Mat expected = (Mat_<float>(5,6) << 
        1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f,
        1000.0f, 237.35f, 151.02f, 234.09f, 107.89f, 1000.0f,
        1000.0f, 138.69f, 228.10f, 133.07f, 211.51f, 1000.0f,
        1000.0f, 153.88f, 174.01f, 284.01f, 194.50f, 1000.0f,
        1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f
    );

    for (int row = 0; row < energy_matrix.rows; ++row) {
        for (int col = 0; col < energy_matrix.cols; ++col) {
            float val = energy_matrix.at<float>(row, col);
            float expected_val = expected.at<float>(row, col);
            INFO("Mismatch at row: " << row << " col: "<< col << "\nExpected: " << expected_val << " Got: " << val);
            REQUIRE_THAT(val, WithinAbs(expected_val, 0.1));
        }
    }
}

TEST_CASE("Mutating source Picture doesn't affect SeamCarver energy", "[energy]") {    
    Picture pic("../images/6x5.png");
    SeamCarver sc(pic);

    Mat pre_energy = sc.energy();

    // Overwriting all pixels in original pic
    for (int row = 0; row < pic.height(); ++row) {
        for (int col = 0; col < pic.width(); ++col) {
            pic.setPixel(row, col, Vec3b(0, 255, 0));
        }
    }

    Mat post_energy = sc.energy();

    Mat difference;
    cv::bitwise_xor(pre_energy, post_energy, difference);
    REQUIRE(cv::countNonZero(difference) == 0);
}

TEST_CASE("Single column vertical seam", "[seams]") {
    Picture pic("../images/1x8.png");
    SeamCarver sc(pic);

    std::stack<int> seam = sc.findVerticalSeam();

    REQUIRE(seam.size() == 8);
    while (!seam.empty()) {
        REQUIRE(seam.top() == 0);
        seam.pop();
    }
}

TEST_CASE("Single row vertical seam", "[seams]") {
    Picture pic("../images/8x1.png");
    SeamCarver sc(pic);

    std::stack<int> seam = sc.findVerticalSeam();

    REQUIRE(seam.size() == 1);
    REQUIRE(seam.top() == 0);
}

TEST_CASE("Finding the minimum vertical seam", "[seams]") {
    Picture pic("../images/6x5.png");
    SeamCarver sc(pic);

    std::stack<int> seam = sc.findVerticalSeam();
    
    REQUIRE(seam.size() == 5);
    // A bit inelegant, but this is just a hard-coded seam path.
    REQUIRE(seam.top() == 4);
    seam.pop();
    REQUIRE(seam.top() == 4);
    seam.pop();
    REQUIRE(seam.top() == 3);
    seam.pop();
    REQUIRE(seam.top() == 2);
    seam.pop();
    REQUIRE(seam.top() == 1);
    seam.pop();
}

TEST_CASE("Single row horizontal seam", "[seams]") {
    Picture pic("../images/8x1.png");
    SeamCarver sc(pic);

    std::stack<int> seam = sc.findHorizontalSeam();

    REQUIRE(seam.size() == 8);
    while (!seam.empty()) {
        REQUIRE(seam.top() == 0);
        seam.pop();
    }
}

TEST_CASE("Single column horizontal seam", "[seams]") {
    Picture pic("../images/1x8.png");
    SeamCarver sc(pic);

    std::stack<int> seam = sc.findHorizontalSeam();

    REQUIRE(seam.size() == 1);
    REQUIRE(seam.top() == 0);
}

TEST_CASE("Finding the minimum horizontal seam", "[seams]") {
    Picture pic("../images/6x5.png");
    SeamCarver sc(pic);

    std::stack<int> seam = sc.findHorizontalSeam();

    REQUIRE(seam.size() == 6);
    REQUIRE(seam.top() == 2);
    seam.pop();
    REQUIRE(seam.top() == 2);
    seam.pop();
    REQUIRE(seam.top() == 1);
    seam.pop();
    REQUIRE(seam.top() == 2);
    seam.pop();
    REQUIRE(seam.top() == 1);
    seam.pop();
    REQUIRE(seam.top() == 0);
    seam.pop();
}