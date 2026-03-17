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

TEST_CASE("Testing move semantics of SeamCarver constructor", "[seams]") {
    Picture pic("../images/6x5.png");
    // Checking that fields are accessible before moving
    REQUIRE(pic.width() == 6);
    REQUIRE(pic.height() == 5);

    SeamCarver sc(std::move(pic));

    // pic should've been moved into the SeamCarver instance
    REQUIRE(pic.width() == 0);
    REQUIRE(pic.height() == 0);

    // Trying to access fields in moved object should throw an out of bounds
    REQUIRE_THROWS_AS(pic.getPixel(0, 0), std::out_of_range);

    REQUIRE(sc.energy().cols == 6);
    REQUIRE(sc.energy().rows == 5);
}

TEST_CASE("Boundary energy calculation is correct", "[energy]") {
    Mat image = Mat::zeros(3, 1, CV_8UC3);
    
    // opencv uses BGR order by default.
    image.at<Vec3b>(0, 0) = Vec3b(10, 0, 0);
    image.at<Vec3b>(1, 0) = Vec3b(0, 20, 0);
    image.at<Vec3b>(2, 0) = Vec3b(0, 50, 0);

    Picture pic(image);
    SeamCarver sc(std::move(pic));

    Mat energy_matrix = sc.energy();
    for (int row = 0; row < energy_matrix.rows; ++row) {
        for (int col = 0; col < energy_matrix.cols; ++col) {
            REQUIRE_THAT(energy_matrix.at<float>(row, col), WithinAbs(1000.0, 0.1));
        }
    }
}

TEST_CASE("Dual-gradient energy calculation is correct", "[energy]") {    
    Picture pic("../images/6x5.png");
    SeamCarver sc(std::move(pic));

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

TEST_CASE("Single column vertical seam", "[seams]") {
    Picture pic("../images/1x8.png");
    SeamCarver sc(std::move(pic));

    std::stack<int> seam = sc.findVerticalSeam();

    while (!seam.empty()) {
        REQUIRE(seam.top() == 0);
        seam.pop();
    }
}

TEST_CASE("Finding the minimum vertical seam", "[seams]") {
    Picture pic("../images/6x5.png");
    SeamCarver sc(std::move(pic));

    std::stack<int> seam = sc.findVerticalSeam();
    
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