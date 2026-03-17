#include <picture.h>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
using namespace std;
using namespace cv;

TEST_CASE("Picture class returns correct RGB values", "[picture]") {
    Mat image = imread("../images/1x8.png");

    Picture pic(image);
    // opencv::Vec3b returns color in BGR order for some ungodly reason.
    REQUIRE(pic.getPixel(0, 0) == Vec3b(10, 13, 5));
    REQUIRE(pic.getPixel(1, 0) == Vec3b(3, 2, 13));
    REQUIRE(pic.getPixel(2, 0) == Vec3b(6, 0, 5));
    REQUIRE(pic.getPixel(3, 0) == Vec3b(8, 9, 10));
    REQUIRE(pic.getPixel(4, 0) == Vec3b(5, 4, 0));
    REQUIRE(pic.getPixel(5, 0) == Vec3b(13, 12, 10));
    REQUIRE(pic.getPixel(6, 0) == Vec3b(0, 12, 2));
    REQUIRE(pic.getPixel(7, 0) == Vec3b(0, 5, 0));
}

TEST_CASE("Setting individual pixels in Picture class", "[picture]") {
    // Initial 2 by 2 pixel image set to black
    Mat image(2, 2, CV_8UC3, Scalar(0, 0, 0));
    Picture pic(image);
    
    REQUIRE(pic.getPixel(0, 0) == Vec3b(0, 0, 0));

    Vec3b rgb(255, 215, 115);
    pic.setPixel(0, 0, rgb);

    REQUIRE(pic.getPixel(0, 0) == rgb);
}

TEST_CASE("Attempting to set an out of bounds pixel", "[picture]") {
    Mat image(2, 2, CV_8UC3, Scalar(0, 0, 0));
    Picture pic(image);

    Vec3b rgb(0, 0, 255);

    REQUIRE_THROWS_AS(pic.setPixel(10, 10, rgb), std::out_of_range);
}