#include <resize_demo.h>
#include "opencv2/highgui.hpp"

#include <stack>
#include <iostream>
#include <numeric>
#include <stdexcept>

#define RESIZE 1
#define SHOW_SEAMS 2
#define STEP_SEAMS 3

ResizeDemo::ResizeDemo(const std::string& filename)
    : original(filename), sc(original) {}

void ResizeDemo::precompute(int cols) {
    for (int i = 0; i < cols; ++i) {
        std::stack<int> seam = sc.findVerticalSeam();
        std::stack<int> temp = seam;
        std::vector<int> indices;
    
        while (!temp.empty()) {
            indices.push_back(temp.top());
            temp.pop();
        }
        vertical_seams.push_back(indices);
        sc.removeVerticalSeam(seam);
    }
}

void ResizeDemo::render(int step) {
    for (int s = latest_step; s < step; ++s) {
        cv::Mat next(latest_frame.rows, latest_frame.cols - 1, CV_8UC3);
        for (int row = 0; row < latest_frame.rows; ++row) {
            int col = vertical_seams[s][row];
            const cv::Vec3b* src = latest_frame.ptr<cv::Vec3b>(row);
            cv::Vec3b* dst = next.ptr<cv::Vec3b>(row);
            std::copy(src, src + col, dst);
            std::copy(src + col + 1, src + latest_frame.cols, dst + col);
        }
        latest_frame = next;
    }
}

void ResizeDemo::on_trackbar(int step, void* demo) {
    ResizeDemo* self = static_cast<ResizeDemo*>(demo);
    if (step <= self->latest_step) {
        cv::setTrackbarPos("Steps", "Seams", self->latest_step);
        cv::imshow("Seams", self->latest_frame);
        return;
    }
    self->render(step);
    self->latest_step = step;
    cv::imshow("Seams", self->latest_frame);
}

void ResizeDemo::resize(int cols, int rows) {
    for (int i = 0; i < cols; ++i) {
        sc.removeVerticalSeam(sc.findVerticalSeam());
    }

    for (int i = 0; i < rows; ++i) {
        sc.removeHorizontalSeam(sc.findHorizontalSeam());
    }

    sc.picture().display();
}

// Implementing single column for now
void ResizeDemo::show_seams(int cols, int rows) {
    std::cout << rows << std::endl;
    precompute(cols);
    int height = original.height();
    std::cout << height << std::endl;
    // This requires O(N) time to erase each pixel in the remaining 2D vector.
    std::vector<std::vector<int>> remaining(height, std::vector<int>(original.width()));
    for (int row = 0; row < height; ++row) {
        std::iota(remaining[row].begin(), remaining[row].end(), 0);
    }

    // New seams are generated after each removal, so indices must be
    // mapped back to the original picture. Otherwise, they'll overlap a lot.
    for (auto seam : vertical_seams) {
        for (int row = 0; row < height; ++row) {
            int col = seam[row];
            seam[row] = remaining[row][col];
            original.setPixel(row, seam[row], cv::Vec3b(0, 0, 255));
            remaining[row].erase(remaining[row].begin() + col);
        }
    }

    original.display();
}

void ResizeDemo::step_seams(int cols) {
    precompute(cols);
    cv::namedWindow("Seams");
    latest_frame = original.toMat().clone();
    cv::imshow("Seams", latest_frame);
    cv::createTrackbar(
        "Steps",
        "Seams",
        nullptr,
        cols,
        on_trackbar,
        this
    );

    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [image filename]" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    cv::Mat img = cv::imread(image_path);
    ResizeDemo demo = ResizeDemo(image_path);

    int mode;
    std::cout << "Select mode:\n\
    [1] Resize and display\n\
    [2] Show seams \n\
    [3] Step through seams" << std::endl;
    if (!(std::cin >> mode) || mode < RESIZE || mode > STEP_SEAMS) {
        std::cerr << "Invalid mode. Please pick from the options listed" << std::endl;
        return 1;
    }

    int cols = 0, rows = 0;
    std::cout << "How many columns do you want to delete? ";
    if (!(std::cin >> cols)) {
        std::cerr << "Invalid input: #columns must be an integer" << std::endl;
        return 1;
    }
    if (cols < 0 || cols > img.cols) {
        std::cerr << "#columns must be between 0 and " << img.cols - 1 << std::endl;
        return 1;
    }
    std::cout << "How many rows do you want to delete? ";
    if (!(std::cin >> rows)) {
        std::cerr << "Invalid input: #rows must be an integer" << std::endl;
        return 1;
    }
    if (rows < 0 || rows > img.rows) {
        std::cerr << "#columns must be between 0 and " << img.rows - 1 << std::endl;
        return 1;
    }

    std::cout << "Seam carving..." << std::endl;
    switch (mode) {
        case RESIZE:
            demo.resize(cols, rows);
            break;
        case SHOW_SEAMS:
            demo.show_seams(cols, rows);
            break;
        case STEP_SEAMS:
            if (rows > 0) {
                std::cout << "Horizontal seam carving not supported for seam stepping. Proceeding with columns only." << std::endl;
            }
            demo.step_seams(cols);
            break;
        default:
            std::cerr << "Invalid mode" << std::endl;
            return 1;
    }
}
