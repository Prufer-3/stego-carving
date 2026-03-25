#include <resize_demo.h>
#include "opencv2/highgui.hpp"

#include <stack>
#include <iostream>
#include <stdexcept>

#define RESIZE 1
#define SHOW_SEAMS 2
#define STEP_SEAMS 3

ResizeDemo::ResizeDemo(const std::string& filename)
    : original(filename), sc(original) {}

void ResizeDemo::precompute(int cols, int rows) {
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

    for (int i = 0; i < rows; ++i) {
        std::stack<int> seam = sc.findHorizontalSeam();
        std::stack<int> temp = seam;
        std::vector<int> indices;
    
        while (!temp.empty()) {
            indices.push_back(temp.top());
            temp.pop();
        }
        horizontal_seams.push_back(indices);
        sc.removeHorizontalSeam(seam);
    }
}

void ResizeDemo::render(int step) {
    cv::Mat frame = original.toMat().clone();
    for (int s = 0; s < step; ++s) {
        cv::Mat next(frame.rows, frame.cols - 1, CV_8UC3);
        for (int row = 0; row < frame.rows; ++row) {
            int col = vertical_seams[s][row];
            const cv::Vec3b* src = frame.ptr<cv::Vec3b>(row);
            cv::Vec3b* dst = next.ptr<cv::Vec3b>(row);
            std::copy(src, src + col, dst);
            std::copy(src + col + 1, src + frame.cols, dst + col);
        }
        frame = next;
    }
    cv::imshow("Seams", frame);
}

void ResizeDemo::on_trackbar(int step, void* demo) {
    static_cast<ResizeDemo*>(demo)->render(step);
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
    std::cout << cols << rows << std::endl;
    std::stack seam = sc.findVerticalSeam();
    Picture pic = sc.picture();

    for (int row = 0; row < pic.height(); ++row) {
        pic.setPixel(row, seam.top(), cv::Vec3b(0, 0, 255));
        seam.pop();
    }

    pic.display();
}

void ResizeDemo::step_seams(int cols, int rows) {
    this->precompute(cols, rows);
    cv::namedWindow("Seams");
    render(0);
    cv::createTrackbar(
        "Steps",
        "Seams",
        nullptr,
        cols + rows,
        on_trackbar,
        this
    );

    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char const *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " [image filename] [num cols to remove] [num rows to remove]" << std::endl;
        return -1;
    }

    int cols = 0, rows = 0;

    try {
        cols = std::stoi(argv[2]);
        if (cols < 0) {
            throw std::invalid_argument("remove_cols must be positive");
        }

        rows = std::stoi(argv[3]);
        if (rows < 0) {
            throw std::invalid_argument("remove_rows must be positive");
        }
    } catch (...) {
        std::cerr << "Invalid row and col arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [image filename] [num cols to remove] [num rows to remove]" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    ResizeDemo demo = ResizeDemo(image_path);

    int mode;
    std::cout << "Select mode:\n\
    [1] Resize and display immediately\n\
    [2] Show seams \n\
    [3] Step through seams" << std::endl;
    std::cin >> mode;

    switch (mode) {
        case RESIZE:
            demo.resize(cols, rows);
            break;
        case SHOW_SEAMS:
            demo.show_seams(cols, rows);
            break;
        case STEP_SEAMS:
            demo.step_seams(cols, rows);
            break;
        default:
            std::cerr << "Invalid mode" << std::endl;
            return -1;
    }
}
