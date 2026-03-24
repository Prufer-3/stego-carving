#include <resize_demo.h>

#include <stack>
#include <iostream>
#include <stdexcept>

#define RESIZE 1
#define SHOW_SEAM 2

ResizeDemo::ResizeDemo(const std::string& filename)
    : original(filename), sc(original) {}

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

int main(int argc, char const *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " [image filename] [num cols to remove] [num rows to remove]" << std::endl;
        return -1;
    }

    int cols = 0, rows = 0;

    try {
        int cols = std::stoi(argv[2]);
        if (cols < 0) {
            throw std::invalid_argument("remove_cols must be positive");
        }

        int rows = std::stoi(argv[3]);
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
    std::cout << "Select mode:\n \
    [1] Resize and display\n \
    [2] Show seam" << std::endl;
    std::cin >> mode;

    switch (mode) {
        case RESIZE:
            demo.resize(cols, rows);
            break;
        case SHOW_SEAM:
            demo.show_seams(cols, rows);
            break;
        default:
            std::cerr << "Invalid mode" << std::endl;
            return -1;
    }
}
