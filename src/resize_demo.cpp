#include <resize_demo.h>

#include <iostream>
#include <stdexcept>

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

int main(int argc, char const *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " [image filename] [num cols to remove] [num rows to remove]" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    ResizeDemo demo = ResizeDemo(image_path);

    try {
        int cols = std::stoi(argv[2]);
        if (cols < 0) {
            throw std::invalid_argument("remove_cols must be positive");
        }

        int rows = std::stoi(argv[3]);
        if (rows < 0) {
            throw std::invalid_argument("remove_rows must be positive");
        }

        demo.resize(cols, rows);
    } catch (...) {
        std::cerr << "Invalid row and col arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [image filename] [num cols to remove] [num rows to remove]" << std::endl;
        return -1;
    }
}
