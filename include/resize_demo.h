#pragma once
#include <picture.h>
#include <seam_carver.h>

#include <string>
#include <vector>

class ResizeDemo {
private:
    Picture original;
    SeamCarver sc;
    std::vector<std::vector<int>> vertical_seams;
    std::vector<std::vector<int>> horizontal_seams;
    int latest_step = 0;
    cv::Mat latest_frame;

    // Runs removals and stores removed seams in a vector
    void precompute(int cols, int rows);
    // Reconstructs picture at different steps based on recorded removed seams
    void render(int step);
    // opencv::Trackbar callback
    static void on_trackbar(int pos, void* demo);

public:
    ResizeDemo(const std::string& filename);
    // Immediately resize and display Picture
    void resize(int remove_cols, int remove_rows);
    // Show seams to be removed as red lines on the original Picture
    void show_seams(int num_cols, int num_rows);
    // Displays an interable interface showing every seam during deletion
    void step_seams(int num_cols, int num_rows);
};
