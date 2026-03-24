#pragma once
#include <picture.h>
#include <seam_carver.h>

#include <string>

class ResizeDemo {
private:
    Picture original;
    SeamCarver sc;

    // Runs removals
    // void precompute();
    // void render(int step);

public:
    ResizeDemo(const std::string& filename);
    // Immediately resize and display Picture
    void resize(int remove_cols, int remove_rows);
    void show_seams(int num_cols, int num_rows);
};
